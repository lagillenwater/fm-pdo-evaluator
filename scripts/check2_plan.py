"""Stage 1 of the Check-2 array: compute everything that is shared, once.

The Check-2 grid is embarrassingly parallel -- roughly 540 independent penalized fits for the
noise draws plus ~57 independent scored rows -- so it belongs in an array job rather than a
three-hour serial run. One thing blocks a naive per-representation split, and it is the reason
this stage exists.

``restrict_representation_support`` is CROSS-representation by construction: it drops a drug if
ANY representation lacks it, and restricts each surviving drug to the lines present in EVERY
representation. So the (patient, drug) support a row is scored on depends on every other
representation in the grid. An array task that built only its own features would compute a
different support, and every row would be scored on a different set -- silently, with the table
still looking well-formed. That is exactly the defect ``restrict_common_support`` exists to
prevent, and it is what produced the n=202-vs-n=150 mismatch in August.

So: compute the shared quantities here, pin them to disk, and let the scatter stage restrict to
what this stage decided rather than deciding for itself.

Writes, under ``--out-dir``:
  plan.json                     resolved parameters, git sha, representation names, fold map
  support/<rep>/<drug>.parquet  the restricted feature frame each task will score
  design_target.parquet         the labels, already filtered to the target drugs

The features are persisted rather than recomputed per task for two reasons: rebuilding a
leave-one-line-out source is the expensive part of the setup, and persisting what was actually
scored is itself provenance -- a reader can check the inputs, not just the outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from fmharness.check2 import (
    RANDOM_DRAWS,
    load_line_matrix,
    random_control_representation,
    repr_by_drug,
    restrict_representation_support,
    seed_for_name,
)
from fmharness.data.loaders import load_tranche
from fmharness.deltas import build_generated_deltas, load_pert_map, loo_baseline_source
from fmharness.evaluation import build_sample_design
from fmharness.signatures import load_hallmark


def git_sha() -> str:
    """The commit these features were built at."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    """Build every representation, fix the shared support, and persist it."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deltas-bundle", default="tahoe_deltas")
    ap.add_argument("--generated-dir", default="generated_agg")
    ap.add_argument("--query-baseline", default="tahoe_query_baseline.h5ad")
    ap.add_argument("--pert-map", default="context_by_drug/pert_to_cid.tsv")
    ap.add_argument("--stack-emb", nargs="*", default=[], help="label=path pairs")
    ap.add_argument("--auc-tranche", default="gdsc2_sarcoma")
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--random-draws", type=int, default=RANDOM_DRAWS)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    bdir = Path(args.deltas_bundle)
    real_delta = pd.read_parquet(bdir / "real_delta.parquet")
    real_key = pd.read_parquet(bdir / "real_key.parquet")
    base = pd.read_parquet(bdir / "base.parquet")
    hvg = pd.Index(real_delta.var(axis=0).sort_values(ascending=False).index[: args.n_hvg])
    _, design = build_sample_design(
        load_tranche(args.auc_tranche, repo), "all", "auc", drug_key="pubchem_cid"
    )
    load_hallmark(repo / "data/static/hallmark_signatures.gmt")  # fail early if absent

    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "additive": loo_baseline_source("additive", real_delta, real_key, base, k=args.k),
        "knn": loo_baseline_source("knn", real_delta, real_key, base, k=args.k),
        "pca": loo_baseline_source("pca", real_delta, real_key, base, k=args.k),
        "nmf": loo_baseline_source("nmf", real_delta, real_key, base, k=args.k),
        "measured_delta": (real_delta.copy(), real_key.copy()),
    }
    if args.generated_dir:
        sources["stack"] = build_generated_deltas(
            Path(args.generated_dir), Path(args.query_baseline), load_pert_map(Path(args.pert_map))
        )

    uniq_lines = sorted(set(real_key["patient"].astype(str)))
    n_folds = max(1, min(args.folds, len(uniq_lines)))
    fold_of = {ln: i % n_folds for i, ln in enumerate(uniq_lines)}
    target_drugs = sorted({str(d) for d in real_key["drug"]})
    design_target = design[design["drug"].astype(str).isin(target_drugs)]

    base_hvg = base.reindex(columns=hvg).fillna(0.0)
    representations: dict[str, object] = {
        "expr": lambda _d: base_hvg,
        "prior": lambda _d: pd.DataFrame(0.0, index=base_hvg.index, columns=pd.Index(["const"])),
    }
    for name, (d, kk) in sources.items():
        representations[name] = repr_by_drug(d, kk, hvg)
    for spec in args.stack_emb:
        label, _, path = spec.partition("=")
        representations[label.strip()] = (lambda e: lambda _d: e)(load_line_matrix(Path(path)))

    # The displayed noise row per representation, so the scatter stage does not have to know
    # which seeds the serial version used.
    for name in [n for n in representations if n != "prior"]:
        representations[f"{name}_random"] = random_control_representation(
            representations[name], target_drugs, seed=seed_for_name(name)
        )

    restricted = restrict_representation_support(representations, design_target)

    out = args.out_dir
    (out / "support").mkdir(parents=True, exist_ok=True)
    for name, per_drug in restricted.items():
        rdir = out / "support" / name
        rdir.mkdir(parents=True, exist_ok=True)
        for drug, frame in per_drug.items():
            frame.to_parquet(rdir / f"{_safe(drug)}.parquet")
    design_target.to_parquet(out / "design_target.parquet")

    plan = {
        "git_sha": git_sha(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "representations": sorted(restricted),
        "drugs": sorted({d for pd_ in restricted.values() for d in pd_}),
        "uniq_lines": uniq_lines,
        "n_folds": n_folds,
        "fold_of": fold_of,
        "random_draws": args.random_draws,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"plan: {len(plan['representations'])} representations x {len(plan['drugs'])} drugs")
    print(f"      support pinned across ALL representations, written to {out}/support")
    print(f"      {len(plan['representations'])} array tasks needed (0..{len(plan['representations']) - 1})")


def _safe(drug: str) -> str:
    """Filesystem-safe drug id; drug names carry slashes and spaces."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(drug))


if __name__ == "__main__":
    main()
