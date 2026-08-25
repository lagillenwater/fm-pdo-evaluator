"""Check 2 (end-to-end GDSC2 AUC prediction) through the harness-core registries, leakage-aware.

Filters the GDSC2 AUC ``design`` frame (not the Tahoe ``real_key``) via the same
``filter_leakage``/``PregeneratedStackGenerator``/``LeakageQueryable`` composition
``check1_registry_driver.py`` uses -- ``design`` is Check 2's actual evaluation set. Every
representation (``expr``, ``additive``, ``knn``, ``pca``, ``nmf``, ``stack``, any
``--stack-emb``) is scored via a merge/groupby against ``design`` downstream (inside
``fmharness.check2.score_check2``); filtering it once bounds the leakage-safe universe every
representation draws from, but does NOT by itself equalize each representation's own native
coverage of that universe (2026-08-21 audit finding: additive/knn/pca/nmf broadcast to
essentially every drug while stack only covers its own generated set) -- same-pair-count
parity across the table is now ``score_check2``'s own job (``restrict_common_support`` for
the fixed-readout rows, ``restrict_representation_support`` for the penalized grid), not a
side effect of filtering ``design``. Delta sources still stay built from the full,
unfiltered Tahoe triple. See
docs/superpowers/specs/2026-08-13-check2-leakage-aware-drug-aligned-design.md for the full
design rationale.

Run (see check1_registry_driver.py's own --context-vs---deltas-bundle caveat -- it applies here
identically: prefer --deltas-bundle unless you have specifically verified --context agrees):
  uv run python scripts/check2_registry_driver.py \\
      --deltas-bundle tahoe_deltas \\
      --query-baseline tahoe_query.h5ad \\
      --generated-dir generated_sciplex \\
      --pert-map context_by_drug/pert_to_cid.tsv \\
      --checkpoint-label drug-aligned \\
      --corpus-lines ACH-000681 \\
      --corpus-drugs 6918289,11626560,104741,11707110,3385
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import anndata as ad
import pandas as pd

from fmharness.check2 import FIXED_READOUTS, PENALTY_NAMES, load_line_matrix, score_check2
from fmharness.data.loaders import load_tranche
from fmharness.deltas import (
    build_generated_deltas,
    build_tahoe_deltas,
    learned_gene_panel,
    load_pert_map,
    loo_baseline_source,
)
from fmharness.evaluation import build_sample_design
from fmharness.leakage import (
    corpus_declared_partially,
    filter_leakage,
    ground_truth_source_declared_ambiguously,
    parse_corpus_set,
)
from fmharness.models.stack_generator import PregeneratedStackGenerator
from fmharness.schema import TaskSignal
from fmharness.signatures import load_hallmark


def _rel(repo: Path, p: str) -> Path:
    """Resolve ``p`` against the repo root unless it is already absolute."""
    q = Path(p)
    return q if q.is_absolute() else repo / q


def run_check2(
    real_delta: pd.DataFrame,
    real_key: pd.DataFrame,
    base: pd.DataFrame,
    *,
    query_baseline: Path,
    generated_dir: Path,
    pert_to_drug: dict[str, str],
    checkpoint_label: str,
    hallmark_path: Path,
    auc_design: pd.DataFrame,
    n_hvg: int = 2000,
    k: int = 10,
    fixed_methods: tuple[str, ...] = FIXED_READOUTS,
    penalties: tuple[str, ...] = PENALTY_NAMES,
    folds: int = 5,
    stack_emb: dict[str, pd.DataFrame] | None = None,
    n_permutations: int = 1000,
    pretraining_lines: set[str] | None = None,
    pretraining_drugs: set[str] | None = None,
    task_signal_in_pretrain: TaskSignal = "none",
) -> pd.DataFrame:
    """Check-2 table: fixed-signature readouts + representation grid, leakage-filtered.

    ``real_delta``/``real_key``/``base`` are the Tahoe ground-truth triple, used UNFILTERED to
    build every delta source (additive/knn/pca/nmf/stack) -- a source's prediction for a
    contaminated pair sitting unused in the sources dict is harmless; it never gets scored once
    ``auc_design`` excludes that pair. ``auc_design`` is the GDSC2 (patient, drug, y) AUC label
    frame (``build_sample_design``'s own output) -- THIS is what gets leakage-filtered, once,
    before any scoring: every representation is scored via a merge/groupby against it
    downstream, so filtering it once uniformly restricts every representation (not just
    ``stack``) to the same surviving pairs. ``pretraining_lines``/``pretraining_drugs`` (both
    default ``None``) declare the Stack checkpoint's measured pretraining overlap, exactly as
    ``check1_registry_driver.run_check1`` -- both must be given together for filtering to
    activate.
    """
    model = PregeneratedStackGenerator(
        generated_dir,
        pert_to_drug,
        checkpoint_label=checkpoint_label,
        pretraining_lines=pretraining_lines,
        pretraining_drugs=pretraining_drugs,
        task_signal_in_pretrain=task_signal_in_pretrain,
    )
    filtered_design, profile = filter_leakage(auc_design, model)
    if profile.basis == "measured":
        print(
            f"Check-2 leakage filter: basis=measured, "
            f"doubly_exposed_frac={profile.doubly_exposed_frac:.3f}, "
            f"line_overlap_frac={profile.line_overlap_frac:.3f}, "
            f"drug_overlap_fraction={profile.drug_overlap_fraction:.3f}"
        )
    else:
        print(f"Check-2 leakage filter: basis={profile.basis} (no corpus declared -- unfiltered)")

    hallmark = load_hallmark(hallmark_path)
    hvg = pd.Index(real_delta.var(axis=0).sort_values(ascending=False).index[:n_hvg])
    learned_genes = learned_gene_panel(real_delta, hallmark_path, n_hvg=n_hvg)
    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "additive": loo_baseline_source("additive", real_delta, real_key, base, k=k),
        "knn": loo_baseline_source("knn", real_delta, real_key, base, k=k),
        "pca": loo_baseline_source("pca", real_delta, real_key, base, k=k, genes=learned_genes),
        "nmf": loo_baseline_source("nmf", real_delta, real_key, base, k=k, genes=learned_genes),
        "stack": build_generated_deltas(generated_dir, query_baseline, pert_to_drug),
    }

    # measured-delta reference (not a positive control -- see fmharness.controls'
    # plant_interaction/"planted", the flowchart's real "planted interaction, recovered"):
    # the REAL measured delta as its own "prediction", the best-case ceiling for the
    # penalized grid (score_check2's part b), passed via
    # measured_delta= rather than folded into `sources` (part a) -- see score_check2's measured_delta=
    # docstring for why (this driver has no Gate print of its own, but score_check2's own
    # part (a)/part (b) split stays consistent regardless of caller).
    return score_check2(
        sources,
        real_key,
        base,
        hvg,
        filtered_design,
        hallmark=hallmark,
        fixed_methods=fixed_methods,
        penalties=penalties,
        folds=folds,
        stack_emb=stack_emb,
        measured_delta=(real_delta.copy(), real_key.copy()),
        n_permutations=n_permutations,
    )


def emit(table, name, out_dir, params) -> None:
    """Write a result table and the parameters that produced it. Never optional.

    See scripts/score_generation_eval.py's emit for why: printing is for watching, this is for
    keeping. A number that exists only in an uncommitted job log cannot be regenerated.
    """
    import json
    import subprocess

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{name}.csv"
    table.to_csv(dest, index=False)
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        sha = "unknown"
    (out_dir / f"{name}.params.json").write_text(
        json.dumps({"result": dest.name, "git_sha": sha, "rows": int(len(table)), **params},
                   indent=2, default=str) + "\n"
    )
    print(f"  wrote {dest} ({len(table)} rows) + {name}.params.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context", default=None, help="Tahoe context AnnData (build_tahoe_context)")
    ap.add_argument(
        "--deltas-bundle",
        default=None,
        help="dir of real_delta.parquet/real_key.parquet/base.parquet -- provide this OR "
        "--context, not both (a live --context rebuild is not guaranteed to match a bundle "
        "built earlier from a different Tahoe context snapshot -- see "
        "check1_registry_driver.py's own --deltas-bundle help text for the full caveat)",
    )
    ap.add_argument(
        "--query-baseline", required=True, help="AnnData fed to stack-generation as --test-adata"
    )
    ap.add_argument("--generated-dir", required=True, help="dir of Stack-generated <pert>.h5ad")
    ap.add_argument("--pert-map", required=True, help="TSV 'pert_id<TAB>cid' (context split)")
    ap.add_argument("--checkpoint-label", required=True, help="e.g. cytokine- or drug-aligned")
    ap.add_argument("--auc-tranche", default="gdscv2", help="measured-AUC cohort for check 2")
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument(
        "--k",
        type=int,
        default=None,
        help="neighbors for k-NN / n_components for PCA/NMF; omit to CV-select per fold "
        "(fmharness.deltas._K_GRID) instead of a fixed value",
    )
    ap.add_argument(
        "--hallmark-path", default="data/static/hallmark_signatures.gmt", help="Hallmark .gmt path"
    )
    ap.add_argument(
        "--methods",
        default=",".join(FIXED_READOUTS),
        help="fixed-signature readouts on the delta sources (subset of hallmark, proliferation)",
    )
    ap.add_argument(
        "--penalties",
        default=",".join(PENALTY_NAMES),
        help="penalized regressions for the representation grid (subset of l2, l1, en)",
    )
    ap.add_argument(
        "--folds",
        type=int,
        default=5,
        help="grouped-by-cell-line folds for the leakage-free penalized fit; set >= #lines "
        "(e.g. 999) for true leave-one-cell-line-out",
    )
    ap.add_argument(
        "--stack-emb",
        nargs="*",
        default=None,
        help="precomputed per-line FM embeddings to add as check-2 representations, each "
        "'label=path' (path .h5ad/.parquet/.csv, index/obs = cell line id). Repeatable, e.g. "
        "--stack-emb base=emb_base.h5ad aligned=emb_aligned.h5ad",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="where result tables and parameter sidecars go. Defaults to "
        "results/<job id or 'local'>. Output is always written.",
    )
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--corpus-lines", default=None, help="comma-separated declared pretrain lines")
    ap.add_argument("--corpus-drugs", default=None, help="comma-separated declared pretrain drugs")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if getattr(args, 'out_dir', None) else Path('results') / (
        os.environ.get('SLURM_JOB_ID') or 'local'
    )
    run_params = {k: v for k, v in vars(args).items()}
    print(f'writing results to {out_dir}')

    if corpus_declared_partially(args.corpus_lines, args.corpus_drugs):
        ap.error(
            "--corpus-lines and --corpus-drugs must both be given together (or neither, to "
            "run unfiltered) -- giving only one silently disables leakage filtering"
        )
    if ground_truth_source_declared_ambiguously(args.context, args.deltas_bundle):
        ap.error("provide exactly one of --context (single-cell) or --deltas-bundle (pseudobulk)")

    repo = Path(__file__).resolve().parent.parent
    if args.deltas_bundle:
        bdir = _rel(repo, args.deltas_bundle)
        real_delta = pd.read_parquet(bdir / "real_delta.parquet")
        real_key = pd.read_parquet(bdir / "real_key.parquet")
        base = pd.read_parquet(bdir / "base.parquet")
    else:
        real_delta, real_key, base = build_tahoe_deltas(ad.read_h5ad(args.context))
    pert_to_drug = load_pert_map(_rel(repo, args.pert_map))

    _, auc_design = build_sample_design(
        load_tranche(args.auc_tranche, repo), "all", "auc", drug_key="pubchem_cid"
    )

    stack_emb_map: dict[str, pd.DataFrame] = {}
    for spec in args.stack_emb or []:
        label, _, p = spec.partition("=")
        if not (label.strip() and p.strip()):
            ap.error(f"--stack-emb expects 'label=path', got {spec!r}")
        stack_emb_map[label.strip()] = load_line_matrix(_rel(repo, p.strip()))

    fixed_methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    penalties = tuple(p.strip() for p in args.penalties.split(",") if p.strip())
    table = run_check2(
        real_delta,
        real_key,
        base,
        query_baseline=_rel(repo, args.query_baseline),
        generated_dir=_rel(repo, args.generated_dir),
        pert_to_drug=pert_to_drug,
        checkpoint_label=args.checkpoint_label,
        hallmark_path=repo / args.hallmark_path,
        auc_design=auc_design,
        n_hvg=args.n_hvg,
        k=args.k,
        fixed_methods=fixed_methods,
        penalties=penalties,
        folds=args.folds,
        stack_emb=stack_emb_map,
        n_permutations=args.n_permutations,
        pretraining_lines=parse_corpus_set(args.corpus_lines),
        pretraining_drugs=parse_corpus_set(args.corpus_drugs),
        task_signal_in_pretrain="adjacent" if args.corpus_lines else "none",
    )
    print(table.to_string(index=False))
    emit(table, "check2_registry", out_dir, run_params)


if __name__ == "__main__":
    main()
