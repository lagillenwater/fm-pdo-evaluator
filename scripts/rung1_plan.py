"""Rung 1, stage 1: build the generated deltas once, fix the panel, pin the shared inputs.

Rung 1 was a single serial process that built every source in turn and then scored them. The
expensive part is source CONSTRUCTION -- pca and nmf fit a ridge per fold across every panel
gene -- and the sources are independent once the panel is fixed, so they belong in an array.
Scoring is not independent: score_delta_sources restricts all sources to the (patient, drug)
support they SHARE, so it has to see every source at once and belongs in the gather.

Two things must happen here rather than per task. The generated deltas set the panel ceiling, so
computing them once fixes the gene set every task builds on; and pinning real_delta, real_key and
base to that panel means a task cannot silently construct on a different one.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from fmharness.deltas import (
    build_generated_deltas,
    common_gene_panel,
    load_panel_constraint,
    load_pert_map,
)

BASELINES = ("observed_delta", "knn", "pca", "nmf")


def git_sha() -> str:
    """The commit these inputs were built at."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    """Fix the panel, pin the inputs, and write the per-source task grid."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deltas-bundle", default="tahoe_deltas")
    ap.add_argument("--generated-dir", action="append", default=None, help="label=dir, repeatable")
    ap.add_argument("--query-baseline", default="tahoe_query_baseline.h5ad")
    ap.add_argument("--pert-map", default="context_by_drug/pert_to_cid.tsv")
    ap.add_argument("--panel-source", action="append", default=None, help="label=path, repeatable")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bdir = Path(args.deltas_bundle)
    real_delta = pd.read_parquet(bdir / "real_delta.parquet")
    real_key = pd.read_parquet(bdir / "real_key.parquet")
    base = pd.read_parquet(bdir / "base.parquet")

    gen_specs = args.generated_dir or [
        "stack_cytokine=generated_agg",
        "stack_drug_aligned=generated_drug_aligned_agg",
    ]
    pert = load_pert_map(Path(args.pert_map))
    generated: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for spec in gen_specs:
        label, _, gdir = spec.partition("=")
        if not gdir:
            label, gdir = "stack", label
        if not Path(gdir).exists():
            print(f"  SKIP {label}: {gdir} not present")
            continue
        generated[label] = build_generated_deltas(Path(gdir), Path(args.query_baseline), pert)
        print(f"  built {label} from {gdir}: {generated[label][0].shape[1]} genes")

    panel_inputs: dict[str, pd.DataFrame] = {k: v[0] for k, v in generated.items()}
    for spec in args.panel_source or []:
        label, _, path = spec.partition("=")
        pp = Path(path)
        if not pp.exists():
            raise SystemExit(f"--panel-source {label}={pp} not present; refusing to widen the panel")
        panel_inputs[label] = pd.DataFrame(columns=load_panel_constraint(pp))
        print(f"  panel constraint {label}: {panel_inputs[label].shape[1]} genes")

    panel = common_gene_panel(real_delta, panel_inputs)
    print(f"common gene panel: {len(panel)} genes")
    if len(panel) < 1000:
        raise SystemExit(
            f"panel collapsed to {len(panel)} genes -- a gene-identifier mismatch, not a real "
            "intersection. Refusing to build a plan on it."
        )

    real_delta[panel].to_parquet(args.out_dir / "real_delta.parquet")
    real_key.to_parquet(args.out_dir / "real_key.parquet")
    base_cols = [g for g in panel if g in base.columns]
    base[base_cols].to_parquet(args.out_dir / "base.parquet")
    sdir = args.out_dir / "sources"
    sdir.mkdir(parents=True, exist_ok=True)
    for label, (gd, gk) in generated.items():
        gd[[g for g in panel if g in gd.columns]].to_parquet(sdir / f"{label}_delta.parquet")
        gk.to_parquet(sdir / f"{label}_key.parquet")
        print(f"  pinned {label} on the panel")
    # measured_delta is the panel-restricted real delta; pinned here so the gather needs no
    # special case and the array does not spend a task copying a frame.
    real_delta[panel].to_parquet(sdir / "measured_delta_delta.parquet")
    real_key.to_parquet(sdir / "measured_delta_key.parquet")

    grid = list(BASELINES)
    (args.out_dir / "plan.json").write_text(
        json.dumps(
            {
                "git_sha": git_sha(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
                "panel_size": int(len(panel)),
                "grid": grid,
                "prebuilt": sorted(list(generated) + ["measured_delta"]),
                "folds": int(args.folds),
                "k": args.k,
                "n_pairs": int(len(real_key)),
                "args": {k: str(v) for k, v in vars(args).items()},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"plan: {len(grid)} sources to build -> submit --array=0-{len(grid) - 1}")
    print(f"      {len(generated) + 1} sources already pinned (generated + measured_delta)")


if __name__ == "__main__":
    main()
