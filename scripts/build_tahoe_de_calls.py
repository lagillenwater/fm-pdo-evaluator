"""Build the ground-truth DE-calls bundle Check 1's DE-based metrics score against.

Wilcoxon rank-sum + Benjamini-Hochberg FDR per (line, drug, gene), from the raw per-cell
tahoe_context.h5ad -- see fmharness.deltas.build_tahoe_de_calls for the method and
docs/superpowers/specs/2026-08-18-stack-faithful-generation-and-de-metrics-design.md's Change 2
section for why this needs the raw per-cell context (not the tahoe_deltas/ pseudobulk bundle,
which retains no per-cell/significance information). Real, one-time compute (~1,650 (line, drug)
pairs); cache the output, matching the existing tahoe_deltas/ bundle pattern, rather than
repeating it on every Check-1 run.

Run (CPU-only, no GPU needed -- Wilcoxon rank-sum + BH correction is not a model call; runs
directly in this worktree since tahoe_context.h5ad is already local, no Alpine submission
required):
    uv run python scripts/build_tahoe_de_calls.py --context tahoe_context.h5ad \\
        --out tahoe_de_calls/de_calls.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad

from fmharness.deltas import build_tahoe_de_calls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context", default="tahoe_context.h5ad", help="Tahoe context AnnData")
    ap.add_argument("--out", default="tahoe_de_calls/de_calls.parquet")
    ap.add_argument("--lfc-threshold", type=float, default=0.25)
    ap.add_argument("--fdr-threshold", type=float, default=0.05)
    args = ap.parse_args()

    calls = build_tahoe_de_calls(
        ad.read_h5ad(args.context),
        lfc_threshold=args.lfc_threshold,
        fdr_threshold=args.fdr_threshold,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    calls.to_parquet(out)
    n_pairs = len(calls[["patient", "drug"]].drop_duplicates())
    n_sig = int(calls["significant"].sum())
    print(
        f"{len(calls)} (line, drug, gene) rows, {n_pairs} pairs, "
        f"{n_sig} significant calls -> {out}"
    )


if __name__ == "__main__":
    main()
