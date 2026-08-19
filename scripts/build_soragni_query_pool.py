"""Build the synthetic query pool Path B's faithful (--mode mdm) generation needs.

stack_input_sarcoma.h5ad (the Soragni tumor-RNA query) is bulk RNA-seq: one CPM row per
patient, 17 total. --mode mdm's scheduled context draw needs up to ~281 cells from the
whole query pool regardless of true sample count (see scripts/alpine/04_stack_generate.sbatch),
and unlike Tahoe there are no real additional single cells to draw from. This expands the
bulk baseline into a synthetic per-patient replicate pool via
fmharness.stack_aggregate.build_synthetic_replicate_pool (Poisson-resampled at a nominal
single-cell library size, renormalized back to CPM) -- NOT real cells, a documented,
seeded approximation.

  PYTHONPATH=src python scripts/build_soragni_query_pool.py \\
      --baseline data/reference/stack_input_sarcoma.h5ad \\
      --out soragni_query_pool.h5ad
"""

from __future__ import annotations

import argparse

import anndata as ad

from fmharness.stack_aggregate import build_synthetic_replicate_pool


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default="data/reference/stack_input_sarcoma.h5ad")
    ap.add_argument("--out", default="soragni_query_pool.h5ad")
    ap.add_argument(
        "--n-replicates",
        type=int,
        default=24,
        help="synthetic replicates per patient (17 patients x 24 = 408, ~1.4x the mdm "
        "schedule's ~281-cell floor -- matches Tahoe's own headroom ratio)",
    )
    ap.add_argument(
        "--library-size",
        type=float,
        default=5000.0,
        help="nominal single-cell depth for the Poisson resample; smaller is noisier",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    baseline = ad.read_h5ad(args.baseline)
    pool = build_synthetic_replicate_pool(
        baseline,
        n_replicates=args.n_replicates,
        library_size=args.library_size,
        seed=args.seed,
    )
    pool.write_h5ad(args.out)
    print(
        f"{baseline.n_obs} patients x {args.n_replicates} replicates = {pool.n_obs} synthetic "
        f"cells x {pool.n_vars} genes (library_size={args.library_size}, seed={args.seed}) "
        f"-> {args.out}"
    )


if __name__ == "__main__":
    main()
