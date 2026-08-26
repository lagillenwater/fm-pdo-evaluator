"""Rung 1, stage 2: build ONE baseline source from the pinned inputs.

pca and nmf each fit a ridge per fold over every panel gene, so building four baselines in
sequence is what pushed rung 1 past its walltime. They share nothing but the pinned inputs, so
one per task turns a serial chain into the slowest single source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fmharness.deltas import loo_baseline_source


def main() -> None:
    """Build the task's source and write it beside the pinned ones."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan-dir", required=True, type=Path)
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    plan = json.loads((args.plan_dir / "plan.json").read_text())
    grid = plan["grid"]
    name = args.name
    if name is None:
        if args.task_id is None or args.task_id >= len(grid):
            print(f"task {args.task_id} is past the grid ({len(grid)} sources); nothing to do")
            return
        name = grid[args.task_id]

    real_delta = pd.read_parquet(args.plan_dir / "real_delta.parquet")
    real_key = pd.read_parquet(args.plan_dir / "real_key.parquet")
    base = pd.read_parquet(args.plan_dir / "base.parquet")
    panel = pd.Index(real_delta.columns)
    print(f"building {name} on {len(panel)} genes, {plan['folds']}-fold, k={plan['k']}")

    d, kk = loo_baseline_source(
        name, real_delta, real_key, base, k=plan["k"], genes=panel, n_folds=plan["folds"]
    )
    d = d.reindex(columns=panel)
    sdir = args.plan_dir / "sources"
    sdir.mkdir(parents=True, exist_ok=True)
    d.to_parquet(sdir / f"{name}_delta.parquet")
    kk.to_parquet(sdir / f"{name}_key.parquet")
    print(f"wrote {name}: {d.shape[0]} rows x {d.shape[1]} genes")


if __name__ == "__main__":
    main()
