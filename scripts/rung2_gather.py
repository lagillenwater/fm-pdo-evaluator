"""Rung 2, stage 3: assemble the grid and compute each source's transfer penalty.

The penalty is cross_platform minus in_platform for a source, and it can only be computed once
both arms are present -- so this refuses an incomplete grid rather than reporting a penalty
against a missing arm, which would silently read as a large negative transfer effect.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


def main() -> None:
    """Concatenate the cells, pair the arms, and emit the penalty table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan-dir", required=True, type=Path)
    ap.add_argument("--parts-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()

    plan = json.loads((args.plan_dir / "plan.json").read_text())
    expected = set(plan["grid"])
    parts = sorted(args.parts_dir.glob("part_*.csv"))
    table = pd.concat([pd.read_csv(p) for p in parts]) if parts else pd.DataFrame()
    got = {f"{r.source}|{r.arm}" for r in table.itertuples()} if len(table) else set()
    missing = sorted(expected - got)
    print(f"expected {len(expected)} cells, found {len(got)}")
    if missing:
        print(f"  MISSING: {missing}")
        if not args.allow_incomplete:
            raise SystemExit(
                f"refusing to report transfer penalties with {len(missing)} cell(s) missing -- a "
                "penalty computed against an absent arm reads as a large negative effect."
            )

    piv = table.pivot_table(index="source", columns="arm", values="mean_rho")
    if {"in_platform", "cross_platform"} <= set(piv.columns):
        piv["transfer_penalty"] = piv["cross_platform"] - piv["in_platform"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.sort_values(["source", "arm"]).to_csv(args.out_dir / "rung2_grid.csv", index=False)
    piv.reset_index().to_csv(args.out_dir / "rung2_transfer_penalty.csv", index=False)
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
    except Exception:
        sha = "unknown"
    (args.out_dir / "rung2_grid.params.json").write_text(
        json.dumps({"git_sha": sha, "plan_git_sha": plan["git_sha"],
                    "plan_slurm_job_id": plan["slurm_job_id"], "panel_size": plan["panel_size"],
                    "missing_cells": missing, "plan_args": plan["args"]}, indent=2) + "\n"
    )
    print("\n=== rung 2: transfer penalty (cross_platform - in_platform) ===")
    print(piv.to_string())
    print("\nStack is absent by construction: it is not fitted here, so its penalty is exactly")
    print("zero and it would be a constant sitting beside numbers that measure something.")


if __name__ == "__main__":
    main()
