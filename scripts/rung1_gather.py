"""Rung 1, stage 3: score every source together, because the metric is cross-source.

score_delta_sources restricts all sources to the (patient, drug) support they SHARE, so it
cannot be split across tasks: a task scoring one source alone would compute a different support
and every row would be scored on a different set, silently, with the table still looking
well-formed. That is the defect restrict_common_support exists to prevent, and it is why the
fan-out stops at construction.

Refuses an incomplete set rather than scoring what happens to be present -- a missing source
changes the shared support and therefore every other source's number, so a short table here is
not merely incomplete, it is wrong.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from fmharness.evaluation import score_delta_sources


def main() -> None:
    """Load every source, score Check 1, and write the table with its provenance."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()

    plan = json.loads((args.plan_dir / "plan.json").read_text())
    expected = sorted(set(plan["grid"]) | set(plan["prebuilt"]))
    sdir = args.plan_dir / "sources"
    got = sorted({p.name[: -len("_delta.parquet")] for p in sdir.glob("*_delta.parquet")})
    missing = [s for s in expected if s not in got]
    print(f"expected {len(expected)} sources, found {len(got)}")
    if missing:
        print(f"  MISSING: {missing}")
        if not args.allow_incomplete:
            raise SystemExit(
                f"refusing to score with {len(missing)} source(s) missing. The shared "
                "(patient, drug) support is computed ACROSS sources, so a missing source "
                "changes every other source's number -- this table would be wrong, not short."
            )

    sources = {
        s: (pd.read_parquet(sdir / f"{s}_delta.parquet"), pd.read_parquet(sdir / f"{s}_key.parquet"))
        for s in expected
        if s in got
    }
    sizes = {s: d.shape[1] for s, (d, _) in sources.items()}
    if len(set(sizes.values())) > 1:
        raise SystemExit(f"sources are on different gene panels, so they are not comparable: {sizes}")
    print(f"  all {len(sources)} sources on {next(iter(sizes.values()))} genes -- verified")

    real_delta = pd.read_parquet(args.plan_dir / "real_delta.parquet")
    real_key = pd.read_parquet(args.plan_dir / "real_key.parquet")
    table = score_delta_sources(sources, real_delta, real_key, n_hvg=None)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dest = args.out_dir / "rung1_check1_fidelity.csv"
    table.to_csv(dest, index=False)
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
    except Exception:
        sha = "unknown"
    (args.out_dir / "rung1_check1_fidelity.params.json").write_text(
        json.dumps({"result": dest.name, "git_sha": sha, "plan_git_sha": plan["git_sha"],
                    "plan_slurm_job_id": plan["slurm_job_id"], "panel_size": plan["panel_size"],
                    "folds": plan["folds"], "sources": sorted(sources),
                    "missing_sources": missing, "plan_args": plan["args"]}, indent=2) + "\n"
    )
    print(f"\nwrote {dest}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
