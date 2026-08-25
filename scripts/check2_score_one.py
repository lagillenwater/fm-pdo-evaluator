"""Stage 2 of the Check-2 array: score ONE representation against the pinned support.

Reads what check2_plan.py decided and never recomputes it. The support restriction is
cross-representation -- it depends on every representation in the grid -- so a task that
rebuilt its own features would silently score a different (patient, drug) set from its
neighbours, and the assembled table would look fine while comparing different things.

Each task writes its own result and sidecar, so a row can be traced to the exact task, commit
and parameters that produced it without anyone having kept the job log.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from fmharness.check2 import penalized_preds, random_control_representation, seed_for_name
from fmharness.evaluation import interaction_rho, score_predictions


def load_support(plan_dir: Path, name: str) -> dict[str, pd.DataFrame]:
    """The exact frames the plan stage pinned for this representation."""
    rdir = plan_dir / "support" / name
    if not rdir.exists():
        raise SystemExit(f"no pinned support for {name!r} under {rdir}")
    out: dict[str, pd.DataFrame] = {}
    for f in sorted(rdir.glob("*.parquet")):
        out[f.stem] = pd.read_parquet(f)
    return out


def main() -> None:
    """Score one representation's rows and its noise draws."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan-dir", required=True, type=Path)
    ap.add_argument("--task-id", type=int, default=None, help="index into plan['representations']")
    ap.add_argument("--name", default=None, help="representation name, instead of --task-id")
    ap.add_argument("--penalties", default="l2,l1,en")
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    plan = json.loads((args.plan_dir / "plan.json").read_text())
    names = plan["representations"]
    name = args.name if args.name else names[args.task_id]
    if name not in names:
        raise SystemExit(f"{name!r} is not in the plan: {names}")

    design_target = pd.read_parquet(args.plan_dir / "design_target.parquet")
    fold_of = plan["fold_of"]
    n_folds = int(plan["n_folds"])
    uniq_lines = list(plan["uniq_lines"])
    penalties = tuple(p.strip() for p in args.penalties.split(",") if p.strip())
    feat = load_support(args.plan_dir, name)
    print(f"task: {name}  ({len(feat)} drugs, penalties {penalties})")

    # The noise null for this representation. Drawn from the same seeds the serial version uses,
    # so an array run and a serial run are comparable rather than merely similar.
    draws: dict[str, list[float]] = {}
    if not name.endswith("_random") and name != "prior":
        for d in range(int(plan["random_draws"])):
            drawn = random_control_representation(
                feat, sorted(feat), seed=(seed_for_name(name) + 7919 * d) & 0xFFFFFFFF
            )
            for pen in penalties:
                pr = penalized_preds(drawn, design_target, fold_of, n_folds, uniq_lines, pen)
                if pr.empty:
                    continue
                resid = pr.assign(y_pred=pr["y_pred"] - pr["y_prior"])
                draws.setdefault(pen, []).append(interaction_rho(resid, "y_pred"))
            print(f"  draw {d + 1}/{plan['random_draws']} done")

    rows: list[dict[str, object]] = []
    for pen in penalties:
        preds = penalized_preds(feat, design_target, fold_of, n_folds, uniq_lines, pen)
        if preds.empty:
            continue
        s = score_predictions(preds, n_perm=args.n_permutations)
        row: dict[str, object] = {"source": name, "method": pen, **{k: v for k, v in s.items()}}
        a = np.asarray([x for x in draws.get(pen, []) if np.isfinite(x)], dtype=float)
        if a.size >= 2 and float(a.std(ddof=1)) > 0:
            row["z_random"] = round(float((s["interaction"] - a.mean()) / a.std(ddof=1)), 2)
            row["p_random"] = round(float((1 + np.sum(a >= s["interaction"])) / (1 + a.size)), 4)
            row["random_mean"] = round(float(a.mean()), 3)
            row["n_random_draws"] = int(a.size)
        rows.append(row)
        print(f"  {name}/{pen}: interaction {s['interaction']:+.3f}")

    table = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dest = args.out_dir / f"part_{name}.csv"
    table.to_csv(dest, index=False)
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        sha = "unknown"
    (args.out_dir / f"part_{name}.params.json").write_text(
        json.dumps(
            {
                "representation": name,
                "result": dest.name,
                "git_sha": sha,
                "plan_git_sha": plan["git_sha"],
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "rows": int(len(table)),
                "args": {k: str(v) for k, v in vars(args).items()},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {dest} ({len(table)} rows)")


if __name__ == "__main__":
    main()
