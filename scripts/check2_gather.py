"""Stage 3 of the Check-2 array: assemble the parts, and do the cross-representation checks.

Concatenating is the easy half. The half that matters is that two things can only be decided
once every part is present:

  - DEGENERACY. Whether two representations are linear images of each other is a property of
    the pair, so no single task can see it. `additive` is `measured_delta` sign-flipped
    (correlation -1.000000 after standardisation), and the assembled table must say so per row
    rather than leave a reader to notice that two rows carry identical numbers.
  - COMPLETENESS. A missing part is the failure mode that matters here: an array task that
    died leaves a table that looks whole and is quietly short a representation. This refuses
    to write a final table when parts are missing, rather than emitting a plausible one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from fmharness.check2 import detect_degenerate_representations


def main() -> None:
    """Concatenate the parts, annotate twins, and refuse an incomplete table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan-dir", required=True, type=Path)
    ap.add_argument("--parts-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write the table even when array tasks are missing. Off by default: a short table "
        "that looks whole is worse than no table.",
    )
    ap.add_argument(
        "--allow-missing-variants",
        default="",
        help="comma-separated declared variants that may be absent from this grid. Naming one "
        "makes its absence a decision on the record rather than an oversight.",
    )
    args = ap.parse_args()

    plan = json.loads((args.plan_dir / "plan.json").read_text())
    expected = set(plan["representations"])
    parts = sorted(args.parts_dir.glob("part_*.csv"))
    got = {p.stem[len("part_") :] for p in parts}

    missing = sorted(expected - got)
    extra = sorted(got - expected)
    print(f"expected {len(expected)} parts, found {len(got)}")
    if extra:
        print(f"  unexpected parts (ignored): {extra}")
    if missing:
        print(f"  MISSING: {missing}")
        if not args.allow_incomplete:
            raise SystemExit(
                f"refusing to write a table missing {len(missing)} representation(s). "
                "Rerun those array tasks, or pass --allow-incomplete deliberately."
            )

    table = pd.concat([pd.read_csv(p) for p in parts if p.stem[len("part_") :] in expected])
    table = table.sort_values(["source", "method"]).reset_index(drop=True)

    # Cross-representation, so it can only happen here.
    support = {
        name: {
            f.stem: pd.read_parquet(f)
            for f in sorted((args.plan_dir / "support" / name).glob("*.parquet"))
        }
        for name in sorted(expected)
    }
    twins: dict[str, str] = {}
    for a, b, corr in detect_degenerate_representations(support):
        print(f"  DEGENERATE: {a!r} and {b!r} are the same feature space (corr {corr:+.6f})")
        twins.setdefault(a, b)
        twins.setdefault(b, a)
    if twins:
        table["same_as"] = table["source"].map(twins)

    # A second completeness question, distinct from missing parts: were all the declared model
    # variants in the run to begin with? Job 31655278 had every array part it expected and was
    # still missing stack_drug_aligned, because the flags never included it.
    matrix_path = Path("data/model_matrix.yaml")
    if matrix_path.exists():
        import subprocess as _sp

        chk = _sp.run(
            [
                "python",
                "scripts/check_matrix.py",
                "--check",
                "check2",
                "--result",
                "/dev/stdin",
                "--allow-missing",
                args.allow_missing_variants,
            ],
            input=table.to_csv(index=False),
            capture_output=True,
            text=True,
        )
        print(chk.stdout.strip())
        if chk.returncode != 0 and not args.allow_incomplete:
            raise SystemExit(
                "refusing to write a grid that does not cover data/model_matrix.yaml. "
                "Score the missing variants, or waive them by name with --allow-missing-variants."
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dest = args.out_dir / "check2_grid.csv"
    table.to_csv(dest, index=False)
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        sha = "unknown"
    (args.out_dir / "check2_grid.params.json").write_text(
        json.dumps(
            {
                "result": dest.name,
                "git_sha": sha,
                "plan_git_sha": plan["git_sha"],
                "plan_slurm_job_id": plan["slurm_job_id"],
                "representations": sorted(expected),
                "missing_representations": missing,
                "degenerate_pairs": [[a, b] for a, b in twins.items()],
                "rows": int(len(table)),
                "plan_args": plan["args"],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {dest} ({len(table)} rows)")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
