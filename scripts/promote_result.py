"""Promote a run's output into committed evidence.

The problem this solves: this repo generates far more logs than it has conclusions. Committing
all of them buries the few that matter; committing none of them -- the status quo -- leaves 78
of 106 published claims with nothing a reader can open. Neither "keep everything" nor "keep
nothing" is right, because the useful distinction is not about logs at all.

**A log is not evidence. A log becomes evidence the moment a claim cites it.**

So promotion is an explicit act, tied to a claim, and it is the ONLY way a number enters the
committed record:

1. Run whatever you like. Logs stay ephemeral and nobody tracks them.
2. When a run produces a number you intend to publish, promote it. That writes a small result
   table plus a provenance record into ``docs/results/``.
3. Everything not promoted stays ephemeral. Test logs are never promoted, so they never clutter.

The repo therefore contains exactly the evidence behind its conclusions, and nothing else.

What gets committed is deliberately small: the derived table (kilobytes), not the log. The
provenance record carries the log's sha256 and where it lives, so a reader can confirm they are
holding the same file even when it is a 40MB job log parked on the cluster. For anything cheap
to regenerate, the script plus the pinned inputs ARE the evidence and the log is disposable --
which is most cases. The log's hash matters only for runs too expensive to repeat on demand.

Usage:

    python scripts/promote_result.py \\
        --name check2_grid_5fold \\
        --result /tmp/out.csv \\
        --script scripts/score_generation_eval.py \\
        --claim "Check 2 penalized grid, 5-fold, after the 2026-08-24 metric fixes" \\
        --input tahoe_deltas/real_delta.parquet \\
        --log logs/stack-emb-score-31634484.out --job-id 31634484
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

RESULTS_DIR = Path("docs/results")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Content hash of a file, streamed so a multi-GB input does not need to fit in memory."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_sha(repo: Path) -> str:
    """The commit the result was produced at, or 'unknown' outside a git tree.

    Recorded because a result is only interpretable against the code that made it -- the same
    script at a different commit is a different measurement.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def describe(path: Path) -> dict[str, object]:
    """Record a file by path, size and content hash, without copying it."""
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def main() -> None:
    """Write ``docs/results/<name>.csv`` and ``<name>.provenance.json``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="short slug; becomes the filename")
    ap.add_argument("--result", required=True, type=Path, help="the table to commit (csv/tsv)")
    ap.add_argument("--script", required=True, help="repo-relative script that produced it")
    ap.add_argument("--claim", required=True, help="one line: what this result is evidence FOR")
    ap.add_argument("--input", action="append", default=[], type=Path, help="repeatable")
    ap.add_argument("--log", type=Path, default=None, help="run log; hashed, not copied")
    ap.add_argument("--job-id", default=None, help="scheduler job id, if any")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()

    repo = args.repo.resolve()
    script_path = repo / args.script
    if not script_path.exists():
        raise SystemExit(
            f"--script {args.script} does not exist. A result whose producing script is not in "
            "the repo cannot be regenerated, which is the whole point of promoting it."
        )
    if not args.input:
        raise SystemExit(
            "--input is required (at least one). A result with no recorded inputs cannot be "
            "checked against a rerun, so it is an assertion, not evidence."
        )

    out_dir = repo / RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{args.name}{args.result.suffix or '.csv'}"
    dest.write_bytes(args.result.read_bytes())

    prov: dict[str, object] = {
        "name": args.name,
        "claim": args.claim,
        "script": args.script,
        "git_sha": git_sha(repo),
        "result": describe(dest),
        "inputs": [describe(p) for p in args.input if p.exists()],
        "missing_inputs": [str(p) for p in args.input if not p.exists()],
    }
    if args.job_id:
        prov["job_id"] = args.job_id
    if args.log and args.log.exists():
        # The log is hashed and located, NOT copied. Committing job logs is what turns a repo
        # into a landfill; committing their hashes keeps them checkable without keeping them.
        prov["log"] = describe(args.log)

    (out_dir / f"{args.name}.provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"promoted -> {dest.relative_to(repo)}")
    print(f"           {(out_dir / (args.name + '.provenance.json')).relative_to(repo)}")
    if prov["missing_inputs"]:
        print(f"  WARNING: {len(prov['missing_inputs'])} declared inputs were not found")


if __name__ == "__main__":
    main()
