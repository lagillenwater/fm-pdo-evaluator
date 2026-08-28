"""Promote a run's output into committed evidence (SPEC rule 1; PROCESS §1 'Promote').

A log is not evidence; a result becomes evidence when a claim cites it, and citing means
a ``results/<task-slug>/`` copy with a schema-validated ``PromotedResult`` beside it.
Three fields cannot be reconstructed later and are written here: whether the tree was
clean, the producing commit, and the artifact's checksum. Promotion REFUSES when the
task-side copy and an existing promoted copy differ -- an artifact must not change under
its claim.

Usage (rung 0):
    uv run python scripts/promote_result.py \
        --task rung0-replicate-ceiling \
        --result docs/tasks/rung0-replicate-ceiling/rung0_delta_reproducibility.csv \
        --script scripts/delta_reproducibility.py \
        --input <panel file copy> --input <cid file> \
        --seed 0 --data-commit <tranche content_hash> \
        --arg tranche_id=tahoe100m-pseudobulk-de.v1 --job-id <slurm id> \
        --log results/rung0-replicate-ceiling/<job log>
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fmharness.schema import EnvironmentSnapshot, PromotedResult


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def promote(
    *,
    task: str,
    result: Path,
    script: str,
    inputs: list[Path],
    seed: int,
    data_commit: str,
    args: dict[str, str],
    job_id: str | None,
    log: Path | None,
    repo: Path,
) -> Path:
    repo = repo.resolve()
    if not (repo / script).exists():
        raise SystemExit(
            f"--script {script} is not in the repo; a result whose producing script "
            "cannot be found cannot be regenerated"
        )
    if not inputs:
        raise SystemExit(
            "at least one --input is required; a result with no recorded inputs cannot "
            "be checked against a rerun"
        )
    missing = [p for p in inputs if not p.exists()]
    if missing:
        raise SystemExit(f"declared inputs not found: {[str(p) for p in missing]}")

    # Check clean_tree status before writing any files
    clean_tree_status = _git(repo, "status", "--porcelain") == ""
    code_commit = _git(repo, "rev-parse", "HEAD")

    out_dir = repo / "results" / task
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / result.name
    src_hash = sha256_of(result)
    if dest.exists() and sha256_of(dest) != src_hash:
        raise SystemExit(
            f"refusing: {dest.relative_to(repo)} exists and its checksum differs from "
            f"{result} -- the promoted copy and the task-side copy differ"
        )
    dest.write_bytes(result.read_bytes())

    record = PromotedResult(
        result=str(dest.relative_to(repo)),
        result_sha256=sha256_of(dest),
        task=task,
        script=script,
        args={k: str(v) for k, v in args.items()},
        inputs={str(p): sha256_of(p) for p in inputs},
        log=str(log) if log else None,
        log_sha256=sha256_of(log) if log else None,
        job_id=job_id,
        clean_tree=clean_tree_status,
        environment=EnvironmentSnapshot(
            code_commit=code_commit,
            python_version=platform.python_version(),
            seed=seed,
            cuda_deterministic=False,
            data_commit=data_commit,
        ),
        promoted_at=datetime.now(UTC),
    )
    record_path = dest.with_suffix(".provenance.json")
    record_path.write_text(record.model_dump_json(indent=2) + "\n")
    print(f"promoted -> {dest.relative_to(repo)}")
    print(f"           {record_path.relative_to(repo)}")
    return record_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--result", required=True, type=Path)
    ap.add_argument("--script", required=True)
    ap.add_argument("--input", action="append", default=[], type=Path, dest="inputs")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-commit", required=True)
    ap.add_argument("--arg", action="append", default=[], help="key=value, repeatable")
    ap.add_argument("--job-id", default=None)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ns = ap.parse_args()
    promote(
        task=ns.task,
        result=ns.result,
        script=ns.script,
        inputs=ns.inputs,
        seed=ns.seed,
        data_commit=ns.data_commit,
        args=dict(kv.split("=", 1) for kv in ns.arg),
        job_id=ns.job_id,
        log=ns.log,
        repo=ns.repo,
    )


if __name__ == "__main__":
    main()
