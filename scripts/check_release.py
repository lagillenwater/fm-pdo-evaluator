"""Refuse to commit data that must not be released yet.

This repository is public, and some of the data behind its results is not. Provenance and
embargo pull in opposite directions: one wants the inputs committed, the other forbids it. The
way through is that provenance needs a HASH, not the bytes -- see data/release_manifest.yaml.

This is the enforcement. It fails CLOSED: an artifact nobody has classified is treated as
embargoed, so forgetting to classify something is safe and over-classifying is merely annoying.

Three checks, run over what is STAGED, before it becomes a commit:

  1. PATH TIER      -- the staged path matches an entry marked embargoed or restricted.
  2. ROW-LEVEL DATA -- a staged table carries a sample identifier column. This is the one that
                       catches real mistakes: committing a prediction frame to make a results
                       table reproducible would publish per-(patient, drug) measured response,
                       which is exactly the well-intentioned move this check exists to stop.
  3. HASH MATCH     -- a staged file's sha256 equals that of a known embargoed artifact, which
                       catches the same bytes arriving under a new name.

Install as a pre-commit hook:

    ln -s ../../scripts/check_release.py .git/hooks/pre-commit

or run it directly: `python scripts/check_release.py`. Non-zero exit means do not commit.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import subprocess
import sys
from pathlib import Path

try:  # pyyaml is nicer, but a release gate must not fail open because a package is absent
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only on interpreters without it
    yaml = None

MANIFEST = Path("data/release_manifest.yaml")

# Column names that make a table row-level. A file carrying any of these identifies the sample
# a measurement came from, which is what embargo is about.
SAMPLE_COLUMNS = {
    "patient", "patient_id", "sample", "sample_id", "line", "cell_line",
    "cell_line_id", "organoid", "organoid_id", "donor", "donor_id", "subject",
}
TABLE_SUFFIXES = {".csv", ".tsv", ".parquet"}

# The gate governs DATA release, not code. Source, docs and config are publishable by nature,
# and applying a fail-closed default to them flagged every tracked file, which would have made
# the gate useless noise. In scope: recognised data extensions, plus anything under a directory
# that exists to hold data.
DATA_SUFFIXES = {
    ".csv", ".tsv", ".parquet", ".h5ad", ".gctx", ".pkl", ".ckpt",
    ".npz", ".npy", ".rds", ".h5", ".mtx", ".loom",
}
DATA_DIRS = ("data/", "results/", "docs/results/", "data/tranches/", "data/raw/")


def in_scope(rel: str) -> bool:
    """Is this a data artifact the release policy governs?"""
    return Path(rel).suffix.lower() in DATA_SUFFIXES or rel.startswith(DATA_DIRS)


def load_manifest(path: Path) -> dict:
    """Read the release manifest, with a fallback for interpreters lacking pyyaml.

    The gate runs as a git pre-commit hook under whatever `python3` is on PATH, which is not
    necessarily the project environment. Failing because a package is missing would either block
    every commit or tempt someone to delete the hook, so this parses the small subset of YAML
    the manifest actually uses: two scalars and a list of entries with `path` and `release`.
    Only those two keys drive enforcement; `why` is documentation.
    """
    text = path.read_text()
    if yaml is not None:
        return yaml.safe_load(text)

    out: dict = {"entries": []}
    entry: dict | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("- path:"):
            entry = {"path": stripped.split(":", 1)[1].strip().strip('"\'')}
            out["entries"].append(entry)
        elif entry is not None and stripped.startswith("release:"):
            entry["release"] = stripped.split(":", 1)[1].strip()
        elif not line.startswith((" ", "-")) and ":" in stripped:
            k, v = stripped.split(":", 1)
            if v.strip():
                out[k.strip()] = v.strip()
    return out


def staged_files(repo: Path) -> list[str]:
    """Paths staged for commit. Empty outside a commit, in which case we check the whole tree."""
    out = subprocess.run(
        ["git", "-C", str(repo), "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [ln for ln in out.stdout.split() if ln]


def tier_for(path: str, manifest: dict) -> tuple[str, str]:
    """The tier for a path, and the rule that decided it. Longest matching pattern wins."""
    best: tuple[int, str, str] = (-1, manifest.get("default", "embargoed"), "default (unclassified)")
    for e in manifest.get("entries", []):
        pat = e["path"]
        if fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(path, pat.rstrip("/") + "/**"):
            if len(pat) > best[0]:
                best = (len(pat), e["release"], pat)
    return best[1], best[2]


def sample_columns_in(path: Path) -> set[str]:
    """Sample identifier columns present in a table, read from the header only."""
    try:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            cols = {c.lower() for c in pq.ParquetFile(path).schema.names}
        else:
            sep = "\t" if path.suffix == ".tsv" else ","
            header = path.read_text(errors="ignore").split("\n", 1)[0]
            cols = {c.strip().strip('"').lower() for c in header.split(sep)}
    except Exception:
        return set()
    return cols & SAMPLE_COLUMNS


def sha256(path: Path) -> str:
    """Content hash, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while b := fh.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def known_embargoed_hashes(repo: Path, manifest: dict) -> dict[str, str]:
    """sha256 -> origin, for embargoed artifacts whose hashes have been recorded."""
    out: dict[str, str] = {}
    for e in manifest.get("entries", []):
        for h in e.get("sha256", []) or []:
            out[h] = e["path"]
    return out


def main() -> int:
    """Check staged files against the release manifest."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--all", action="store_true", help="check every tracked file, not just staged")
    args = ap.parse_args()
    repo = args.repo.resolve()

    mpath = repo / MANIFEST
    if not mpath.exists():
        print(f"check_release: {MANIFEST} is missing. Refusing to guess what may be published.")
        return 1
    manifest = load_manifest(mpath)
    hashes = known_embargoed_hashes(repo, manifest)

    if args.all:
        files = subprocess.run(
            ["git", "-C", str(repo), "ls-files"], capture_output=True, text=True, check=True
        ).stdout.split()
    else:
        files = staged_files(repo)
    if not files:
        print("check_release: nothing staged")
        return 0

    problems: list[str] = []
    for rel in files:
        path = repo / rel
        if not path.exists() or not in_scope(rel):
            continue
        tier, rule = tier_for(rel, manifest)

        if tier in ("embargoed", "restricted"):
            problems.append(f"{rel}\n    tier '{tier}' by rule '{rule}' -- commit its hash, not the file")
            continue

        # A table cleared as public still cannot carry sample identifiers.
        if path.suffix in TABLE_SUFFIXES:
            cols = sample_columns_in(path)
            if cols:
                problems.append(
                    f"{rel}\n    marked '{tier}' by rule '{rule}', but carries sample column(s) "
                    f"{sorted(cols)} -- that is row-level data. Aggregate it, or reclassify."
                )
                continue

        if hashes:
            h = sha256(path)
            if h in hashes:
                problems.append(f"{rel}\n    sha256 matches embargoed artifact {hashes[h]}")

    checked = [f for f in files if in_scope(f)]
    print(f"check_release: examined {len(checked)} data artifact(s) of {len(files)} file(s)")
    if problems:
        print(f"\nREFUSING {len(problems)} file(s):\n")
        for p in problems:
            print(f"  {p}\n")
        print("Fix by aggregating away the sample dimension, or by recording the hash with")
        print("scripts/promote_result.py instead of committing the contents.")
        return 1
    print("check_release: nothing here is embargoed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
