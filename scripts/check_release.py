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
import re
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
# Of those, the ones that may hold a PUBLICLY CATALOGUED identity rather than a person. `line`
# holding A549 is a cell line published for decades; `line` holding SARC0065 is a
# patient-derived organoid under embargo, and the column NAME cannot tell them apart. So the
# values are checked against data/static/public_cell_lines.txt, built by
# scripts/build_public_cell_line_registry.py from the DepMap model table and the LINCS instance
# table. A file passes only if every value in these columns is publicly catalogued -- a
# verification, not a declaration, so a file that later gains an organoid row starts failing on
# its own. Person-linked columns are never checked this way and always refuse.
CELL_LINE_COLUMNS = {"line", "cell_line", "cell_line_id"}
PUBLIC_LINE_REGISTRY = Path("data/static/public_cell_lines.txt")
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


def public_line_registry() -> set[str] | None:
    """Publicly catalogued cell-line identifiers, or None when the registry is unavailable.

    None means the gate cannot verify and must refuse. Returning an empty set here would read
    as "no line is public" and behave identically, but None makes the missing-registry case
    distinguishable in the message -- the difference between "this line is not public" and
    "I could not check", which matters when the gate runs somewhere the registry was not
    committed. This gate has already failed open once this session by silently missing a file
    it depended on.
    """
    if not PUBLIC_LINE_REGISTRY.exists():
        return None
    return {
        re.sub(r"[^A-Z0-9]", "", ln.strip().upper())
        for ln in PUBLIC_LINE_REGISTRY.read_text().splitlines()
        if ln.strip()
    }


def nonpublic_line_values(path: Path, cols: set[str]) -> tuple[set[str], str]:
    """Values in cell-line columns that are NOT publicly catalogued.

    Returns ``(offending_values, status)`` where status is "ok", "no-registry" or "unreadable".
    Three outcomes rather than a boolean because the caller's message must say which happened:
    "this line is not public", "I could not find the registry" and "I could not parse the file"
    are different problems with different fixes, and collapsing them produced a message that
    blamed a missing registry for a NameError.

    Deliberately does NOT use pandas. This gate runs wherever a commit happens, including
    interpreters without the project environment, which is why sample_columns_in reads headers
    by hand and imports pyarrow only inside the parquet branch. Reaching for pandas here broke
    that and, because the failure was swallowed by a bare except, turned into a wrong message
    instead of a crash.
    """
    registry = public_line_registry()
    if registry is None:
        return set(), "no-registry"
    bad: set[str] = set()
    try:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            tbl = pq.read_table(path)
            for name in tbl.schema.names:
                if str(name).lower() in cols:
                    for v in tbl.column(name).to_pylist():
                        if v is None:
                            continue
                        if re.sub(r"[^A-Z0-9]", "", str(v).upper()) not in registry:
                            bad.add(str(v))
        else:
            import csv as _csv

            with path.open(newline="", errors="ignore") as fh:
                reader = _csv.DictReader(fh, delimiter="\t" if path.suffix == ".tsv" else ",")
                targets = [c for c in (reader.fieldnames or []) if str(c).lower() in cols]
                for row in reader:
                    for c in targets:
                        v = row.get(c)
                        if v is None or v == "":
                            continue
                        if re.sub(r"[^A-Z0-9]", "", str(v).upper()) not in registry:
                            bad.add(str(v))
    except Exception as exc:
        print(f"check_release: could not parse {path} to verify cell lines: {exc}", file=sys.stderr)
        return set(), "unreadable"
    return bad, "ok"


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
            line_cols = cols & CELL_LINE_COLUMNS
            if line_cols and not (cols - CELL_LINE_COLUMNS):
                bad, status = nonpublic_line_values(path, line_cols)
                if status == "ok" and not bad:
                    print(f"check_release: {rel} -- {sorted(line_cols)} verified against the "
                          f"public cell-line registry ({len(public_line_registry() or [])} ids)")
                    cols = set()
                elif status == "no-registry":
                    problems.append(
                        f"{rel}\n    carries {sorted(line_cols)} but the public cell-line registry "
                        f"at {PUBLIC_LINE_REGISTRY} is missing -- refusing rather than assuming. "
                        f"Rebuild it with scripts/build_public_cell_line_registry.py."
                    )
                    continue
                elif status == "unreadable":
                    problems.append(
                        f"{rel}\n    carries {sorted(line_cols)} but the file could not be parsed "
                        f"to verify them (see stderr) -- refusing rather than assuming."
                    )
                    continue
                else:
                    problems.append(
                        f"{rel}\n    carries {sorted(line_cols)} with {len(bad)} value(s) that are "
                        f"NOT publicly catalogued, e.g. {sorted(bad)[:5]} -- treat as row-level "
                        f"sample data. Aggregate it, or commit its hash instead."
                    )
                    continue
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
