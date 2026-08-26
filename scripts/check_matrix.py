"""Fail a run that does not cover the declared model matrix.

The problem this closes: nothing said which model variants a check was supposed to include, so
each job carried whatever --generated-dir and --stack-emb flags it had been written with. 05,
07, 17 and 19 all passed only `generated_agg`, and the Check-1b permutation null therefore
covered ONE of the two Stack checkpoints the published table reports -- the weaker one. The run
was not broken. There was nothing to check it against, so a missing variant and a deliberate
choice looked identical.

data/model_matrix.yaml declares them. This compares a result table against that declaration and
exits non-zero when a variant that belongs in the check is absent. Run it on a result CSV:

    python scripts/check_matrix.py --check check1b --result promote/de_permutation_null.csv

or list what a check should contain, to build the run in the first place:

    python scripts/check_matrix.py --check check2 --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # a gate must not fail open because a package is absent
    yaml = None

import pandas as pd

MATRIX = Path("data/model_matrix.yaml")


def load_matrix(path: Path) -> dict:
    """Read the matrix, with a fallback for interpreters without pyyaml.

    Mirrors scripts/check_release.py: this runs wherever a job runs, which is not necessarily
    the project environment. The fallback parses the shape actually used -- a list of variants
    with `id` and `checks` -- and nothing more.
    """
    text = path.read_text()
    if yaml is not None:
        return yaml.safe_load(text)
    out: dict = {"variants": [], "excluded": []}
    section, entry = None, None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("variants:"):
            section = "variants"; continue
        if line.startswith("excluded:"):
            section = "excluded"; continue
        s = line.strip()
        if s.startswith("- id:") and section:
            entry = {"id": s.split(":", 1)[1].strip()}
            out[section].append(entry)
        elif entry is not None and s.startswith("checks:"):
            entry["checks"] = [c.strip() for c in s.split("[", 1)[-1].rstrip("]").split(",") if c.strip()]
        elif entry is not None and s.startswith("aliases:"):
            entry["aliases"] = [c.strip() for c in s.split("[", 1)[-1].rstrip("]").split(",") if c.strip()]
    return out


def expected_for(matrix: dict, check: str) -> list[str]:
    """Variant ids that belong in this check."""
    return sorted(
        v["id"] for v in matrix.get("variants", []) if check in (v.get("checks") or [])
    )


def alias_map(matrix: dict) -> dict[str, str]:
    """Historical source name -> canonical variant id.

    Different scripts name the same artifact differently: the Check-1b null takes its source
    name from a parquet filename (`stack`), the Check-2 grid from a --generated-dir label
    (`stack_cytokine`). Without this, one checkpoint under two names reads as one variant
    missing and one undeclared -- two false findings from a naming difference.
    """
    out: dict[str, str] = {}
    for v in matrix.get("variants", []):
        for a in v.get("aliases") or []:
            out[str(a)] = str(v["id"])
    return out


def main() -> int:
    """Compare a result table against the declaration."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", required=True, choices=["check1", "check1b", "check2"])
    ap.add_argument("--result", type=Path, default=None, help="result CSV with a 'source' column")
    ap.add_argument("--list", action="store_true", help="print what this check should contain")
    ap.add_argument("--matrix", type=Path, default=MATRIX)
    ap.add_argument(
        "--allow-missing",
        default="",
        help="comma-separated ids that may be absent, e.g. a variant still running. Each one "
        "has to be named, so a gap is always a stated decision.",
    )
    args = ap.parse_args()

    matrix = load_matrix(args.matrix)
    expected = expected_for(matrix, args.check)

    if args.list or not args.result:
        print(f"{args.check} should contain {len(expected)} variants:")
        for e in expected:
            print(f"  {e}")
        return 0

    df = pd.read_csv(args.result)
    col = "source" if "source" in df.columns else df.columns[0]
    # Noise-control rows are derived per representation, not declared variants.
    aliases = alias_map(matrix)
    present = {
        aliases.get(str(s), str(s)) for s in df[col].unique() if not str(s).endswith("_random")
    }
    allowed = {a.strip() for a in args.allow_missing.split(",") if a.strip()}

    missing = [e for e in expected if e not in present and e not in allowed]
    undeclared = sorted(present - set(expected) - allowed)

    print(f"{args.check}: {len(present)} variants present, {len(expected)} declared")
    if undeclared:
        print(f"  present but NOT declared: {undeclared}")
        print("  -> add them to data/model_matrix.yaml, or they are silently unreviewed")
    if allowed:
        print(f"  waived: {sorted(allowed)}")
    if missing:
        print(f"\nMISSING {len(missing)} declared variant(s): {missing}")
        print("Either score them, or waive them by name with --allow-missing.")
        return 1
    print("  matrix covered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
