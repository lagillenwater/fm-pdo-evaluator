"""CI gate: every committed result is regenerable, and every asserted claim cites one.

Two checks, both mechanical, both cheap enough to run on every push.

**Check A -- committed results are complete.** Each ``docs/results/<name>.<ext>`` must have a
matching ``<name>.provenance.json``, that record must name a script that exists in the tree,
and it must list at least one input. A result whose producer is missing cannot be reproduced,
so it is an assertion wearing a table's clothes.

**Check B -- assertions cite evidence.** Prose that claims something was established --
"verified", "confirmed", "we measured", "empirically", "calibrated" -- must carry an
``[evidence: <name>]`` tag naming a promoted result, on the same line or the one before.

Check B exists because of a measured failure. On 2026-08-24 an audit found 78 of 106 claims in
this repo had no artifact a reader could open, and four of the worst cases were prose sitting
directly beside committed data that said something different. Among them: a comment asserting
"verified empirically: r2 stayed negative even at 30x effect" with no script, test or log
anywhere; and a module diagnosis, "confirmed gone from amem's module tree", that was simply
untrue and cost three failed job submissions before anyone rechecked it.

The tag is deliberately cheap to add and impossible to add honestly without having promoted
something. Writing "verified" costs nothing; writing an evidence tag requires the result to exist.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RESULTS_DIR = Path("docs/results")

# Words that claim an investigation happened. Deliberately narrow: this must flag assertions,
# not ordinary description, or it becomes noise and gets disabled.
ASSERTION = re.compile(
    r"\b(verified|confirmed|we measured|measured empirically|empirically|calibrated against)\b",
    re.IGNORECASE,
)
EVIDENCE_TAG = re.compile(r"\[evidence:\s*([A-Za-z0-9_.\-]+)\s*\]")
# A second, weaker tag for OPERATIONAL facts -- a QoS floor, a partition's GPU type, why a job
# died. These are not scientific results and promoting a CSV for them would be silly, but they
# are still checkable: the tag must name the command or job id that shows it, so a reader can
# rerun `sacctmgr show qos` or `sacct -j 31418001` instead of trusting the sentence. The gate
# only requires that a method is named; it cannot rerun it.
CHECKED_TAG = re.compile(r"\[checked:\s*([^\]]+)\]")
# Prose that explicitly disclaims having evidence is fine, and saying so is the behaviour we
# want to encourage rather than punish.
# Prose that explicitly says it lacks evidence is exactly the behaviour to encourage, so it
# passes. "never verified", "not independently confirmed" and friends are honest disclaimers,
# and several exist only because a previously-asserted claim was corrected.
DISCLAIMED = re.compile(
    r"\b(not established|NOT established|unverified|never verified|not independently confirmed"
    r"|no artifact|never committed|do not assert|nobody could find|cannot be checked)\b"
)

SEARCH_GLOBS = ("docs/**/*.md", "src/**/*.py", "scripts/**/*.py", "scripts/alpine/*.sbatch")

# Only ONE exclusion, and it is a true false-positive: this file's own docstring necessarily
# contains the words it searches for.
#
# docs/superpowers/** was excluded on 2026-08-25 and reinstated the same day. The argument for
# excluding it -- that dated design records should not be rewritten to satisfy a gate -- is
# sound about EDITING them and does not license exempting them from scrutiny. 67 of the
# original 107 hits lived there, so the exclusion was most of a 107->21 "reduction" that
# involved almost no work. It also removed exactly the claims that matter most: the 2026-08-21
# spec's "Ridge is well-posed for p >> n" drove the no-truncation policy and was later
# disproven for this regime; a sci-Plex crosstab claim in a spec invalidated published numbers
# and forced a GPU rerun, in a commit that touched only the spec.
#
# The right treatment is a STATUS NOTE, not an edit to the original claim: leave what was
# believed, and record next to it whether it was ever checked and whether it still holds. The
# DISCLAIMED pattern already accepts that language, so an honest note passes the gate.
EXCLUDE = ("scripts/check_evidence.py",)


def load_promoted(repo: Path) -> set[str]:
    """Names of every promoted result, from the provenance records."""
    return {p.stem.replace(".provenance", "") for p in (repo / RESULTS_DIR).glob("*.provenance.json")}


def check_results(repo: Path) -> list[str]:
    """Check A: every committed result carries a usable provenance record."""
    problems: list[str] = []
    d = repo / RESULTS_DIR
    if not d.exists():
        return problems
    for result in sorted(d.iterdir()):
        if result.suffix == ".json" or result.name.startswith("."):
            continue
        prov_path = d / f"{result.stem}.provenance.json"
        if not prov_path.exists():
            problems.append(f"{result.relative_to(repo)}: no provenance record")
            continue
        prov = json.loads(prov_path.read_text())
        script = prov.get("script")
        if not script or not (repo / script).exists():
            problems.append(f"{prov_path.relative_to(repo)}: script {script!r} does not exist")
        if not prov.get("inputs"):
            problems.append(f"{prov_path.relative_to(repo)}: no inputs recorded")
        if not prov.get("claim"):
            problems.append(f"{prov_path.relative_to(repo)}: no claim recorded")
    return problems


def check_assertions(repo: Path, promoted: set[str], *, strict: bool) -> list[str]:
    """Check B: assertion words carry an evidence tag naming a promoted result."""
    problems: list[str] = []
    for glob in SEARCH_GLOBS:
        for path in sorted(repo.glob(glob)):
            rel_posix = path.relative_to(repo).as_posix()
            if RESULTS_DIR.as_posix() in rel_posix or any(e in rel_posix for e in EXCLUDE):
                continue
            try:
                lines = path.read_text().splitlines()
            except UnicodeDecodeError:
                continue
            for i, line in enumerate(lines):
                if not ASSERTION.search(line) or DISCLAIMED.search(line):
                    continue
                window = line + ("\n" + lines[i - 1] if i else "")
                tags = EVIDENCE_TAG.findall(window)
                rel = f"{path.relative_to(repo)}:{i + 1}"
                if not tags:
                    if strict and not CHECKED_TAG.search(window):
                        problems.append(
                            f"{rel}: asserts evidence with no [evidence: ...] or [checked: ...] tag"
                        )
                elif unknown := [t for t in tags if t not in promoted]:
                    problems.append(f"{rel}: cites unpromoted result(s) {unknown}")
    return problems


def main() -> int:
    """Run both checks; return non-zero when anything fails."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="also fail on assertion words with NO tag. Off by default so the gate can be "
        "adopted on new writing first and the existing backlog burned down deliberately.",
    )
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    repo = args.repo.resolve()

    promoted = load_promoted(repo)
    problems = check_results(repo) + check_assertions(repo, promoted, strict=args.strict)

    print(f"promoted results: {len(promoted)}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
        return 1
    print("evidence check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
