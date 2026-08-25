"""Count data provenance gaps mechanically, so the number does not rest on anyone's word.

Every published number should be traceable to the data and parameters that produced it. This
script checks that by running the checks itself and printing a count, rather than asserting one.
Run it and you get the same answer I do; disagree with a number and you can see exactly which
check produced it.

Four checks, each a yes/no a computer can settle:

  A. UNBACKED TABLE     -- a results table in docs/*.md with no committed file containing its
                           numbers. Nobody can regenerate or diff it.
  B. UNTRACKED INPUT    -- a data path named by a committed script that is neither in the repo
                           nor recorded in the cluster evidence file. A reader cannot obtain it.
  C. CONTRADICTED PARAM -- the repo states a checkpoint or parameter, and the cluster evidence
                           shows a different one was used.
  D. UNRECORDED RUN     -- a pipeline stage whose script does not echo its resolved inputs, so
                           no log can say what it consumed.

The cluster evidence is committed at docs/results/alpine_resolved_ckpt.txt: every distinct
"Resolved:" line found across the 453 job logs on Alpine, with occurrence counts. That file is
the only surviving record of what any generation run actually used, which is itself finding D.

Counts are ROOT CAUSES, not instances. One missing script that explains thirty numbers is one
gap affecting thirty numbers, reported once. Inflating a count by listing symptoms is the
failure this script exists to avoid, so the per-check output prints both figures separately.

Exit code is 0 always; this reports, it does not gate. Use --json for machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

EVIDENCE = Path("docs/results/alpine_resolved_ckpt.txt")
INVENTORY = Path("docs/results/alpine_artifact_inventory.txt")
RESULTS_DOCS = ("docs/tahoe_generation_results.md", "docs/sarcoma_organoids_2024_pathb_results.md")

# Stages whose scripts should echo what they resolved. A stage that consumes a data artifact
# chosen at submit time (via --export) and does not echo it leaves no recoverable record.
STAGES_CONSUMING_EXPORTS = (
    "scripts/alpine/04_stack_generate.sbatch",
    "scripts/alpine/05_stack_score.sbatch",
    "scripts/alpine/06_stack_embed.sbatch",
    "scripts/alpine/07_stack_emb_score.sbatch",
    "scripts/alpine/09_stack_finetune.sbatch",
    "scripts/alpine/11_sarcoma_organoids_2024_generate.sbatch",
    "scripts/alpine/12_sarcoma_organoids_2024_score.sbatch",
    "scripts/alpine/13_sarcoma_organoids_2024_embed.sbatch",
)

CKPT_RE = re.compile(r"finetuned-epoch=\d+-val_loss=[\d.]+\.ckpt")
TABLE_ROW = re.compile(r"^\|.*\|$")
NUMERIC = re.compile(r"-?\d+\.\d+")


@dataclass
class Gap:
    """One root cause, with the count of published numbers it affects."""

    check: str
    title: str
    evidence: str
    affects: int = 0
    detail: list[str] = field(default_factory=list)


def tracked_files(repo: Path) -> set[str]:
    """Every path git tracks, so 'is this obtainable' is answered by git, not by looking."""
    out = subprocess.run(
        ["git", "-C", str(repo), "ls-files"], capture_output=True, text=True, check=True
    )
    return set(out.stdout.split())


def check_a_unbacked_tables(repo: Path, tracked: set[str]) -> list[Gap]:
    """Results tables whose numbers appear in no committed file."""
    committed_numbers: set[str] = set()
    for rel in tracked:
        if rel.endswith((".csv", ".tsv")) and (repo / rel).exists():
            committed_numbers |= set(NUMERIC.findall((repo / rel).read_text()))

    gaps: list[Gap] = []
    for doc in RESULTS_DOCS:
        path = repo / doc
        if not path.exists():
            continue
        unbacked_tables, unbacked_numbers, current = 0, 0, []
        for line in path.read_text().splitlines():
            if TABLE_ROW.match(line.strip()):
                current.extend(NUMERIC.findall(line))
                continue
            if current:
                missing = [n for n in current if n not in committed_numbers]
                # A table is unbacked when most of its numbers appear in no committed file.
                if len(missing) > len(current) / 2:
                    unbacked_tables += 1
                    unbacked_numbers += len(missing)
                current = []
        if unbacked_tables:
            gaps.append(
                Gap(
                    "A",
                    f"{doc}: {unbacked_tables} tables have no committed file behind them",
                    f"searched every committed .csv/.tsv ({len(committed_numbers)} distinct numbers) for each table's values",
                    unbacked_numbers,
                )
            )
    return gaps


def check_b_untracked_inputs(repo: Path, tracked: set[str], evidence: str) -> list[Gap]:
    """Data artifacts a committed script names that a reader cannot obtain."""
    named: dict[str, list[str]] = {}
    for script in sorted((repo / "scripts").rglob("*")):
        if script.suffix not in (".sbatch", ".py") or not script.is_file():
            continue
        if script.name == "audit_provenance.py":
            continue  # this file names example paths in its own docstring
        text = script.read_text()
        # Anchor on a word boundary so leading flag punctuation is not captured, and drop
        # anything that is part of a URL or does not look like a repo-relative path. Without
        # this the audit reported "-stack-aligned/...pkl" (from "--genelist stack-aligned/...")
        # and a zenodo URL as missing artifacts, inflating the count with parsing noise.
        for m in re.findall(r"(?<![\w./-])([\w][\w./-]*\.(?:h5ad|parquet|gctx|pkl|ckpt|tsv))", text):
            at = text.find(m)
            ctx = text[max(0, at - 12) : at]
            prev = text[at - 1] if at else ""
            # $VAR/x.h5ad and ${VAR}/x.h5ad are runtime paths, not repo artifacts.
            if prev in "${" or "http" in ctx or "//" in ctx or m.count("/") > 4:
                continue
            if m.split("/")[0].isupper():
                continue
            # `args.gctx` is Python attribute access, not a path: a lowercase identifier
            # immediately before the dot with no separator anywhere in the match.
            if "/" not in m and prev.isalnum():
                continue
            # printf templates -- "context_by_drug/%04d.h5ad" -- are patterns, not filenames.
            if "%" in text[max(0, at - 4) : at] or m[0].isdigit():
                continue
            named.setdefault(m, []).append(str(script.relative_to(repo)))

    inventory = (repo / INVENTORY).read_text() if (repo / INVENTORY).exists() else ""
    # "manifest.tsv" and "context_by_drug/manifest.tsv" are one artifact referred to two ways.
    with_dir = {Path(a).name for a in named if "/" in a}
    named = {a: s for a, s in named.items() if "/" in a or a not in with_dir}
    unobtainable, untracked_but_present = {}, {}
    for a, s in named.items():
        if a in tracked or (repo / a).exists() or "test" in a.lower():
            continue
        base = Path(a).name
        if a in inventory or f"/{a}" in inventory or base in inventory or a in evidence:
            untracked_but_present[a] = s
        else:
            unobtainable[a] = s

    gaps: list[Gap] = []
    if untracked_but_present:
        gaps.append(
            Gap(
                "B",
                f"{len(untracked_but_present)} inputs exist only on the cluster: present there, untracked here",
                "matched each path against the committed cluster inventory; recoverable by anyone with Alpine access, by nobody else",
                len(untracked_but_present),
                sorted(f"{a}  <- {s[0]}" for a, s in list(untracked_but_present.items())[:6]),
            )
        )
    if unobtainable:
        gaps.append(
            Gap(
                "B",
                f"{len(unobtainable)} inputs are named by scripts but found nowhere -- not in git, not on the cluster",
                "checked git ls-files, the working tree, the committed cluster inventory and the resolved-ckpt evidence",
                len(unobtainable),
                sorted(f"{a}  <- {s[0]}" for a, s in list(unobtainable.items())[:10]),
            )
        )
    return gaps


def check_c_contradicted_params(repo: Path, evidence: str) -> list[Gap]:
    """Checkpoint-to-run pairings the repo states, versus the pairings the logs show.

    An earlier version of this check asked only whether a named checkpoint appeared anywhere in
    the evidence. That passed everything, because every checkpoint the repo names was used by
    SOME run. The contradiction that matters is the PAIRING: the specs say the drug-aligned
    numbers came from finetuned-epoch=4-val_loss=5.0847.ckpt, and the logs show
    OUTDIR=generated_drug_aligned used finetuned-epoch=5-val_loss=6.1078.ckpt. Both checkpoints
    are real; the association is what is wrong.
    """
    pairs: dict[str, set[str]] = {}
    for line in evidence.splitlines():
        if "OUTDIR=" not in line:
            continue
        ck = CKPT_RE.findall(line)
        outdir = re.search(r"OUTDIR=(\S+)", line)
        if ck and outdir:
            pairs.setdefault(outdir.group(1), set()).update(ck)
    if not pairs:
        return []

    # Which checkpoint the evidence says each *drug-aligned* run used.
    drug_aligned = {o: c for o, c in pairs.items() if "drug_aligned" in o}
    truth = {c for cks in drug_aligned.values() for c in cks}
    if not truth:
        return []

    gaps: list[Gap] = []
    sites: dict[str, list[str]] = {}
    for path in sorted(repo.rglob("*")):
        if path.suffix not in (".md", ".sbatch", ".py") or not path.is_file():
            continue
        sp = str(path)
        if "/.git/" in sp or "audit_provenance" in sp or "alpine_resolved" in sp:
            continue
        lines = path.read_text(errors="ignore").splitlines()
        for i, line in enumerate(lines):
            named = CKPT_RE.findall(line)
            if not named:
                continue
            # Look at the line and its neighbours for a drug-alignment context.
            ctx = " ".join(lines[max(0, i - 3) : i + 4]).lower()
            if "drug-align" not in ctx and "drug_align" not in ctx:
                continue
            wrong = [c for c in named if c not in truth]
            if wrong:
                sites.setdefault(wrong[0], []).append(f"{path.relative_to(repo)}:{i + 1}")

    # One root cause per wrongly-paired checkpoint, listing every site that restates it.
    for ck, where in sites.items():
        gaps.append(
            Gap(
                "C",
                f"the repo pairs {ck} with the drug-aligned run; the logs show {sorted(truth)[0]}",
                f"{len(where)} site(s) state it: {', '.join(where)}; evidence pairs "
                + ", ".join(f"{o}={sorted(c)[0]}" for o, c in sorted(drug_aligned.items())),
                len(where),
                where,
            )
        )
    return gaps


def check_d_unrecorded_runs(repo: Path, evidence: str) -> list[Gap]:
    """Stages that take inputs at submit time but never echo what they resolved."""
    silent = []
    for rel in STAGES_CONSUMING_EXPORTS:
        p = repo / rel
        if not p.exists():
            continue
        if "Resolved:" not in p.read_text():
            silent.append(rel)
    if not silent:
        return []
    return [
        Gap(
            "D",
            f"{len(silent)} pipeline stages accept inputs via --export but never echo them",
            "grepped each stage for a 'Resolved:' echo; the committed cluster evidence contains only "
            f"{len([ln for ln in evidence.splitlines() if ln.strip()])} distinct Resolved lines across 453 job logs",
            len(silent),
            silent,
        )
    ]


def main() -> None:
    """Run all four checks and print the count."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    repo = args.repo.resolve()

    ev_path = repo / EVIDENCE
    if not ev_path.exists():
        raise SystemExit(f"missing cluster evidence: {EVIDENCE}. Nothing can be checked without it.")
    evidence = ev_path.read_text()
    tracked = tracked_files(repo)

    gaps = (
        check_a_unbacked_tables(repo, tracked)
        + check_b_untracked_inputs(repo, tracked, evidence)
        + check_c_contradicted_params(repo, evidence)
        + check_d_unrecorded_runs(repo, evidence)
    )

    if args.json:
        print(json.dumps([g.__dict__ for g in gaps], indent=2))
        return

    by_check: dict[str, list[Gap]] = {}
    for g in gaps:
        by_check.setdefault(g.check, []).append(g)

    names = {
        "A": "unbacked results tables",
        "B": "untracked inputs",
        "C": "contradicted parameters",
        "D": "unrecorded runs",
    }
    print("DATA PROVENANCE AUDIT")
    print("=" * 74)
    for k in "ABCD":
        rows = by_check.get(k, [])
        print(f"\n[{k}] {names[k]}: {len(rows)} root cause(s), affecting {sum(r.affects for r in rows)} items")
        for g in rows:
            print(f"    - {g.title}")
            print(f"      evidence: {g.evidence}")
            for d in g.detail:
                print(f"        {d}")
    print("\n" + "=" * 74)
    print(f"TOTAL: {len(gaps)} root causes, affecting {sum(g.affects for g in gaps)} items")
    print("Re-run this script to reproduce. Every count above comes from a file comparison,")
    print("not from a judgement.")


if __name__ == "__main__":
    main()
