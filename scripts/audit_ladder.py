"""Audit every ladder rung for controls and provenance, mechanically.

Reviewing one's own design by reading it is how the gaps in this branch survived: the panel was
imported and never called, Check 2's positive control vanished in the array conversion, and
rung 0's ceiling had no null while being used as a denominator. Each looked fine in prose. So
this checks the code and the artifacts instead of the intention.

Six axes per rung:

  controls_floor      a line-independent floor (prior / observed_delta) is scored
  controls_negative   a null or noise control is computed IN that rung, not borrowed
  controls_positive   a planted or known-answer control the rung must recover
  prov_params         the run writes a sidecar carrying git sha and resolved arguments
  prov_panel          the gene panel is pinned AND asserted, not merely computed
  prov_drugs          the drugs scored are recorded, with how they were matched

WHAT THIS CANNOT DO: it checks that the machinery is present and wired, not that it is correct.
A token can be present and misused. It is a floor on rigor, not a proof of it -- and it is
reported that way rather than as a clean bill of health.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

RUNGS: dict[str, dict] = {
    "rung0_ceiling": {
        "what": "Tahoe replicate ceiling (plate split-half)",
        "scripts": ["scripts/delta_reproducibility.py"],
        # No predictor, so no floor to beat and no positive control to recover: the measurement
        # IS the reliability, and its mismatched-pair null is what makes it meaningful.
        "na": ["controls_floor", "controls_positive"],
    },
    "rung1_check1": {
        "what": "held-out Tahoe line, delta fidelity",
        "scripts": ["scripts/score_generation_eval.py", "scripts/de_permutation_null.py"],
    },
    "rung2_transfer": {
        "what": "cross-platform + granularity transfer",
        "scripts": ["scripts/rung2_plan.py", "scripts/rung2_score_one.py", "scripts/rung2_gather.py"],
    },
    "rung3_check2": {
        "what": "GDSC2 viability",
        "scripts": ["scripts/check2_plan.py", "scripts/check2_score_one.py", "scripts/check2_gather.py"],
    },
    "rung3_ceiling": {
        "what": "screen-agreement ceiling",
        "scripts": ["scripts/label_ceiling.py"],
        # Compares AUC labels between independent screens: no predictor and no gene axis.
        "na": ["controls_floor", "controls_positive", "prov_panel"],
    },
    "rung4_organoid": {
        "what": "organoid viability (FROZEN: embargoed holdout)",
        "scripts": ["scripts/score_viability_adapters.py"],
    },
}

CHECKS: dict[str, list[str]] = {
    "controls_floor": [r'"prior"', r"'prior'", r'"observed_delta"', r"drug_mean"],
    "controls_negative": [r"n_perm", r"null", r"shuffle", r"permut", r"_random", r"shuffled"],
    "controls_positive": [r"planted", r"plant_interaction", r"spearman_brown", r"splithalf"],
    "prov_params": [r"git_sha", r"params\.json", r"rev-parse"],
    "prov_panel": [r"common_gene_panel", r"assert_common_genes", r"panel_size", r"panel_file", r"n_genes"],
    "prov_drugs": [r"shared_drugs", r"pert_map", r"drug", r"pubchem"],
}


def git_sha() -> str:
    """The commit this audit ran at."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    """Score every rung on every axis and write the audit table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("docs/results"))
    ap.add_argument("--out-dir", type=Path, default=Path("promote/ladder_audit"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rung, spec in RUNGS.items():
        present = [Path(s) for s in spec["scripts"] if Path(s).exists()]
        missing = [s for s in spec["scripts"] if not Path(s).exists()]
        blob = "\n".join(p.read_text() for p in present)
        row = {"rung": rung, "what": spec["what"],
               "scripts_present": len(present), "scripts_missing": ";".join(missing)}
        na = set(spec.get("na", []))
        for axis, pats in CHECKS.items():
            if axis in na:
                row[axis] = "n/a"
                row[f"{axis}_ok"] = True
                continue
            hits = sorted({p for p in pats if re.search(p, blob)})
            row[axis] = "; ".join(hits) if hits else ""
            row[f"{axis}_ok"] = bool(hits)
        rows.append(row)

    df = pd.DataFrame(rows)
    print("=" * 100)
    print("LADDER AUDIT -- machinery present and wired (NOT a correctness proof)")
    print("=" * 100)
    hdr = f"{'rung':<18} {'floor':<6} {'neg':<6} {'pos':<6} {'params':<7} {'panel':<6} {'drugs':<6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def m(k: str) -> str:
            if r[k] == "n/a":
                return "n/a"
            return "YES" if r[f"{k}_ok"] else "GAP"
        print(f"{r['rung']:<18} {m('controls_floor'):<6} {m('controls_negative'):<6} "
              f"{m('controls_positive'):<6} {m('prov_params'):<7} {m('prov_panel'):<6} "
              f"{m('prov_drugs'):<6}")

    gaps = []
    for r in rows:
        for axis in CHECKS:
            if r[axis] != "n/a" and not r[f"{axis}_ok"]:
                gaps.append({"rung": r["rung"], "axis": axis})
        if r["scripts_missing"]:
            gaps.append({"rung": r["rung"], "axis": f"scripts missing: {r['scripts_missing']}"})
    print(f"\n{len(gaps)} gap(s):")
    for g in gaps:
        print(f"  {g['rung']:<18} {g['axis']}")

    # STALENESS. A result can carry a flawless sidecar and still be invalid, because a defect it
    # was produced under has since been fixed. Provenance says where a number came from; it does
    # not say whether that number should still be believed. Each fix below invalidates results
    # produced before it, and the check2 grid at job 31655278 is the live example: promoted with
    # full provenance, and produced before the gene panel was wired AND before the positive
    # control was restored to the array path.
    # A fix invalidates only the results produced BY THE PATH it touched. Flagging every result
    # older than the earliest fix marks a gene-overlap measurement as superseded by a panel fix
    # it never used -- noise that buries the two results genuinely at risk.
    SUPERSEDED_BEFORE = [
        (31663950, ("score_generation_eval", "check2_plan", "check2_score_one"),
         "common gene panel wired into both scoring paths"),
        (31664927, ("check2_plan", "check2_score_one"),
         "planted positive control restored to the Check-2 array path"),
        (31665456, ("delta_reproducibility",),
         "mismatched-pair null added to the rung-0 ceiling"),
    ]

    # Promoted evidence: every result must carry a provenance sidecar naming a job.
    print("\n--- promoted results: provenance sidecars ---")
    prov_rows = []
    if args.results_dir.exists():
        for f in sorted(args.results_dir.glob("*.csv")):
            side = f.with_suffix("").with_suffix(".provenance.json")
            side = f.parent / (f.stem + ".provenance.json")
            ok = side.exists()
            job = script = ""
            n_inputs = 0
            if ok:
                try:
                    j = json.loads(side.read_text())
                    job = str(j.get("job_id") or j.get("slurm_job_id") or "")
                    script = str(j.get("script", ""))
                    n_inputs = len(j.get("inputs", []) or j.get("input", []) or [])
                except Exception:
                    ok = False
            reasons = [
                why for cut, scripts, why in SUPERSEDED_BEFORE
                if job.isdigit() and int(job) < cut and any(s in script for s in scripts)
            ]
            stale = bool(reasons)
            prov_rows.append({"result": f.name, "sidecar": ok, "job_id": job,
                              "script": script, "n_inputs": n_inputs, "superseded": stale,
                              "superseded_why": "; ".join(reasons)})
            if not ok:
                flag = "   <-- NO SIDECAR"
            elif not job:
                flag = "   <-- no job id"
            elif stale:
                flag = "   <-- SUPERSEDED (predates a fix; do not cite)"
            else:
                flag = ""
            print(f"  {f.name:<48} job={job or '?':<12}{flag}")
    prov_df = pd.DataFrame(prov_rows)

    df.to_csv(args.out_dir / "ladder_audit.csv", index=False)
    prov_df.to_csv(args.out_dir / "promoted_provenance.csv", index=False)
    (args.out_dir / "ladder_audit.params.json").write_text(
        json.dumps({"git_sha": git_sha(), "n_rungs": len(rows), "n_gaps": len(gaps),
                    "gaps": gaps, "axes": list(CHECKS)}, indent=2) + "\n"
    )
    n_stale = int(prov_df["superseded"].sum()) if len(prov_df) else 0
    print(f"\n{n_stale} promoted result(s) SUPERSEDED by a later fix:")
    for r in prov_rows:
        if r["superseded"]:
            print(f"  {r['result']}  (job {r['job_id']})")
    for r in prov_rows:
        if r["superseded"]:
            print(f"    reason: {r['superseded_why']}")

    print(f"\nwrote {args.out_dir}/ladder_audit.csv and promoted_provenance.csv")


if __name__ == "__main__":
    main()
