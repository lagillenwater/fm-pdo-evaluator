"""Is rung 4 feasible, and at what n? Answers the three open questions with artifacts.

Rung 4 tests cross-modality transfer on the embargoed sarcoma organoid cohort. Three facts
decide whether it can be built at all, and none had been measured:

  1. HOW MANY ORGANOIDS have BOTH an expression profile and drug-screen rows. The screen covers
     94 organoids and the expression artifact carries 17, so the usable n is the intersection --
     not 94, and not necessarily 17 either. Every power statement about rung 4 depends on it.
  2. WHETHER A CEILING EXISTS. The aggregated screen holds exactly one row per
     (organoid, drug), so there is no test-retest reliability to compute and rung 4 has no
     ceiling of its own. If raw dose-response points survive upstream, bootstrapping the curve
     fit would give per-pair uncertainty instead. This checks every table for that granularity.
  3. WHICH DRUGS SURVIVE once the representation source is fixed. A representation built from
     L1000 can only score organoid drugs L1000 profiled; the same for GDSC2. The intersection,
     not the screen's 34, is rung 4's drug axis.

Writes a per-organoid table and a summary. Emits no interpretation -- the point is that the
numbers behind the rung-4 decision are on disk rather than in prose.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
from pathlib import Path

import pandas as pd


def norm_name(s: object) -> str:
    """Lowercase alphanumeric, for joining drug names across cohorts."""
    return "".join(c for c in str(s).lower() if c.isalnum())


def norm_id(s: object) -> str:
    """Uppercase alphanumeric, for joining sample identifiers."""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def h5ad_obs_index(path: Path) -> list[str]:
    """obs_names of an .h5ad, read from metadata only."""
    import h5py

    with h5py.File(path, "r") as f:
        obs = f["obs"]
        idx = obs.attrs.get("_index", "_index")
        idx = idx.decode() if isinstance(idx, bytes) else str(idx)
        return [x.decode() if isinstance(x, bytes) else str(x) for x in obs[idx][:]]


def git_sha() -> str:
    """The commit this was measured at."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    """Measure rung-4's usable cohort, ceiling availability and drug axis."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screen", required=True, type=Path)
    ap.add_argument("--expression", required=True, type=Path, help="stack_input_sarcoma.h5ad")
    ap.add_argument("--tables-dir", type=Path, default=Path("data/raw/soragni/tables"))
    ap.add_argument("--l1000-pert-info", type=Path, default=None)
    ap.add_argument("--gdsc2-compounds", type=Path, default=None)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Q1: usable cohort -------------------------------------------------------------
    scr = pd.read_parquet(args.screen)
    scr_ids = {norm_id(s): str(s) for s in scr["Sample_ID"].dropna().unique()}
    expr_raw = h5ad_obs_index(args.expression)
    expr_ids = {norm_id(s): str(s) for s in expr_raw}
    both = sorted(set(scr_ids) & set(expr_ids))
    print(f"Q1  screen organoids     : {len(scr_ids)}")
    print(f"Q1  expression organoids : {len(expr_ids)}  e.g. {sorted(expr_ids.values())[:4]}")
    print(f"Q1  BOTH (rung-4 n)      : {len(both)}")
    if not both:
        print("Q1  WARNING: zero overlap. Identifier namespaces may differ; "
              f"screen e.g. {sorted(scr_ids.values())[:3]}, expression e.g. {sorted(expr_ids.values())[:3]}")

    per = []
    for k in sorted(set(scr_ids) | set(expr_ids)):
        rows = scr[scr["Sample_ID"].map(norm_id) == k] if k in scr_ids else scr.iloc[0:0]
        per.append({
            "organoid": scr_ids.get(k) or expr_ids.get(k),
            "has_screen": k in scr_ids,
            "has_expression": k in expr_ids,
            "n_drugs_screened": int(rows["Drug_Name"].nunique()),
            "diagnosis": str(rows["Diagnosis"].iloc[0]) if len(rows) else "",
        })
    per_df = pd.DataFrame(per)
    per_df.to_csv(args.out_dir / "organoid_cohort.csv", index=False)

    usable = per_df[per_df["has_screen"] & per_df["has_expression"]]
    print(f"Q1  drugs per usable organoid: min={usable['n_drugs_screened'].min() if len(usable) else 0} "
          f"median={usable['n_drugs_screened'].median() if len(usable) else 0}")

    # ---- Q2: is there any replicate or dose-level granularity for a ceiling? -------------
    print("\nQ2  scanning every table for replicate / dose granularity")
    gran = []
    for f in sorted(args.tables_dir.glob("*.parquet")) if args.tables_dir.exists() else []:
        try:
            d = pd.read_parquet(f)
        except Exception as exc:
            gran.append({"table": f.name, "error": str(exc)[:80]})
            continue
        cols = [str(c) for c in d.columns]
        dose_like = [c for c in cols if re.search(r"dose|conc|replicate|rep\b|well|plate|run", c, re.I)]
        entry = {"table": f.name, "rows": int(len(d)), "columns": ";".join(cols[:14]),
                 "dose_or_replicate_columns": ";".join(dose_like)}
        if {"Sample_ID", "Drug_Name"} <= set(cols):
            g = d.groupby(["Sample_ID", "Drug_Name"]).size()
            entry["max_rows_per_pair"] = int(g.max())
            entry["pairs_with_replicates"] = int((g > 1).sum())
        gran.append(entry)
        print(f"    {f.name:<34} rows={len(d):>6} dose/replicate cols={dose_like or 'NONE'}")
    pd.DataFrame(gran).to_csv(args.out_dir / "organoid_table_granularity.csv", index=False)

    # ---- Q3: drug axis under each candidate representation source -----------------------
    print("\nQ3  drug axis by representation source")
    org_drugs = {norm_name(d): str(d).strip() for d in scr["Drug_Name"].dropna().unique()}
    coverage = {}
    if args.l1000_pert_info and args.l1000_pert_info.exists():
        with gzip.open(args.l1000_pert_info, "rt") as fh:
            pi = pd.read_csv(fh, sep="\t", low_memory=False)
        have = {norm_name(x) for x in pi["pert_iname"].dropna()}
        coverage["l1000"] = sorted(v for k, v in org_drugs.items() if k in have)
    if args.gdsc2_compounds and args.gdsc2_compounds.exists():
        gd = pd.read_csv(args.gdsc2_compounds, low_memory=False)
        have = {norm_name(x) for x in gd["DRUG_NAME"].dropna()}
        for c in ("SYNONYMS",):
            if c in gd.columns:
                for s in gd[c].dropna():
                    have |= {norm_name(x) for x in str(s).split(",")}
        coverage["gdsc2"] = sorted(v for k, v in org_drugs.items() if k in have)
    for src, hits in coverage.items():
        print(f"    {src:<8} covers {len(hits)}/{len(org_drugs)} organoid drugs")
    if len(coverage) == 2:
        a, b = coverage["l1000"], coverage["gdsc2"]
        print(f"    both     {len(set(a) & set(b))}   either {len(set(a) | set(b))}")

    summary = {
        "git_sha": git_sha(),
        "q1_screen_organoids": len(scr_ids),
        "q1_expression_organoids": len(expr_ids),
        "q1_usable_n": len(both),
        "q1_usable_ids": [scr_ids[k] for k in both],
        "q2_any_replicate_granularity": bool(
            any(e.get("pairs_with_replicates", 0) > 0 for e in gran)
        ),
        "q2_tables_scanned": [e["table"] for e in gran],
        "q3_organoid_drugs": len(org_drugs),
        "q3_coverage": {k: len(v) for k, v in coverage.items()},
        "q3_covered_drugs": coverage,
        "inputs": {"screen": str(args.screen), "expression": str(args.expression)},
    }
    (args.out_dir / "rung4_feasibility.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([{
        "question": "Q1 usable organoids (screen AND expression)", "value": len(both),
    }, {
        "question": "Q2 replicate granularity available for a ceiling",
        "value": int(summary["q2_any_replicate_granularity"]),
    }, {
        "question": "Q3 organoid drugs covered by L1000",
        "value": len(coverage.get("l1000", [])),
    }, {
        "question": "Q3 organoid drugs covered by GDSC2",
        "value": len(coverage.get("gdsc2", [])),
    }]).to_csv(args.out_dir / "rung4_feasibility_summary.csv", index=False)
    print(f"\nwrote artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
