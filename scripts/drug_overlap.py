"""Drug overlap between candidate delta-transfer corpora and the organoid panel.

Path B learns a baseline -> delta map on a corpus of real drug perturbations and applies it to
organoids. Gene coverage decides what can be SCORED; drug overlap decides whether the corpus
can supply the organoids' drugs at all, and a corpus that shares no drugs is unusable however
many genes it has. L1000 has 25,157 perturbagens across 76 lines but only 978 measured genes;
Tahoe has a full transcriptome across 50 lines but a much smaller drug panel. That trade is
the whole question, and it cannot be read off the gene tables.

Drugs are matched on PubChem CID, not name. Names disagree across cohorts in every direction
-- salt forms, hyphenation, brand vs generic, upstream truncation -- so a name join
understates overlap in a way that silently argues against whichever corpus spells things
differently. Raw counts and the CID-mapped counts are both reported so the mapping's own
contribution is visible.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


def norm_name(s: object) -> str:
    """Lowercase alphanumeric, for the name-based fallback join."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _find(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Locate a column CASE-INSENSITIVELY, ignoring separators.

    Matching exactly cost a whole run: the organoid screen's column is `Drug_Name`, which is
    not `drug_name` or `DRUG_NAME`, so the cohort reported zero drugs and every overlap
    involving it read as a clean 0. A zero from a missed column is indistinguishable from a
    zero that means "these cohorts share no drugs", and the second is a finding.
    """
    norm = {str(c).lower().replace("_", "").replace(" ", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace("_", "").replace(" ", "")
        if key in norm:
            return norm[key]
    return None


def cids_from(df: pd.DataFrame) -> set[str]:
    """PubChem CIDs from whichever column carries them."""
    for c in ("pubchem_cid", "cid", "drug"):
        col = _find(df, (c,))
        if col is not None:
            vals = {str(v).strip() for v in df[col].dropna()}
            numeric = {v.split(".")[0] for v in vals if v.replace(".", "", 1).isdigit()}
            if len(numeric) > len(vals) / 2:
                return numeric
    return set()


def names_from(df: pd.DataFrame) -> set[str]:
    """Drug names from whichever column carries them."""
    for c in ("drug_name", "drug", "pert_iname", "compound", "product_name", "name"):
        col = _find(df, (c,))
        if col is not None:
            vals = {norm_name(v) for v in df[col].dropna()}
            if vals and not all(v.isdigit() for v in vals):
                return vals - {""}
    return set()


def git_sha() -> str:
    """The commit this overlap was measured at."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    """Report each cohort's drug axis and the pairwise overlaps, by CID and by name."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", action="append", required=True, help="label=path (parquet/tsv/csv)")
    ap.add_argument("--generated-dir", action="append", default=[],
                    help="label=dir whose *.h5ad FILENAMES are drug names (the organoid panel)")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    cids: dict[str, set[str]] = {}
    names: dict[str, set[str]] = {}
    meta: dict[str, dict] = {}

    for spec in args.source:
        label, _, path = spec.partition("=")
        p = Path(path)
        if not p.exists():
            print(f"  SKIP {label}: {p} not present")
            continue
        if p.suffix in (".tsv", ".txt"):
            df = pd.read_csv(p, sep="\t", low_memory=False)
            if not any(str(c).lower().replace("_", "") in
                       ("drug", "drugname", "cid", "pubchemcid", "pertiname") for c in df.columns):
                # headerless: the first data row became the column labels
                df = pd.read_csv(p, sep="\t", header=None, names=["drug_name", "pubchem_cid"])
        elif p.suffix == ".csv":
            df = pd.read_csv(p, low_memory=False)
        else:
            df = pd.read_parquet(p)
        cids[label], names[label] = cids_from(df), names_from(df)
        meta[label] = {"path": str(p), "columns": [str(c) for c in df.columns][:12],
                       "n_cid": len(cids[label]), "n_name": len(names[label])}
        print(f"  {label:<22} {len(cids[label]):>6} CIDs  {len(names[label]):>6} names  "
              f"cols={[str(c) for c in df.columns][:6]}")
        print(f"  {'':22} cid ex={sorted(cids[label])[:3]} name ex={sorted(names[label])[:3]}")
        if not cids[label] and not names[label]:
            print(f"  {'':22} WARNING: no drug axis found -- every overlap for {label} will be a")
            print(f"  {'':22} false zero. Columns present: {[str(c) for c in df.columns][:12]}")

    for spec in args.generated_dir:
        label, _, d = spec.partition("=")
        dd = Path(d)
        if not dd.exists():
            print(f"  SKIP {label}: {dd} not present")
            continue
        nm = {norm_name(f.stem) for f in dd.glob("*.h5ad")}
        cids[label], names[label] = set(), nm
        meta[label] = {"path": str(dd), "n_cid": 0, "n_name": len(nm)}
        print(f"  {label:<22} {0:>6} CIDs  {len(nm):>6} names  (from filenames)")
        print(f"  {'':22} name ex={sorted(nm)[:4]}")

    labels = list(names)
    for title, sets in (("BY PUBCHEM CID", cids), ("BY NORMALISED NAME", names)):
        print(f"\n================ {title} ================")
        tab = pd.DataFrame(
            [[len(sets[a] & sets[b]) for b in labels] for a in labels],
            index=labels, columns=labels,
        )
        print(tab.to_string())
        args.out_dir.mkdir(parents=True, exist_ok=True)
        tab.to_csv(args.out_dir / f"drug_overlap_{title.split()[-1].lower()}.csv")

    (args.out_dir / "drug_overlap.params.json").write_text(
        json.dumps({"git_sha": git_sha(), "sources": meta}, indent=2) + "\n"
    )
    print(f"\nwrote drug overlap tables to {args.out_dir}")


if __name__ == "__main__":
    main()
