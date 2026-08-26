"""Emit the list of publicly-catalogued cell lines, so the release gate can VERIFY rather than trust.

The release gate treats any `line` or `cell_line` column as row-level sample data and refuses
it. That is right for SARC0065, a patient-derived organoid under embargo, and wrong for A549,
a cell line whose identity has been public for decades. The column name cannot distinguish
them, so something has to.

The weak fix is a per-file waiver declaring "these rows are public" -- which is a promise, not
a check, and a file that later gains an organoid row keeps the waiver. This instead builds a
registry of identifiers that ARE publicly catalogued, from the DepMap model table and the LINCS
instance table, so the gate can test each value. A public line passes because it is in the
registry; an organoid fails because it is not. Nothing is taken on trust.

The output is public reference data of the same kind as data/static/drug_xref.parquet: a list
of cell-line names already published by DepMap and LINCS, carrying no measurements.
"""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path

import pandas as pd


def norm(s: object) -> str:
    """Uppercase alphanumeric, matching how the gate and fmharness.deltas normalise lines."""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def main() -> None:
    """Collect public cell-line identifiers from DepMap and LINCS into one sorted list."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-csv", type=Path, default=Path("data/raw/gdsc2_sarcoma/depmap/Model.csv"))
    ap.add_argument("--inst-info", type=Path, default=Path("GSE92742_Broad_LINCS_inst_info.txt.gz"))
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    names: set[str] = set()
    if args.model_csv.exists():
        m = pd.read_csv(args.model_csv, low_memory=False)
        for col in ("ModelID", "StrippedCellLineName", "CellLineName"):
            if col in m.columns:
                names |= {norm(v) for v in m[col].dropna()}
        print(f"DepMap Model.csv: {len(m)} models")
    else:
        print(f"  MISSING {args.model_csv}")

    if args.inst_info.exists():
        with gzip.open(args.inst_info, "rt") as fh:
            inst = pd.read_csv(fh, sep="\t", low_memory=False, usecols=["cell_id"])
        names |= {norm(v) for v in inst["cell_id"].dropna().unique()}
        print(f"LINCS inst_info: {inst['cell_id'].nunique()} cell ids")
    else:
        print(f"  MISSING {args.inst_info}")

    out = sorted(n for n in names if n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    print(f"wrote {len(out)} public cell-line identifiers to {args.out}")


if __name__ == "__main__":
    main()
