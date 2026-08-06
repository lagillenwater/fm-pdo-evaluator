"""Delta reproducibility ceiling: how noisy is the real Tahoe delta itself?

Check 1 asks how faithfully a source reproduces the real (line, drug) delta and finds r ~ 0.2.
Whether 0.2 is "near the achievable ceiling" or "leaves room for Stack" depends on how reproducible
the real delta is to begin with. This measures that noise floor the same way the label-ceiling
script measured viability reproducibility: split each (line, drug)'s replicate plates into two
halves, aggregate each half's delta, and correlate the two halves over the same top-HVG genes
Check 1 uses. That split-half correlation is the delta's own test-retest reliability -- the most any
predictor, Stack included, could score on Check 1. Spearman-Brown then lifts the half-data number to
the full-data delta the Check-1 sources actually target.

Reuses the DuckDB-over-local-parquet path (the Tahoe DE config is already on scratch from the
pseudobulk shortcut), so it needs no GPU and runs alongside the context build.

  python scripts/delta_reproducibility.py --local-dir /scratch/alpine/$USER/tahoe_pseudobulk_de \\
      --drugs-cid-file data/static/tahoe_target_cids.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TAHOE = "tahoebio/Tahoe-100M"
DE = "pseudobulk_differential_expression"
REPL_CANDIDATES = ("plate", "Plate", "plate_barcode", "plate_id", "replicate", "batch")


def _target_names(repo: Path, cid_file: Path) -> list[str]:
    """Tahoe drug NAMES for the target PubChem CIDs (drug_metadata), matching the shortcut."""
    from datasets import load_dataset  # type: ignore  # Alpine-only

    cids = {t for t in cid_file.read_text().split() if t}
    dm = load_dataset(TAHOE, "drug_metadata", split="train").to_pandas()
    dm = dm[dm["pubchem_cid"].notna()].copy()
    dm["cid"] = dm["pubchem_cid"].map(lambda c: str(int(c)))
    return sorted(dm[dm["cid"].isin(cids)]["drug"].astype(str).unique())


def _split_half_deltas(paths: list[str], target_names: list[str], repl: str | None, tmp: Path):
    """Per (line, drug, gene), the mean log2FoldChange in each of two plate halves, via DuckDB.

    Splits plates by ``hash(repl) % 2`` (deterministic, no RNG) and aggregates each half IN-ENGINE,
    so raw rows are never materialized. Returns the long frame plus the chosen replicate column.
    """
    import duckdb  # type: ignore  # Alpine-only

    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{tmp}'")
    con.execute("SET memory_limit='36GB'")
    con.execute("SET preserve_insertion_order=false")
    schema = con.execute("DESCRIBE SELECT * FROM read_parquet(?) LIMIT 0", [paths]).df()
    cols = list(schema["column_name"])
    print(f"DE columns: {cols}")
    candidates = ([repl] if repl else []) + list(REPL_CANDIDATES)
    chosen = next((c for c in candidates if c in cols), None)
    if chosen is None:
        raise SystemExit(f"no replicate column in {cols}; pass --replicate-col")
    print(f"splitting plates by hash({chosen}) % 2")
    de = con.execute(
        f"""SELECT Cell_ID_DepMap AS patient, drug, gene_name,
                   avg(log2FoldChange) FILTER (WHERE hash({chosen}) % 2 = 0) AS lfc0,
                   avg(log2FoldChange) FILTER (WHERE hash({chosen}) % 2 = 1) AS lfc1
            FROM read_parquet(?)
            WHERE drug IN (SELECT unnest(?)) AND {chosen} IS NOT NULL
            GROUP BY Cell_ID_DepMap, drug, gene_name""",
        [paths, target_names],
    ).df()
    return de, chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local-dir", required=True, help="dir with the Tahoe DE parquet (on scratch)")
    ap.add_argument("--drugs-cid-file", default="data/static/tahoe_target_cids.txt")
    ap.add_argument("--replicate-col", default=None, help="plate/replicate column (auto-detected)")
    ap.add_argument("--n-hvg", type=int, default=2000, help="top HVGs, matching check 1")
    ap.add_argument("--min-genes", type=int, default=50, help="min shared genes to score a pair")
    ap.add_argument("--out", default="results/delta_reproducibility.csv")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent

    local = Path(args.local_dir)
    local = local if local.is_absolute() else repo / local
    paths = sorted(str(p) for p in local.rglob("*.parquet") if DE in str(p))
    if not paths:
        raise SystemExit(f"no {DE} parquet under {local}")
    cid_file = Path(args.drugs_cid_file)
    cid_file = cid_file if cid_file.is_absolute() else repo / cid_file
    names = _target_names(repo, cid_file)
    print(f"{len(names)} target drugs; reading {len(paths)} DE parquet files ...")

    de, repl = _split_half_deltas(paths, names, args.replicate_col, local.parent / "duckdb_tmp")
    de = de.dropna(subset=["lfc0", "lfc1"])
    if de.empty:
        raise SystemExit("no (line, drug, gene) had both plate halves -- too few plates per pair?")
    de["mean"] = (de["lfc0"].to_numpy() + de["lfc1"].to_numpy()) / 2.0

    # HVG = top-variance genes of the mean delta across pairs (matches check 1's panel basis).
    gene_var = de.groupby("gene_name")["mean"].var()
    hvg = set(gene_var.sort_values(ascending=False).index[: args.n_hvg])
    d = de[de["gene_name"].isin(hvg)]

    # per (line, drug): split-half Pearson r between the two plate halves over the HVG genes.
    def _corr(g: pd.DataFrame) -> float:
        if len(g) < args.min_genes:
            return float("nan")
        a, b = g["lfc0"].to_numpy(), g["lfc1"].to_numpy()
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    r = d.groupby(["patient", "drug"]).apply(_corr, include_groups=False).dropna().to_numpy()
    if r.size == 0:
        raise SystemExit("no (line, drug) pair had enough shared HVG genes to score")

    med = float(np.median(r))
    # Spearman-Brown lifts the half-data reliability to the full (all-plate) delta check 1 targets.
    sb = 2 * med / (1 + med) if med > -1 else float("nan")
    summary = {
        "replicate_col": repl,
        "n_pairs": int(r.size),
        "n_genes": len(hvg),
        "splithalf_median_r": round(med, 3),
        "splithalf_mean_r": round(float(np.mean(r)), 3),
        "splithalf_q1_r": round(float(np.quantile(r, 0.25)), 3),
        "splithalf_q3_r": round(float(np.quantile(r, 0.75)), 3),
        "spearman_brown_full": round(sb, 3),
        "frac_pos": round(float(np.mean(r > 0)), 3),
    }
    out = Path(args.out) if Path(args.out).is_absolute() else repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out, index=False)
    print("\n=== delta reproducibility ceiling (real Tahoe delta, plate split-half) ===")
    for k, v in summary.items():
        print(f"  {k:22s} {v}")
    print(
        f"\nCheck-1 achieved r ~ 0.2; the ceiling is the split-half median ({med:.3f}) / "
        f"Spearman-Brown full-data ({sb:.3f}).\nwrote {out}"
    )


if __name__ == "__main__":
    main()
