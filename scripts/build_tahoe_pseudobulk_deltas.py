"""Build the Tahoe per-(line, drug) delta bundle from the pseudobulk DESeq2 table.

The streaming-free shortcut for the baseline-floor scorer. Tahoe ships a
``pseudobulk_differential_expression`` config (~4.1B rows / 89 GB: per cell line x drug x dose
x plate, per gene) carrying ``log2FoldChange`` (treated vs DMSO) and ``baseMean``. Rather than
scan the 95M single cells, this reads ONLY the rows for the target drugs (GDSC2 PubChem CIDs ->
Tahoe drug names via ``drug_metadata``), projected to five columns, via pyarrow predicate
pushdown over the HF parquet -- pulling a slice, not the whole table. It reads ONE drug at a
time and reshapes immediately (peak memory = one drug, not all of them), then aggregates to the
``(delta, key, baseline)`` contract the scorer reads and writes a small parquet bundle.

Run on Alpine as a SINGLE process (needs datasets + pyarrow + huggingface_hub; the compute node
has internet). The table is a flat 1026-file shard set with no drug partition, so every query
scans all footers -- running many of these concurrently trips HF's 429 rate limit; keep it one
process (auth with HF_TOKEN to raise the ceiling):
  python scripts/build_tahoe_pseudobulk_deltas.py \\
      --drugs-cid-file data/static/tahoe_target_cids.txt --out-dir tahoe_deltas/
then score (no single-cell context needed):
  PYTHONPATH=src python scripts/score_generation_eval.py --deltas-bundle tahoe_deltas/ --k 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fmharness.deltas import pseudobulk_de_to_deltas

TAHOE = "tahoebio/Tahoe-100M"
DE = "pseudobulk_differential_expression"
DE_COLS = ["gene_name", "log2FoldChange", "baseMean", "Cell_ID_DepMap", "drug"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--drugs-cid-file", required=True, help="PubChem CIDs to keep (the GDSC2 answer key)"
    )
    ap.add_argument("--out-dir", default="tahoe_deltas")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent

    import pyarrow.dataset as pads  # type: ignore  # Alpine-only
    from datasets import load_dataset  # type: ignore
    from huggingface_hub import HfFileSystem  # type: ignore

    cid_path = Path(args.drugs_cid_file)
    cid_path = cid_path if cid_path.is_absolute() else repo / cid_path
    target_cids = {t for t in cid_path.read_text().split() if t}

    # Tahoe keys the pseudobulk table by drug NAME; map name <-> PubChem CID, keep target CIDs.
    dm = load_dataset(TAHOE, "drug_metadata", split="train").to_pandas()
    dm = dm[dm["pubchem_cid"].notna()].copy()
    dm["cid"] = dm["pubchem_cid"].map(lambda c: str(int(c)))
    dm = dm[dm["cid"].isin(target_cids)]
    name_to_cid = dict(zip(dm["drug"].astype(str), dm["cid"].astype(str), strict=False))
    target_names = sorted(name_to_cid)
    print(f"{len(target_names)} of Tahoe's 379 drugs map to a GDSC2 AUC CID")
    if not target_names:
        raise SystemExit("no Tahoe drug maps to a target CID -- check the CID file")

    # locate the config's parquet on HF, then read only the target-drug rows + 5 columns.
    fs = HfFileSystem()
    paths = [p for p in fs.glob(f"datasets/{TAHOE}/**/*.parquet") if DE in p]
    if not paths:
        raise SystemExit(f"could not locate {DE} parquet files under datasets/{TAHOE}")
    print(f"reading {len(paths)} pseudobulk parquet files (drug-filtered, {len(DE_COLS)} cols) ...")
    dset = pads.dataset([f"hf://{p}" for p in paths], filesystem=fs, format="parquet")

    # Pull and reshape ONE drug at a time: a single drug is ~1e7 rows, whereas all target
    # drugs at once is ~1e8 -- with object-dtype string columns that OOMs a 64 GB node.
    # Reshaping each drug to its compact (delta, key, base) and dropping the raw rows caps
    # peak memory at one drug, and the serial requests ease HF's rate limiting (the flat
    # 1026-file layout means every query still scans all footers, so DO NOT parallelize this).
    delta_parts: list[pd.DataFrame] = []
    key_parts: list[pd.DataFrame] = []
    base_parts: list[pd.DataFrame] = []
    for i, name in enumerate(target_names, start=1):
        tbl = dset.to_table(columns=DE_COLS, filter=pads.field("drug") == name)
        de = tbl.to_pandas(self_destruct=True)
        del tbl
        if de.empty:
            print(f"  [{i}/{len(target_names)}] {name}: no rows")
            continue
        d, k, b = pseudobulk_de_to_deltas(de, name_to_cid)
        delta_parts.append(d)
        key_parts.append(k)
        base_parts.append(b)
        print(f"  [{i}/{len(target_names)}] {name}: {len(k)} lines, {len(de):,} rows")
        del de
    if not key_parts:
        raise SystemExit("no target drug returned any DE rows")

    # concat the per-drug pieces: real_delta/real_key are row-disjoint (union genes, NaN->0);
    # base is per-line baseMean averaged across drugs (a k-NN neighbor proxy only).
    real_delta = pd.concat(delta_parts, ignore_index=True).fillna(0.0)
    real_key = pd.concat(key_parts, ignore_index=True)
    base = pd.concat(base_parts).groupby(level=0).mean().fillna(0.0)
    base.index = base.index.astype(str)
    out = Path(args.out_dir) if Path(args.out_dir).is_absolute() else repo / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    real_delta.to_parquet(out / "real_delta.parquet", index=False)
    real_key.to_parquet(out / "real_key.parquet", index=False)
    base.to_parquet(out / "base.parquet")  # keeps the DepMap-line index
    print(
        f"wrote {out}: {len(real_key)} (line, drug) pairs over {base.shape[0]} lines, "
        f"{real_delta.shape[1]} genes"
    )


if __name__ == "__main__":
    main()
