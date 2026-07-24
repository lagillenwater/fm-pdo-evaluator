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
touches all files -- concurrent queries trip HF's 429 rate limit, so keep it one process, and
authenticate first (``hf auth login``) to raise the ceiling. ``--local-dir`` is the robust path:
one authenticated bulk download to scratch, then local bounded reads (no per-query network):
  hf auth login
  python scripts/build_tahoe_pseudobulk_deltas.py \\
      --drugs-cid-file data/static/tahoe_target_cids.txt \\
      --local-dir /scratch/alpine/$USER/tahoe_pseudobulk_de --out-dir tahoe_deltas/
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
    ap.add_argument(
        "--local-dir",
        default=None,
        help="query parquet from this local dir instead of hf:// (downloads the DE config there "
        "once if empty). The robust path: one authenticated bulk pull, then local bounded reads.",
    )
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

    # locate the config's parquet, locally (robust) or on HF (hf://).
    if args.local_dir:
        local = Path(args.local_dir)
        local = local if local.is_absolute() else repo / local
        # DE is the parent DIRECTORY, not part of the filename (train-NNNNN-of-01026.parquet),
        # so match on the full path, not the filename glob.
        if not any(DE in str(p) for p in local.rglob("*.parquet")):
            print(f"downloading the {DE} config to {local} (one-time, authenticated) ...")
            from huggingface_hub import snapshot_download  # type: ignore

            snapshot_download(
                TAHOE, repo_type="dataset", allow_patterns=[f"*{DE}*"], local_dir=str(local)
            )
        paths = sorted(str(p) for p in local.rglob("*.parquet") if DE in str(p))
        if not paths:
            raise SystemExit(f"no {DE} parquet under {local}")
        print(f"reading {len(paths)} LOCAL pseudobulk parquet files ({len(DE_COLS)} cols) ...")
        dset = pads.dataset(paths, format="parquet")
    else:
        fs = HfFileSystem()
        paths = [p for p in fs.glob(f"datasets/{TAHOE}/**/*.parquet") if DE in p]
        if not paths:
            raise SystemExit(f"could not locate {DE} parquet files under datasets/{TAHOE}")
        print(f"reading {len(paths)} REMOTE pseudobulk parquet (hf://, {len(DE_COLS)} cols) ...")
        dset = pads.dataset([f"hf://{p}" for p in paths], filesystem=fs, format="parquet")

    # Read ONE drug at a time through a BOUNDED scanner. The earlier OOM was pyarrow's default
    # scanner over-buffering the scan across the 1026 files (aggressive readahead + pre_buffer),
    # not the per-drug result (~1e7 rows). fragment_readahead=1 / batch_readahead=1 / no threads /
    # no pre_buffer cap the in-flight scan to a couple of small batches, so peak memory is one
    # drug's reshaped result, not the scan. The flat no-drug-partition layout means every query
    # still touches all files, so keep this a single process (DO NOT parallelize the scan).
    scan_opts = {
        "columns": DE_COLS,
        "batch_size": 64_000,
        "batch_readahead": 1,
        "fragment_readahead": 1,
        "use_threads": False,
        "fragment_scan_options": pads.ParquetFragmentScanOptions(pre_buffer=False),
    }
    delta_parts: list[pd.DataFrame] = []
    key_parts: list[pd.DataFrame] = []
    base_parts: list[pd.DataFrame] = []
    for i, name in enumerate(target_names, start=1):
        de = dset.scanner(filter=pads.field("drug") == name, **scan_opts).to_table().to_pandas(
            self_destruct=True
        )
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
