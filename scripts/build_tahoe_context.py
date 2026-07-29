"""Build the Tahoe-100M single-cell perturbation context for Stack-Large-Aligned generation.

Tahoe is the in-domain single-cell drug context (replacing bulk L1000): given a perturbation
context of drug-treated cells and a query baseline, Stack generates the query's treated state.
This reads only a *subset* -- the target drugs plus their DMSO_TF vehicle controls, in the target
cell lines -- directly from the HuggingFace parquet via a pushed-down ``drug`` filter (pyarrow
prunes non-matching row-groups, so target drugs are found wherever their plates sit and the
~95% of non-target cells are never decoded), reconstructs expression over the Stack gene panel
from the tokenized (``genes`` token-id + ``expressions`` value) format, maps the Cellosaurus
``cell_line_id`` to its DepMap id, and writes a context AnnData whose obs schema matches
``build_l1000_context`` (pert_id / pert_iname / cell_id / is_control) so the stack-generation
call and the delta builders consume it unchanged. Treated and DMSO cells are tagged, so the
per-line baseline (is_control) and the real treated state (the truth for generation-quality)
are both slices of this one file -- no separate query/baseline build needed for cell lines.

Run on Alpine (needs ``datasets`` + ``pyarrow``; reads from HF so no full ~100M-cell download,
and only target + DMSO cells are decoded). ``--max-cells-per-cond`` caps cells per (line, drug)
to bound memory:
  python scripts/build_tahoe_context.py --drugs-cid-file data/static/gdsc2_auc_pubchem_cids.txt \\
      --max-cells-per-cond 200 --out tahoe_context.h5ad
then generate (same call shape as the L1000 path, with the Tahoe context as base-adata):
  stack-generation --checkpoint stack-aligned/bc_large_aligned.ckpt \\
      --base-adata tahoe_context.h5ad --test-adata stack_input_sarcoma.h5ad \\
      --genelist stack-aligned/basecount_1000per_15000max.pkl --gene-name-col feature_name \\
      --output-dir generated/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from fmharness.tahoe import parse_dose_um, scatter_tokens

TAHOE = "tahoebio/Tahoe-100M"
DMSO = "DMSO_TF"  # Tahoe's vehicle-control drug name


def _ncid(x: object) -> str:
    """Normalize a PubChem CID to a plain int-string. Tahoe stores it float-formatted
    ('1923.0'), so a raw ``str()`` never matches the target int-strings ('1923')."""
    try:
        return str(int(float(x)))  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--drugs-cid", nargs="*", default=None, help="PubChem CIDs to keep (default all)"
    )
    ap.add_argument(
        "--drugs-cid-file",
        default=None,
        help="file of PubChem CIDs (whitespace/newline-separated); merged with --drugs-cid",
    )
    ap.add_argument(
        "--cell-lines", nargs="*", default=None, help="Cellosaurus cell_line_ids (default all)"
    )
    ap.add_argument("--dose-um", type=float, default=None, help="keep only this drug dose in uM")
    ap.add_argument(
        "--max-cells-per-cond",
        type=int,
        default=None,
        help="subsample cap on cells kept per (cell line, drug); bounds memory (default no cap)",
    )
    ap.add_argument(
        "--max-scan-cells",
        type=int,
        default=None,
        help="stop after reading this many MATCHING (target + DMSO) cells; default reads all. "
        "The drug filter is pushed into the parquet read, so only target drugs' cells are "
        "decoded (not the full ~100M stream) -- this cap rarely matters",
    )
    ap.add_argument("--out", default="tahoe_context.h5ad")
    ap.add_argument(
        "--batch", type=int, default=50000, help="cells scattered per chunk (bounds memory)"
    )
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent

    from datasets import load_dataset  # type: ignore  # Alpine-only, heavy import

    # small metadata tables load fully; the cell matrix streams.
    gm = load_dataset(TAHOE, "gene_metadata", split="train").to_pandas()
    clm = load_dataset(TAHOE, "cell_line_metadata", split="train").to_pandas()
    sm = load_dataset(TAHOE, "sample_metadata", split="train").to_pandas()

    # Stack's native 15,012-gene vocabulary (its genelist). The context uses the full panel;
    # Stack zero-pads any of these a query (e.g. the Soragni ~12.8k subset) does not measure.
    # Match by uppercased gene symbol, mirroring Stack's own .str.upper() gene alignment.
    hvg = pd.read_csv(repo / "data/static/stack_hvg_genes.txt", header=None)[0].astype(str)
    panel = {s.upper() for s in hvg}
    sym_u = gm["gene_symbol"].astype(str).str.upper()
    pan = gm[sym_u.isin(panel)].drop_duplicates("token_id")
    panel_syms = [s.upper() for s in pan["gene_symbol"].astype(str)]
    token_to_col = {int(t): i for i, t in enumerate(pan["token_id"])}
    print(
        f"panel: {len(panel_syms)} of Stack's 15,012-gene vocabulary covered by Tahoe genes",
        flush=True,
    )

    # The stream's cell_line_id is a Cellosaurus accession (CVCL_...); map it to DepMap via the
    # metadata's Cellosaurus + DepMap columns (Check 2 and the GDSC2 join key on DepMap).
    dep_col = next((c for c in clm.columns if "depmap" in c.lower()), None)
    cvcl_col = next((c for c in clm.columns if "cellosaur" in c.lower()), None)
    cl2dep: dict[str, str] = {}
    if dep_col and cvcl_col:
        cl2dep = dict(zip(clm[cvcl_col].astype(str), clm[dep_col].astype(str), strict=False))
    sample_dose = {
        str(s): parse_dose_um(str(c))
        for s, c in zip(sm["sample"], sm["drugname_drugconc"], strict=False)
    }

    cids = set(map(str, args.drugs_cid)) if args.drugs_cid else set()
    if args.drugs_cid_file:
        cids |= {tok for tok in Path(args.drugs_cid_file).read_text().split() if tok}
    keep_cids: set[str] | None = {n for c in cids if (n := _ncid(c))} or None
    lines = set(args.cell_lines) if args.cell_lines else None
    cap = args.max_cells_per_cond
    max_scan = args.max_scan_cells
    cond_count: dict[tuple[str, str], int] = {}
    # ---- push the drug filter into the parquet read (no full-stream decode) ----------------
    # Tahoe's expression_data shards are ordered by plate, and each drug lives on specific
    # plates, so a streamed scan-cap silently drops most target drugs. Instead read ONLY the
    # target-drug + DMSO cells: pyarrow prunes whole row-groups by the `drug` column, so this
    # is complete regardless of shard order and never decodes the ~95% of cells that a stream
    # would read then discard. The kept set is identical to a full-scan streaming run.
    import os

    import pyarrow.dataset as pads
    from huggingface_hub import HfFileSystem  # type: ignore  # Alpine-only, heavy import
    from pyarrow.fs import FSSpecHandler, PyFileSystem

    # Map target CIDs -> Tahoe drug names so the pushed filter is a clean string `isin` on the
    # `drug` column (row-group-prunable), not a match on the float-formatted pubchem_cid.
    target_drugs: set[str] | None = None
    if keep_cids is not None:
        dm = load_dataset(TAHOE, "drug_metadata", split="train").to_pandas()
        dm_cid = next(c for c in dm.columns if "pubchem" in c.lower())
        dm_drug = "drug" if "drug" in dm.columns else next(
            c for c in dm.columns if "drug" in c.lower()
        )
        target_drugs = set(dm.loc[dm[dm_cid].map(_ncid).isin(keep_cids), dm_drug].astype(str))
        print(f"target drugs: {len(target_drugs)} Tahoe names <- {len(keep_cids)} CIDs", flush=True)
        if not target_drugs:
            raise SystemExit("no Tahoe drug names matched the requested CIDs")

    fs = HfFileSystem(token=os.environ.get("HF_TOKEN") or None)
    # 'expression_data' is the config DIRECTORY; the pseudobulk config ('..._expression') does
    # not contain the substring 'expression_data', so this selects the cell matrix cleanly.
    files = [f for f in fs.glob(f"datasets/{TAHOE}/**/*.parquet") if "expression_data" in f]
    if not files:
        raise SystemExit("no expression_data parquet shards resolved on HF")
    print(f"expression_data: {len(files)} shards; first = {files[0]}", flush=True)

    pafs = PyFileSystem(FSSpecHandler(fs))
    dataset = pads.dataset(files, filesystem=pafs, format="parquet")
    read_cols = ["drug", "pubchem_cid", "cell_line_id", "plate", "sample", "genes", "expressions"]
    filt = pads.field("drug").isin(sorted(target_drugs | {DMSO})) if target_drugs else None
    scanner = dataset.scanner(columns=read_cols, filter=filt, batch_size=args.batch)

    g_acc: list[np.ndarray] = []
    e_acc: list[np.ndarray] = []
    obs_rows: list[tuple[object, ...]] = []
    mats: list[sparse.csr_matrix] = []

    def flush() -> None:
        if g_acc:
            mats.append(scatter_tokens(g_acc, e_acc, token_to_col, len(panel_syms)))
            g_acc.clear()
            e_acc.clear()

    scanned = 0
    stop = False
    for batch in scanner.to_batches():
        # small columns to Python once per batch; the big genes/expressions arrays are
        # materialized per row only when a cell is actually kept (after the per-cond cap).
        drug_c = batch.column("drug").to_pylist()
        cid_c = batch.column("pubchem_cid").to_pylist()
        cl_c = batch.column("cell_line_id").to_pylist()
        plate_c = batch.column("plate").to_pylist()
        samp_c = batch.column("sample").to_pylist()
        genes_c = batch.column("genes")
        expr_c = batch.column("expressions")
        for i in range(batch.num_rows):
            scanned += 1
            if max_scan is not None and scanned > max_scan:
                stop = True
                break
            if scanned % 1_000_000 == 0:  # heartbeat: matching cells read + cells kept
                print(f"  read {scanned:,} matching cells, kept {len(obs_rows):,}", flush=True)
            drug = drug_c[i]
            is_ctl = drug == DMSO
            cl = cl_c[i]
            if lines is not None and cl not in lines:
                continue
            dose = sample_dose.get(str(samp_c[i]), float("nan"))
            if args.dose_um is not None and not is_ctl and not np.isclose(dose, args.dose_um):
                continue
            if cap is not None:
                ckey = (str(cl), str(drug))
                if cond_count.get(ckey, 0) >= cap:
                    continue
                cond_count[ckey] = cond_count.get(ckey, 0) + 1
            g_acc.append(genes_c[i].values.to_numpy(zero_copy_only=False))
            e_acc.append(expr_c[i].values.to_numpy(zero_copy_only=False))
            obs_rows.append(
                (drug, _ncid(cid_c[i]), cl, cl2dep.get(str(cl), ""), bool(is_ctl),
                 plate_c[i], samp_c[i], dose)
            )
            if len(g_acc) >= args.batch:
                flush()
        if stop:
            break
    flush()

    n_cols = len(panel_syms)
    X = sparse.vstack(mats).tocsr() if mats else sparse.csr_matrix((0, n_cols), dtype=np.float32)
    obs = pd.DataFrame(
        obs_rows,
        columns=pd.Index(
            [
                "pert_iname",
                "pubchem_cid",
                "cell_line_id",
                "cell_id",
                "is_control",
                "plate",
                "sample",
                "dose_um",
            ]
        ),
    )
    obs["pert_id"] = obs["pert_iname"]  # mirror build_l1000_context (pert_id keys generated files)
    adata = ad.AnnData(X=X, obs=obs)
    adata.obs_names = [str(i) for i in range(adata.n_obs)]
    adata.var_names = panel_syms
    adata.var["feature_name"] = panel_syms
    out = repo / args.out if not Path(args.out).is_absolute() else Path(args.out)
    adata.write_h5ad(out)
    n_trt = int((~obs["is_control"].astype(bool)).sum())
    print(
        f"wrote {out}  (scanned {scanned:,} cells -> {adata.n_obs} kept x {adata.n_vars} genes, "
        f"{n_trt} treated, {int(obs['is_control'].sum())} DMSO, "
        f"{obs['cell_id'].ne('').sum()} with a DepMap id)"
    )


if __name__ == "__main__":
    main()
