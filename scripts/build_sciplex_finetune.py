"""Reformat sci-Plex 3 into the single-cell drug-perturbation set stack-finetune consumes.

sci-Plex 3 (Srivatsan et al. 2020, Science; GEO GSE139944) is a single-cell drug screen --
A549/K562/MCF7 x ~188 compounds x doses -- the DRUG-domain analogue of the cytokine data the
Stack generation checkpoint was aligned on, and disjoint from Tahoe so a Tahoe test stays
non-circular. This turns a sci-Plex AnnData into what fine-tuning wants: RAW COUNTS with gene
SYMBOLS in var (stack-finetune aligns to its own panel via --genelist_path), and an obs schema
mirroring the Tahoe context -- pert_id (compound, with a vehicle control), is_control, cell_line,
dose -- so the ICL replacement can pair perturbed cells with their controls.

Input: a sci-Plex AnnData. Two common flavors, both auto-detected (obs/layer names differ):
  * chemCPA: sciplex_complete_middle_subset.h5ad -- obs condition/cell_type/dose + a control flag,
    raw counts usually in a 'counts' layer (chemCPA .X is normalized).
  * scPerturb: SrivatsanTrapnell2020_sciplex3.h5ad -- obs perturbation/cell_line/dose_value.
The script prints the columns/layer it picked so you can confirm; override with the flags if wrong.

  python scripts/build_sciplex_finetune.py --input sciplex_complete_middle_subset.h5ad \\
      --out /scratch/alpine/$USER/sciplex/sciplex_finetune.h5ad --min-cells-per-cond 50
"""

from __future__ import annotations

import argparse

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from fmharness.sciplex_prep import check_gene_count, check_perturbation_count, check_raw_counts

PERT_CANDIDATES = ["perturbation", "condition", "product_name", "treatment", "drug", "compound"]
LINE_CANDIDATES = ["cell_line", "cell line", "cell_type", "cell_name", "line"]
DOSE_CANDIDATES = ["dose_value", "dose", "dose_val", "dose_uM", "dose_unit"]
CTRL_FLAG_CANDIDATES = ["control", "is_control", "vehicle"]
SYM_CANDIDATES = ["gene_symbol", "symbol", "gene_name", "feature_name", "gene_short_name"]
VEHICLE_NAMES = {"control", "vehicle", "dmso", "none", "nan"}


def _pick(cols: object, override: str | None, candidates: list[str], kind: str) -> str | None:
    """Chosen column: the override if given (must exist), else the first candidate present."""
    have = set(cols)  # type: ignore[arg-type]
    if override:
        if override not in have:
            raise SystemExit(f"--{kind} {override!r} not in {sorted(have)}")
        return override
    return next((c for c in candidates if c in have), None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="sci-Plex AnnData (.h5ad); chemCPA/scPerturb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pert-col", default=None, help="obs column with the compound name (auto)")
    ap.add_argument("--control-value", default=None, help="pert-col value marking vehicle cells")
    ap.add_argument("--cell-line-col", default=None)
    ap.add_argument("--dose-col", default=None)
    ap.add_argument("--counts-layer", default=None, help="RAW counts layer (auto: counts else .X)")
    ap.add_argument("--gene-symbol-col", default=None, help="var column with gene symbols (auto)")
    ap.add_argument(
        "--min-cells-per-cond", type=int, default=0, help="drop (cell_line, pert) conds below this"
    )
    args = ap.parse_args()

    a = ad.read_h5ad(args.input)
    check_gene_count(a.n_vars)
    pert_col = _pick(a.obs.columns, args.pert_col, PERT_CANDIDATES, "pert-col")
    line_col = _pick(a.obs.columns, args.cell_line_col, LINE_CANDIDATES, "cell-line-col")
    dose_col = _pick(a.obs.columns, args.dose_col, DOSE_CANDIDATES, "dose-col")
    if pert_col is None or line_col is None:
        raise SystemExit(f"need perturbation + cell-line columns; obs has {list(a.obs.columns)}")

    # ---- RAW counts (Stack is a count model; chemCPA .X is normalized -> prefer a counts layer) --
    if args.counts_layer:
        x, src = a.layers[args.counts_layer], f"layer '{args.counts_layer}'"
    elif "counts" in a.layers:
        x, src = a.layers["counts"], "layer 'counts'"
    else:
        x, src = a.X, ".X (VERIFY these are raw counts, not normalized)"
    x = (x if sparse.issparse(x) else sparse.csr_matrix(np.asarray(x))).tocsr().astype(np.float32)
    check_raw_counts(x, src)

    # ---- gene symbols (uppercased, mirroring Stack's .str.upper() alignment) ----
    sym_col = _pick(a.var.columns, args.gene_symbol_col, SYM_CANDIDATES, "gene-symbol-col")
    syms = a.var[sym_col].astype(str) if sym_col else pd.Series([str(v) for v in a.var_names])
    var = pd.DataFrame(index=pd.Index([s.upper() for s in syms]))
    var["feature_name"] = list(var.index)

    # ---- obs: pert_id, is_control, cell_line, dose ----
    pert = a.obs[pert_col].astype(str).to_numpy()
    check_perturbation_count(pd.Series(pert))
    ctrl_col = next((c for c in CTRL_FLAG_CANDIDATES if c in a.obs.columns), None)
    if args.control_value is not None:
        is_ctl = pert == args.control_value
    elif ctrl_col is not None:
        is_ctl = a.obs[ctrl_col].to_numpy().astype(bool)
    else:
        is_ctl = np.array([p.strip().lower() in VEHICLE_NAMES for p in pert])
    # stack-finetune's drug config identifies controls by a fixed VALUE in the condition column
    # (control_condition), not a flag -- normalize every control cell's pert_id to "control".
    obs = pd.DataFrame(
        {
            "pert_id": np.where(is_ctl, "control", pert),
            "is_control": is_ctl,
            "cell_line": a.obs[line_col].astype(str).to_numpy(),
            "dose": a.obs[dose_col].to_numpy() if dose_col else np.full(a.n_obs, np.nan),
        }
    )
    print(
        f"detected: pert='{pert_col}', cell_line='{line_col}', dose='{dose_col}', "
        f"gene_symbol='{sym_col or 'var_names'}', counts={src}\n"
        f"  perts e.g. {list(pd.unique(pert))[:6]}; controls {int(is_ctl.sum())}/{is_ctl.size}"
    )

    # ---- drop thin conditions (vectorized; no per-condition loop) ----
    if args.min_cells_per_cond > 0:
        cond = obs["cell_line"].str.cat(obs["pert_id"], sep="|")
        keep = (cond.map(cond.value_counts()) >= args.min_cells_per_cond).to_numpy()
        x, obs = x[keep], obs[keep].reset_index(drop=True)

    out = ad.AnnData(X=x, obs=obs, var=var)
    out.obs_names = [str(i) for i in range(out.n_obs)]
    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out)
    print(
        f"wrote {args.out}: {out.n_obs} cells x {out.n_vars} genes, "
        f"{int((~obs['is_control'].to_numpy()).sum())} treated / {int(obs['is_control'].sum())} "
        f"control, {obs['pert_id'].nunique()} perts, {obs['cell_line'].nunique()} cell lines"
    )


if __name__ == "__main__":
    main()
