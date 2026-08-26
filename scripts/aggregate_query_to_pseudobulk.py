"""Collapse a Stack query or context to one pseudobulk profile per line, matching the organoid setup.

Stack is the only model in this harness that consumes CELLS. Every cell-line rung hands it
single cells -- ``tahoe_query.h5ad`` carries 8 control cells per line -- while the organoid rung
hands it one bulk profile per organoid, and the L1000 context is bulk wells "treated as
pseudo-cells". So two granularity shifts sit between the cell-line path and the organoid path,
both currently confounded with the modality change. If rung 4 underperforms rung 3 we cannot
say whether organoids are biologically different or whether Stack simply got one bulk profile
instead of eight cells.

This produces the control that separates them: the same lines, the same drugs, the same
checkpoint, with the input collapsed to one profile per group. Comparing a rung run on the
collapsed input against the same rung on cells isolates the granularity cost, measured where a
ceiling exists.

The baselines need no such control -- pca/nmf/knn already consume one pseudobulk profile per
(line, drug) on both sides, so their granularity is already matched. This is Stack-specific.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def main() -> None:
    """Average cells within each group to a single profile, preserving obs schema."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--group-col",
        default=None,
        help="obs column to collapse within (e.g. cell_id). Defaults to the obs index, which "
        "for a query built 8-cells-per-line means collapsing to one profile per line.",
    )
    ap.add_argument(
        "--strip-suffix",
        action="store_true",
        help="treat obs names as <line>-<replicate> and group on the prefix. Query files number "
        "their cells per line, so without this every cell is its own group and the collapse is "
        "a silent no-op that still writes a plausible file.",
    )
    args = ap.parse_args()

    adata = ad.read_h5ad(args.input)
    obs = adata.obs.copy()
    if args.group_col and args.group_col in obs.columns:
        groups = obs[args.group_col].astype(str).to_numpy()
    elif args.strip_suffix:
        groups = np.array([str(n).rsplit("-", 1)[0] for n in adata.obs_names])
    else:
        groups = np.asarray([str(n) for n in adata.obs_names])

    uniq = sorted(set(groups))
    if len(uniq) == adata.n_obs:
        raise SystemExit(
            f"grouping yields {len(uniq)} groups for {adata.n_obs} cells -- nothing would be "
            "collapsed. Pass --group-col or --strip-suffix; a no-op that writes a plausible "
            "file is the failure mode this check exists to prevent."
        )

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = np.asarray(X, dtype=np.float64)
    rows = np.vstack([X[groups == g].mean(axis=0) for g in uniq])
    new_obs = pd.DataFrame(index=pd.Index(uniq, name=adata.obs.index.name))
    for c in obs.columns:
        # keep a column only where it is constant within every group, so a per-cell attribute
        # is dropped rather than silently represented by its first value
        vals = {}
        ok = True
        for g in uniq:
            v = obs.loc[groups == g, c].astype(str).unique()
            if len(v) != 1:
                ok = False
                break
            vals[g] = v[0]
        if ok:
            new_obs[c] = [vals[g] for g in uniq]

    out = ad.AnnData(X=rows.astype(np.float32), obs=new_obs, var=adata.var.copy())
    out.var["feature_name"] = [str(v) for v in out.var_names]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out)
    print(f"{adata.n_obs} cells -> {out.n_obs} pseudobulk profiles x {out.n_vars} genes")
    print(f"  obs columns kept (constant within group): {list(new_obs.columns)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
