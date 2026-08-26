"""Split a Stack perturbation context into the per-drug shards 04_stack_generate expects.

build_l1000_context writes one AnnData; the generation array reads one shard per task from
``<dir>/%04d.h5ad`` with ``manifest.tsv`` mapping array index -> pert_id. This produces that
layout for any context, so an L1000-derived context can drive the same generation array as the
Tahoe one and the two runs differ only in the corpus Stack reads a drug's effect from.

Mirrors 03_stack_context's sharding exactly, including the two details that break generation
silently when missed: ``ad.concat`` drops var columns, so ``feature_name`` is re-attached or
stack-generation's ``--gene-name-col`` falls back to the index and aligns on the wrong
namespace; and obs names must be made unique or the concat produces duplicate cells.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad


def main() -> None:
    """Write one shard per drug, each carrying that drug's wells plus all controls."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--pert-col", default="pert_id")
    ap.add_argument("--control-col", default="is_control")
    args = ap.parse_args()

    adata = ad.read_h5ad(args.context)
    obs = adata.obs
    if args.control_col not in obs.columns:
        raise SystemExit(f"{args.control_col!r} not in obs: {list(obs.columns)}")
    is_ctl = obs[args.control_col].to_numpy().astype(bool)
    ctl = adata[is_ctl]
    perts = sorted(set(obs.loc[~is_ctl, args.pert_col].astype(str)))
    if not perts:
        raise SystemExit("no treated wells in this context -- refusing to write empty shards")
    print(f"{adata.n_obs} cells, {len(perts)} drugs, {int(is_ctl.sum())} controls")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, p in enumerate(perts):
        trt = adata[(obs[args.pert_col].astype(str) == p).to_numpy()]
        shard = ad.concat([trt, ctl])
        shard.var["feature_name"] = [str(v) for v in shard.var_names]
        shard.obs_names_make_unique()
        shard.write_h5ad(args.out_dir / f"{i:04d}.h5ad")
        lines.append(f"{i}\t{p}\t{i:04d}.h5ad")
        print(f"  {i:04d} {p}: {trt.n_obs} treated + {ctl.n_obs} control")
    (args.out_dir / "manifest.tsv").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(perts)} shards to {args.out_dir}; array is 0-{len(perts) - 1}")


if __name__ == "__main__":
    main()
