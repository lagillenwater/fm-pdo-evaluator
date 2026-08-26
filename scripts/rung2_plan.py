"""Rung 2, stage 1: build the L1000 and Tahoe sides once, on a panel they can share.

Rung 2 asks whether a baseline->delta map fit on ONE platform still works on another. The
number that carries the rung is not the cross-platform score itself but the difference from the
same model fit in-platform -- the transfer penalty. That difference is only meaningful if both
arms are scored on the same genes, and they cannot be taken from separate jobs: rung 1 runs on
tahoe n stack n sciplex = 14,121, while any panel including L1000 is bounded by
l1000 n stack = 8,865. Subtracting a score on 14,121 genes from one on 8,865 would repeat
exactly the error this project has been unpicking all week.

So this stage pins BOTH arms' inputs on the L1000-inclusive panel, and the scatter stage
computes both from the same pinned data. Rung 1's headline number stays on its own wider panel;
the in-platform arm recomputed here exists only to be subtracted.

Writes, under --out-dir:
  plan.json          resolved parameters, git sha, panel size, the (source, arm) grid
  tahoe_*.parquet    target-side delta/key/base, panel-restricted
  l1000_*.parquet    train-side delta/key/base, panel-restricted
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from fmharness.deltas import common_gene_panel, load_panel_constraint

# Controls are part of the grid, not an afterthought. `prior` is the line-independent floor:
# it predicts each drug's mean and nothing line-specific, so any source failing to beat it has
# learned nothing about lines. `shuffled` is the negative control: the same fitted map applied
# to a line-permuted baseline, which must collapse to the null. Without a floor and a noise row
# a transfer penalty is uninterpretable -- a source could "transfer well" simply by predicting
# the drug mean on both platforms, which is exactly what `additive`/`measured_delta` does.
# `planted` is the positive control: a delta built to be perfectly predictable from the target
# baseline, which the same fit must recover. Without something that MUST succeed, a grid of
# small numbers cannot distinguish "transfer is hard" from "this pipeline cannot fit anything",
# and rung 2 is exactly where that ambiguity bites -- every arm is expected to score low.
SOURCES = ("prior", "knn", "pca", "nmf", "observed_delta", "shuffled", "planted")
# bulk_target is the BASELINES' granularity control, and it exists because "both sides are one
# profile per line" describes shape, not distribution. A Tahoe profile is a pseudobulk average
# over thousands of single cells, carrying 10x dropout structure and a homogeneous cell line; an
# organoid profile is a bulk library of heterogeneous tissue. The maps are fit on the first and
# applied to the second, so the baselines face a representation shift too -- a different one from
# Stack's, which is about WHERE aggregation happens rather than what the profile is.
#
# 44 cell lines carry both a Tahoe pseudobulk baseline and GDSC2 bulk RNA-seq [job 31659975], so
# the same biological samples exist under both constructions. This arm fits in-platform on Tahoe
# and predicts from the GDSC2 BULK profile of the same line, holding platform, drug and line
# fixed so only the profile's construction varies.
ARMS = ("in_platform", "cross_platform", "bulk_target")


def norm_line(s: object) -> str:
    """Uppercase alphanumeric, matching fmharness.deltas._norm."""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def norm_name(s: object) -> str:
    """Lowercase alphanumeric, for joining drug names."""
    return "".join(c for c in str(s).lower() if c.isalnum())


def git_sha() -> str:
    """The commit these inputs were built at."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    """Build both platforms' deltas on one shared panel and pin them."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deltas-bundle", default="tahoe_deltas")
    ap.add_argument("--l1000-dir", type=Path, default=Path("."))
    ap.add_argument("--gctx", required=True)
    ap.add_argument("--pert-map", type=Path, default=Path("context_by_drug/pert_to_cid.tsv"))
    ap.add_argument("--model-csv", type=Path, default=Path("data/raw/gdsc2_sarcoma/depmap/Model.csv"))
    ap.add_argument("--panel-source", action="append", default=None, help="label=path, repeatable")
    ap.add_argument(
        "--bulk-base",
        type=Path,
        default=Path("data/reference/stack_input_gdscv2.h5ad"),
        help="bulk RNA-seq baseline for the SAME lines, for the bulk_target granularity arm",
    )
    ap.add_argument("--time", type=float, default=24.0)
    ap.add_argument("--treated-cap", type=int, default=8)
    ap.add_argument("--dmso-cap", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    from cmapPy.pandasGEXpress.parse_gctx import parse

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bdir = Path(args.deltas_bundle)
    t_delta = pd.read_parquet(bdir / "real_delta.parquet")
    t_key = pd.read_parquet(bdir / "real_key.parquet")
    t_base = pd.read_parquet(bdir / "base.parquet")

    pm = pd.read_csv(args.pert_map, sep="\t", header=None, names=["drug_name", "pubchem_cid"])
    cid2name = {str(c).split(".")[0]: str(n) for n, c in zip(pm["drug_name"], pm["pubchem_cid"], strict=True)}
    model = pd.read_csv(args.model_csv, low_memory=False)
    ach2name = {
        norm_line(m): norm_line(n)
        for m, n in zip(model["ModelID"], model["StrippedCellLineName"], strict=True)
        if isinstance(n, str) and n
    }
    t_key = t_key.assign(
        line=[ach2name.get(norm_line(p), norm_line(p)) for p in t_key["patient"]],
        dname=[norm_name(cid2name.get(str(d).split(".")[0], "")) for d in t_key["drug"]],
    )
    t_base.index = pd.Index([ach2name.get(norm_line(i), norm_line(i)) for i in t_base.index])

    with gzip.open(args.l1000_dir / "GSE92742_Broad_LINCS_gene_info.txt.gz", "rt") as fh:
        gi = pd.read_csv(fh, sep="\t")
    sym = gi.set_index("pr_gene_id")["pr_gene_symbol"].astype(str)

    with gzip.open(args.l1000_dir / "GSE92742_Broad_LINCS_inst_info.txt.gz", "rt") as fh:
        inst = pd.read_csv(fh, sep="\t", low_memory=False)
    inst["line"] = [norm_line(c) for c in inst["cell_id"]]
    inst["dname"] = [norm_name(n) for n in inst["pert_iname"]]

    # Train on EVERY L1000 line carrying the target drugs, not only the 7 that overlap Tahoe.
    # The map is baseline -> delta residual, so more training lines is strictly better, and
    # restricting to the overlap would confound the transfer question with a sample-size drop.
    drugs = sorted({d for d in t_key["dname"] if d} & set(inst["dname"]))
    tw = inst[inst["dname"].isin(drugs) & (inst["pert_time"] == args.time)]
    tw = tw.sort_values("inst_id").groupby(["line", "dname"], sort=False).head(args.treated_cap)
    lines_with_treated = set(tw["line"])
    cw = inst[
        (inst["pert_iname"] == "DMSO")
        & inst["line"].isin(lines_with_treated)
        & (inst["pert_time"] == args.time)
    ].sort_values("inst_id").groupby("line", sort=False).head(args.dmso_cap)
    tw = tw[tw["line"].isin(set(cw["line"]))]
    print(f"L1000 train: {len(drugs)} drugs, {tw['line'].nunique()} lines, "
          f"{len(tw)} treated + {len(cw)} DMSO wells")

    def group_means(ids: list[str], lab: dict[str, str]) -> pd.DataFrame:
        """Mean profile per label, reading the .gctx in column chunks."""
        tot: pd.DataFrame | None = None
        cnt: pd.Series | None = None
        for i in range(0, len(ids), args.chunk):
            block = parse(args.gctx, cid=ids[i : i + args.chunk]).data_df.T
            block.index = block.index.map(lab)
            s, n = block.groupby(level=0).sum(), block.groupby(level=0).size()
            tot = s if tot is None else tot.add(s, fill_value=0.0)
            cnt = n if cnt is None else cnt.add(n, fill_value=0)
        assert tot is not None and cnt is not None
        return tot.div(cnt, axis=0)

    tmean = group_means(
        tw["inst_id"].tolist(), dict(zip(tw["inst_id"], tw["line"] + "\t" + tw["dname"], strict=True))
    )
    dmean = group_means(cw["inst_id"].tolist(), dict(zip(cw["inst_id"], cw["line"], strict=True)))
    for m in (tmean, dmean):
        m.columns = pd.Index([str(sym.get(int(c), "")) for c in m.columns])
    tmean = tmean.loc[:, (tmean.columns != "") & ~tmean.columns.duplicated()]
    dmean = dmean.loc[:, (dmean.columns != "") & ~dmean.columns.duplicated()]

    parts = pd.Series(tmean.index).str.split("\t", expand=True)
    l_lines, l_drugs = parts[0].to_numpy(), parts[1].to_numpy()
    keep = pd.Series(l_lines).isin(dmean.index).to_numpy()
    tmean, l_lines, l_drugs = tmean[keep], l_lines[keep], l_drugs[keep]
    l_delta = pd.DataFrame(
        tmean.to_numpy() - dmean.reindex(index=l_lines, columns=tmean.columns).to_numpy(),
        columns=tmean.columns,
    )
    l_key = pd.DataFrame({"patient": l_lines, "drug": l_drugs})

    # The panel must include L1000, which is what makes it narrower than rung 1's.
    constraints = {"l1000": l_delta}
    for spec in args.panel_source or []:
        label, _, path = spec.partition("=")
        pp = Path(path)
        if not pp.exists():
            raise SystemExit(f"--panel-source {label}={pp} not present; refusing to widen the panel")
        constraints[label] = pd.DataFrame(columns=load_panel_constraint(pp))
        print(f"  panel constraint {label}: {constraints[label].shape[1]} genes")
    panel = common_gene_panel(t_delta, constraints)
    panel = pd.Index([g for g in panel if g in dmean.columns and g in t_base.columns])
    print(f"rung-2 panel: {len(panel)} genes (rung 1 runs on a wider one; see the docstring)")
    if len(panel) < 500:
        raise SystemExit(f"panel collapsed to {len(panel)} genes -- refusing to score on it")

    # Pin the bulk baseline for the lines Tahoe and GDSC2 share, on the same panel.
    if args.bulk_base.exists():
        import h5py

        with h5py.File(args.bulk_base, "r") as f:
            var, obs = f["var"], f["obs"]
            vi = var.attrs.get("_index", "_index"); vi = vi.decode() if isinstance(vi, bytes) else str(vi)
            oi = obs.attrs.get("_index", "_index"); oi = oi.decode() if isinstance(oi, bytes) else str(oi)
            genes_b = [x.decode() if isinstance(x, bytes) else str(x) for x in var[vi][:]]
            rows_b = [x.decode() if isinstance(x, bytes) else str(x) for x in obs[oi][:]]
            Xb = f["X"][:] if isinstance(f["X"], h5py.Dataset) else None
        if Xb is not None:
            bulk = pd.DataFrame(Xb, index=pd.Index([ach2name.get(norm_line(r), norm_line(r)) for r in rows_b]),
                                columns=pd.Index(genes_b))
            bulk = bulk.loc[~bulk.index.duplicated()]
            shared_b = [ln for ln in t_base.index if ln in bulk.index]
            print(f"  bulk_target arm: {len(shared_b)} lines with BOTH Tahoe pseudobulk and GDSC2 bulk")
            bulk.reindex(index=shared_b, columns=panel).to_parquet(args.out_dir / "bulk_base.parquet")
        else:
            print("  bulk_target arm: X is not a dense dataset; skipping (arm will be absent)")
    else:
        print(f"  bulk_target arm: {args.bulk_base} not present; skipping")

    t_delta[panel].to_parquet(args.out_dir / "tahoe_delta.parquet")
    t_key.to_parquet(args.out_dir / "tahoe_key.parquet")
    t_base[panel].to_parquet(args.out_dir / "tahoe_base.parquet")
    l_delta[panel].to_parquet(args.out_dir / "l1000_delta.parquet")
    l_key.to_parquet(args.out_dir / "l1000_key.parquet")
    dmean[panel].to_parquet(args.out_dir / "l1000_base.parquet")

    grid = [f"{s}|{a}" for s in SOURCES for a in ARMS]
    (args.out_dir / "plan.json").write_text(
        json.dumps(
            {
                "git_sha": git_sha(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
                "panel_size": int(len(panel)),
                "grid": grid,
                "n_l1000_lines": int(pd.Series(l_lines).nunique()),
                "n_l1000_pairs": int(len(l_key)),
                "n_tahoe_pairs": int(len(t_key)),
                "shared_drugs": drugs,
                "args": {k: str(v) for k, v in vars(args).items()},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"plan: {len(grid)} cells ({len(SOURCES)} sources x {len(ARMS)} arms) -> {args.out_dir}")
    print(f"      submit --array=0-{len(grid) - 1}")


if __name__ == "__main__":
    main()
