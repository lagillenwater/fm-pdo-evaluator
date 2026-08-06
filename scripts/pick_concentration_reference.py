"""Reference values for "are the shortlists just the pan-toxic drugs?" (slide 6, panel 4).

Selection gap@k rewards putting a genuinely sensitive drug in the top k, but a handful of GDSC2
compounds are cytotoxic on nearly every line, so a model that learned only drug potency scores
well without knowing anything about the cell line. Before that can be read off a model, we need
the reference: on a panel of this size, how concentrated are the *observed* best drugs, and how
often is the true best drug already one of the broadly-active ones?

Both quantities come from measured GDSC2 AUC alone -- no model, no predictions -- so they can be
computed now and are what the model columns will be judged against.

    python3 scripts/pick_concentration_reference.py [--n-lines 50] [--n-boot 300]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENTS = Path("data/raw/coderdata/gdscv2_experiments.tsv.gz")
DRUGS = Path("data/raw/coderdata/gdscv2_drugs.tsv.gz")
CIDS = Path("data/static/gdsc2_auc_pubchem_cids.txt")
METRIC = "fit_auc"


def auc_matrix(cids: set[str]) -> pd.DataFrame:
    """Cell line x drug matrix of GDSC2 fit_auc, restricted to the Tahoe-matched compounds."""
    drugs = pd.read_csv(DRUGS, sep="\t", usecols=["improve_drug_id", "pubchem_id"])
    drugs = drugs.dropna(subset=["pubchem_id"])
    drugs["cid"] = drugs["pubchem_id"].map(lambda c: str(int(float(c))))
    keep = drugs[drugs["cid"].isin(cids)].drop_duplicates("improve_drug_id")
    id2cid = dict(zip(keep["improve_drug_id"], keep["cid"], strict=True))

    exp = pd.read_csv(EXPERIMENTS, sep="\t")
    exp = exp[(exp["dose_response_metric"] == METRIC) & exp["improve_drug_id"].isin(id2cid)]
    exp = exp.assign(cid=exp["improve_drug_id"].map(id2cid))
    # One AUC per (line, compound); duplicate assay rows are averaged.
    return exp.pivot_table(index="improve_sample_id", columns="cid", values="dose_response_value")


def breadth(mat: np.ndarray) -> np.ndarray:
    """Per drug, the fraction of lines whose AUC for it falls below that line's own median.

    "Broadly active" in the sense the question means: works on nearly everything. Computed on the
    row-wise comparison so it is invariant to per-line offsets in the assay.
    """
    below = mat < np.nanmedian(mat, axis=1, keepdims=True)
    return below.mean(axis=0)


def concentration(mat: np.ndarray, broad: np.ndarray, ks=(1, 3)) -> dict:
    """How concentrated are the observed best drugs, and how often are they broadly active."""
    order = np.argsort(mat, axis=1)  # ascending AUC: the line's genuinely best drugs first
    top1 = order[:, 0]
    out = {
        "distinct_top1": int(np.unique(top1).size),
        "modal_top1_share": float(np.bincount(top1, minlength=mat.shape[1]).max() / mat.shape[0]),
    }
    for k in ks:
        picks = order[:, :k]
        out[f"broad_share_top{k}"] = float(broad[picks].any(axis=1).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-lines", type=int, default=50, help="cell lines per resampled panel")
    ap.add_argument("--n-drugs", type=int, default=26, help="compounds per resampled panel")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--broad-quantile", type=float, default=0.75, help="top quartile by breadth")
    ap.add_argument("--out", type=Path, default=Path("results/pick_concentration_reference.json"))
    args = ap.parse_args()

    cids = {t for t in CIDS.read_text().split() if t}
    mat = auc_matrix(cids)
    # Complete cases only: gap@k needs every line scored on every drug of the panel.
    mat = mat.dropna(axis=1, thresh=int(0.8 * len(mat))).dropna(axis=0)
    print(f"complete GDSC2 panel: {mat.shape[0]} lines x {mat.shape[1]} compounds")

    full = mat.to_numpy(dtype=float)
    broad_full = breadth(full)
    cut = np.quantile(broad_full, args.broad_quantile)
    print(f"breadth cut (top quartile): {cut:.3f}; {int((broad_full >= cut).sum())} broad actives")

    whole = concentration(full, broad_full >= cut)
    print(f"\nwhole panel ({mat.shape[0]} lines): {json.dumps(whole)}")

    rng = np.random.default_rng(0)
    # Resample to the Check-2 geometry: ~50 held-out lines x ~26 scored compounds. Both matter --
    # distinct_top1 is bounded by the panel width, and breadth is re-derived within each panel.
    draws = []
    for _ in range(args.n_boot):
        li = rng.choice(full.shape[0], args.n_lines, replace=False)
        di = rng.choice(full.shape[1], min(args.n_drugs, full.shape[1]), replace=False)
        sub = full[np.ix_(li, di)]
        b = breadth(sub)
        draws.append(concentration(sub, b >= np.quantile(b, args.broad_quantile)))
    boot = pd.DataFrame(draws)
    summary = {
        "n_lines_panel": args.n_lines,
        "n_drugs_panel": args.n_drugs,
        "n_drugs_source": int(mat.shape[1]),
        "n_lines_source": int(mat.shape[0]),
        "whole_panel": whole,
    }
    print(f"\nresampled to {args.n_lines}-line panels (n_boot={args.n_boot}):")
    for col in boot.columns:
        lo, hi = np.quantile(boot[col], [0.025, 0.975])
        summary[col] = {"mean": float(boot[col].mean()), "lo": float(lo), "hi": float(hi)}
        print(f"  {col:20s} {boot[col].mean():.3f}   95% [{lo:.3f}, {hi:.3f}]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
