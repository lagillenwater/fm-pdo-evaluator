"""Label ceiling: how reproducible is cell-line drug response ACROSS independent screens?

The Tahoe generation eval scores a transcriptome-derived prediction against GDSC2 AUC and finds
no within-drug (cell-line-specific) signal. Before attributing that to the model, this measures
the ceiling: how well GDSC2's OWN within-drug line rankings agree with an independent viability
screen (CTRPv2 / PRISM). If two screens do not agree within-drug, no transcriptome -- Tahoe,
Stack, or anything -- can predict that axis; the ceiling is the target's own irreproducibility,
not the model.

Screens are joined on BIOLOGICAL identity (DepMap cell-line id + PubChem CID), not CoderData's
per-build internal ids, so independently downloaded datasets line up. Agreement is reported on
the shared pairs three ways, reusing the harness scorer so the numbers are directly comparable to
check 2:
  * global      -- Spearman over all pairs (dominated by the drug main effect)
  * interaction -- within-drug rank corr after removing each line's mean (the harness headline;
    the cell-line-specific axis) + its within-drug label-permutation p
  * per-drug median Spearman -- the classic cross-screen consistency (Haibe-Kains) statistic

  # GDSC2 (local) vs CTRPv2 and PRISM downloaded to <dir> via `coderdata.download`:
  uv run python scripts/label_ceiling.py --comp-dir <dir> --comparators ctrpv2,prism
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fmharness.evaluation import score_predictions

METRIC_PREF = ("fit_auc", "auc", "aac")  # first metric present in both screens is used


def _read(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix == ".gz" or path.name.endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def _find(comp_dir: Path, prefix: str, kind: str) -> Path:
    hits = sorted(comp_dir.glob(f"{prefix}_{kind}*"))
    if not hits:
        raise SystemExit(f"missing {prefix}_{kind}* under {comp_dir}")
    return hits[0]


def load_screen(exp: Path, samp: Path, drug: Path) -> tuple[pd.DataFrame, set[str]]:
    """One row per (DepMap id, PubChem CID, metric) -> mean value, plus the metrics present.

    Maps CoderData's improve_sample_id -> DepMap id (samples table) and improve_drug_id -> PubChem
    CID (drugs table), so the screen is keyed by identity shared across independent downloads.
    """
    e, s, d = _read(exp), _read(samp), _read(drug)
    dep = (
        s[s["other_id_source"] == "DepMap"][["improve_sample_id", "other_id"]]
        .rename(columns={"other_id": "depmap"})
        .drop_duplicates("improve_sample_id")
    )
    dd = d[["improve_drug_id", "pubchem_id"]].dropna(subset=["pubchem_id"]).copy()
    dd["cid"] = dd["pubchem_id"].map(lambda c: str(int(float(c))))
    dd = dd.drop_duplicates("improve_drug_id")
    m = e.merge(dep, on="improve_sample_id").merge(
        dd[["improve_drug_id", "cid"]], on="improve_drug_id"
    )
    agg = (
        m.groupby(["depmap", "cid", "dose_response_metric"])["dose_response_value"]
        .mean()
        .reset_index()
    )
    return agg, set(agg["dose_response_metric"].unique())


def compare(ref: pd.DataFrame, comp: pd.DataFrame, metric: str, n_perm: int) -> dict[str, object]:
    r = ref[ref["dose_response_metric"] == metric][["depmap", "cid", "dose_response_value"]]
    c = comp[comp["dose_response_metric"] == metric][["depmap", "cid", "dose_response_value"]]
    j = r.merge(c, on=["depmap", "cid"], suffixes=("_ref", "_comp"))
    if j.empty:
        return {"metric": metric, "n_pairs": 0}
    preds = pd.DataFrame(
        {
            "patient": j["depmap"].to_numpy(),
            "drug": j["cid"].to_numpy(),
            "y_true": j["dose_response_value_ref"].to_numpy(),
            "y_pred": j["dose_response_value_comp"].to_numpy(),
        }
    )
    s = score_predictions(preds, n_perm=n_perm)
    # per-drug within-line Spearman between the two screens (Haibe-Kains consistency).
    per = [
        spearmanr(g["dose_response_value_ref"], g["dose_response_value_comp"])[0]
        for _, g in j.groupby("cid")
        if len(g) >= 5
    ]
    per = np.array([x for x in per if np.isfinite(x)])
    return {
        "metric": metric,
        "n_pairs": len(j),
        "n_lines": int(j["depmap"].nunique()),
        "n_drugs": int(j["cid"].nunique()),
        "global": s["global"],
        "interaction": s["interaction"],
        "p_label": s["p_label"],
        "perdrug_median_rho": round(float(np.median(per)), 3) if len(per) else float("nan"),
        "perdrug_frac_pos": round(float(np.mean(per > 0)), 3) if len(per) else float("nan"),
        "n_drugs_scored": len(per),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref-dir", default="data/raw/coderdata", help="dir with gdscv2_* tables")
    ap.add_argument("--ref-prefix", default="gdscv2")
    ap.add_argument("--comp-dir", required=True, help="dir with the comparator *_experiments etc.")
    ap.add_argument("--comparators", default="ctrpv2,prism", help="comma-separated prefixes")
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--out", default="results/label_ceiling.csv")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent
    ref_dir = Path(args.ref_dir) if Path(args.ref_dir).is_absolute() else repo / args.ref_dir
    comp_dir = Path(args.comp_dir) if Path(args.comp_dir).is_absolute() else repo / args.comp_dir

    ref, ref_metrics = load_screen(
        _find(ref_dir, args.ref_prefix, "experiments"),
        _find(ref_dir, args.ref_prefix, "samples"),
        _find(ref_dir, args.ref_prefix, "drugs"),
    )
    rows: list[dict[str, object]] = []
    for prefix in [p.strip() for p in args.comparators.split(",") if p.strip()]:
        comp, comp_metrics = load_screen(
            _find(comp_dir, prefix, "experiments"),
            _find(comp_dir, prefix, "samples"),
            _find(comp_dir, prefix, "drugs"),
        )
        metric = next((m for m in METRIC_PREF if m in ref_metrics and m in comp_metrics), None)
        if metric is None:
            print(f"[{prefix}] no shared metric with the reference; skipping")
            continue
        row = {
            "reference": args.ref_prefix,
            "comparator": prefix,
            **compare(ref, comp, metric, args.n_permutations),
        }
        rows.append(row)
        print(f"[{prefix}] {row}")

    out = pd.DataFrame(rows)
    out_path = Path(args.out) if Path(args.out).is_absolute() else repo / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("\n=== label ceiling: GDSC2 within-drug agreement with independent screens ===")
    print(out.to_string(index=False) if not out.empty else "(no comparisons)")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
