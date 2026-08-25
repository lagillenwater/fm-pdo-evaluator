"""Why do the `oracle` and `additive` rows of Check 2 return identical numbers?

Job 31631728 (2026-08-24) scored `oracle` and `additive` as byte-identical across all three
penalties -- global 0.628, interaction -0.095, per-drug -0.149, regret 0.264/0.111. They are
built from different data (`oracle` is the real measured delta, per (line, drug); `additive` is
the drug mean over the OTHER lines, identical for every line within a drug), so identical output
should be impossible unless the fit is degenerate.

The hypothesis this script tests, in three parts:

1. `additive` has ZERO within-drug feature variance by construction. `penalized_preds` fits per
   drug, so `StandardScaler` maps its training block to all-zeros and the penalized model can
   only fit an intercept. Its prediction is therefore the training-fold mean AUC for that drug,
   no matter what the features contain.
2. `oracle` at p ~ 2000 with ~40 training lines per fold may be shrunk so hard that it lands on
   the same intercept-only solution. `make_penalty("l2")` searches `logspace(-2, 3, 12)`, so the
   largest alpha available is 1e3.
3. If (2) holds, the alpha should be sitting AT that ceiling. `fmharness.probe.base` uses
   `logspace(0, 8, 9)` and states in a comment that the path must reach well above 1e3 or the
   slope cannot shrink to zero on an uninformative embedding -- so the two paths in this repo
   disagree by five orders of magnitude. Re-fitting `oracle` on the wider path settles whether
   the ceiling is binding.

Outputs a verdict for each part. Nothing is written; this only reads the deltas bundle and the
AUC tranche.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from fmharness.check2 import repr_by_drug
from fmharness.data.loaders import load_tranche
from fmharness.deltas import loo_baseline_source
from fmharness.evaluation import build_sample_design

WIDE_ALPHAS = np.logspace(-2, 8, 24)
NARROW_ALPHAS = np.logspace(-2, 3, 12)  # what make_penalty("l2") actually uses


def fold_assignment(lines: list[str], folds: int) -> tuple[dict[str, int], int]:
    """Reproduce ``score_check2``'s fold split exactly (``i % n_folds`` over sorted lines)."""
    n_folds = max(1, min(folds, len(lines)))
    return {ln: i % n_folds for i, ln in enumerate(lines)}, n_folds


def within_drug_variance(rep: dict[str, pd.DataFrame]) -> pd.Series:
    """Mean per-gene standard deviation across lines, per drug.

    Near-zero means every line has the same feature vector for that drug, so a per-drug fit
    cannot use the features at all.
    """
    return pd.Series(
        {d: float(np.nanmean(f.to_numpy(dtype=np.float64).std(axis=0))) for d, f in rep.items()}
    )


def fit_and_report(
    rep: dict[str, pd.DataFrame],
    design: pd.DataFrame,
    fold_of: dict[str, int],
    n_folds: int,
    lines: list[str],
    alphas: np.ndarray,
    label: str,
) -> tuple[pd.DataFrame, list[float]]:
    """Fit the per-drug ridge exactly as ``penalized_preds`` does, recording chosen alphas.

    Returns the predictions frame and every alpha the inner CV selected, so the caller can check
    how often the search landed on the largest value in the path (which would mean the ceiling,
    not the data, chose the amount of shrinkage).
    """
    auc_by_drug = {
        str(d): dict(zip(g["patient"].astype(str), g["y"], strict=False))
        for d, g in design.groupby("drug")
    }
    rows: list[tuple[str, str, float, float]] = []
    chosen: list[float] = []
    collapse: list[tuple[float, float]] = []
    for drug, auc in auc_by_drug.items():
        fdf = rep.get(drug)
        if fdf is None or fdf.empty:
            continue
        fdf = fdf.copy()
        fdf.index = pd.Index([str(i) for i in fdf.index])
        lines_d = [ln for ln in fdf.index if ln in auc]
        if len(lines_d) < 8:
            continue
        for f in range(n_folds):
            held = {ln for ln in lines if fold_of[ln] == f}
            tr = [ln for ln in lines_d if ln not in held]
            te = [ln for ln in lines_d if ln in held]
            if len(tr) < 5 or not te:
                continue
            sc = StandardScaler().fit(fdf.loc[tr].to_numpy(dtype=np.float64))
            y_tr = [auc[ln] for ln in tr]
            model = RidgeCV(alphas=alphas).fit(
                sc.transform(fdf.loc[tr].to_numpy(dtype=np.float64)), y_tr
            )
            chosen.append(float(model.alpha_))
            pred = model.predict(sc.transform(fdf.loc[te].to_numpy(dtype=np.float64)))
            # The decisive check: is the prediction just the training-fold mean? If the
            # coefficient term contributes nothing, the "representation" is not being used at
            # all and the row is the drug mean wearing a different label.
            train_mean = float(np.mean(y_tr))
            coef_norm = float(np.linalg.norm(np.asarray(model.coef_, dtype=np.float64)))
            for ln, p in zip(te, pred, strict=False):
                rows.append((ln, drug, float(auc[ln]), float(p)))
                collapse.append((float(p) - train_mean, coef_norm))
    cols = pd.Index(["patient", "drug", "y_true", "y_pred"])
    out = pd.DataFrame(rows, columns=cols)
    dev = np.array([c[0] for c in collapse])
    cn = np.array([c[1] for c in collapse])
    print(f"  [{label}] {len(out)} predictions, {len(chosen)} per-drug-fold fits")
    print(f"      |pred - training-fold mean|: max {np.abs(dev).max():.4g}, mean {np.abs(dev).mean():.4g}")
    print(f"      ridge coefficient L2 norm:   max {cn.max():.4g}, median {np.median(cn):.4g}")
    return out, chosen


def main() -> None:
    """Run the three checks and print a verdict for each."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deltas-bundle", default="tahoe_deltas")
    ap.add_argument("--auc-tranche", default="gdsc2_sarcoma")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    bdir = Path(args.deltas_bundle)
    real_delta = pd.read_parquet(bdir / "real_delta.parquet")
    real_key = pd.read_parquet(bdir / "real_key.parquet")
    base = pd.read_parquet(bdir / "base.parquet")
    hvg = pd.Index(real_delta.var(axis=0).sort_values(ascending=False).index[: args.n_hvg])
    _, design = build_sample_design(
        load_tranche(args.auc_tranche, repo), "all", "auc", drug_key="pubchem_cid"
    )

    add_delta, add_key = loo_baseline_source("additive", real_delta, real_key, base, k=args.k)
    reps = {
        "additive": repr_by_drug(add_delta, add_key, hvg),
        "oracle": repr_by_drug(real_delta, real_key, hvg),
    }

    print("\n" + "=" * 78)
    print("PART 1 -- are the two representations actually different data?")
    print("=" * 78)
    shared = sorted(set(reps["additive"]) & set(reps["oracle"]))
    print(f"  drugs in both: {len(shared)}")
    diffs = []
    for d in shared[:10]:
        a, o = reps["additive"][d], reps["oracle"][d]
        common = a.index.intersection(o.index)
        g = a.columns.intersection(o.columns)
        if len(common) and len(g):
            diffs.append(float(np.nanmax(np.abs(a.loc[common, g].to_numpy() - o.loc[common, g].to_numpy()))))
    print(f"  max |additive - oracle| over the first 10 shared drugs: {max(diffs):.4g}")
    print("  VERDICT: different data" if max(diffs) > 1e-9 else "  VERDICT: IDENTICAL data")

    print("\n" + "=" * 78)
    print("PART 2 -- within-drug feature variance (can a per-drug fit use the features?)")
    print("=" * 78)
    for name in ("additive", "oracle"):
        v = within_drug_variance(reps[name])
        print(f"  {name:>9}: mean per-gene sd across lines = {v.mean():.6f}  (min {v.min():.6f}, max {v.max():.6f})")
    print("  A value of ~0 means every line shares one feature vector -> intercept-only fit.")

    lines = sorted(set(real_key["patient"].astype(str)))
    fold_of, n_folds = fold_assignment(lines, args.folds)

    print("\n" + "=" * 78)
    print("PART 3 -- do the fits collapse to the same intercept-only solution?")
    print("=" * 78)
    preds = {}
    alphas_used = {}
    for name in ("additive", "oracle"):
        preds[name], alphas_used[name] = fit_and_report(
            reps[name], design, fold_of, n_folds, lines, NARROW_ALPHAS, f"{name}/narrow"
        )
    m = preds["additive"].merge(
        preds["oracle"], on=["patient", "drug"], suffixes=("_add", "_orc")
    )
    d = np.abs(m["y_pred_add"].to_numpy() - m["y_pred_orc"].to_numpy())
    print(f"  matched predictions: {len(m)}")
    print(f"  max |additive_pred - oracle_pred| = {d.max():.6g}, mean = {d.mean():.6g}")
    print("  VERDICT: predictions IDENTICAL" if d.max() < 1e-9 else "  VERDICT: predictions differ")

    ceiling = NARROW_ALPHAS.max()
    for name in ("additive", "oracle"):
        a = np.array(alphas_used[name])
        at_ceiling = float(np.mean(np.isclose(a, ceiling)))
        print(f"  {name:>9}: alpha at path ceiling ({ceiling:g}) in {at_ceiling:.1%} of fits; median alpha {np.median(a):.4g}")

    print("\n" + "=" * 78)
    print("PART 4 -- does widening the alpha path change the oracle fit?")
    print("=" * 78)
    wide_pred, wide_alphas = fit_and_report(
        reps["oracle"], design, fold_of, n_folds, lines, WIDE_ALPHAS, "oracle/wide"
    )
    mw = preds["oracle"].merge(wide_pred, on=["patient", "drug"], suffixes=("_narrow", "_wide"))
    dw = np.abs(mw["y_pred_narrow"].to_numpy() - mw["y_pred_wide"].to_numpy())
    aw = np.array(wide_alphas)
    print(f"  max |narrow - wide| = {dw.max():.6g}, mean = {dw.mean():.6g}")
    print(f"  wide path: alpha at its ceiling ({WIDE_ALPHAS.max():g}) in {float(np.mean(np.isclose(aw, WIDE_ALPHAS.max()))):.1%} of fits; median {np.median(aw):.4g}")
    print(f"  fits whose chosen alpha EXCEEDS the narrow path's ceiling: {float(np.mean(aw > ceiling)):.1%}")
    print("\n  If that last number is large, make_penalty's logspace(-2, 3, 12) was capping the")
    print("  shrinkage for every high-dimensional representation, not just the oracle.")


if __name__ == "__main__":
    main()
