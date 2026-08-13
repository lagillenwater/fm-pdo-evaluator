"""Check-2 scoring building blocks: penalty models, per-drug representation splitting, and
the leave-cell-line-out penalized-regression fit, plus ``score_check2``, the composition of
these into the fixed-signature-readout scoring and the representation-controlled penalized
grid.

Shared by ``scripts/score_generation_eval.py`` and ``scripts/check2_registry_driver.py``.
Stays leakage-agnostic -- ``score_check2`` scores whatever ``design`` frame it is handed, the
same way ``evaluation.score_delta_sources`` does for Check 1. ``filter_leakage`` is always the
caller's job, not this module's.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler

# The Hallmark proliferation sets -- the two that cleared the gate's random-gene-set control
# (G2M clearly, E2F marginally); the death sets (P53, apoptosis) add only noise on Tahoe. A
# ``proliferation`` readout scores just these, so a real but weak signal is not diluted away.
PROLIFERATION = ("HALLMARK_E2F_TARGETS", "HALLMARK_G2M_CHECKPOINT")
FIXED_READOUTS = ("hallmark", "proliferation")  # fixed-signature readouts, applied to delta sources
PENALTY_NAMES = ("l2", "l1", "en")  # penalized regressions for the representation-controlled grid


def make_penalty(name: str) -> object:
    """A fresh ALPHA-CV-TUNED penalized model: l2=RidgeCV (efficient GCV), l1=LassoCV, en=
    ElasticNetCV (both inner 3-fold on the training lines). Tuning the penalty per representation
    makes the grid model-fair -- a fixed alpha over-/under-regularizes some representations and
    flips the ranking (Kurilov 2020)."""
    if name == "l2":
        return RidgeCV(alphas=np.logspace(-2, 3, 12))
    if name == "l1":
        return LassoCV(n_alphas=30, cv=3, max_iter=20000, random_state=0)
    if name == "en":
        return ElasticNetCV(l1_ratio=0.5, n_alphas=30, cv=3, max_iter=20000, random_state=0)
    raise ValueError(f"unknown penalty {name!r}")


def load_line_matrix(path: Path) -> pd.DataFrame:
    """Load a per-cell-line feature matrix (index = line id) for the check-2 grid, from a
    ``.h5ad`` (X + obs_names), ``.parquet``, or ``.csv``. Used to fold a precomputed FM
    embedding (one vector per line) in head-to-head with expr/pca."""
    if path.suffix == ".h5ad":
        a = ad.read_h5ad(path)
        x = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        return pd.DataFrame(x, index=pd.Index([str(o) for o in a.obs_names])).astype(float)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, index_col=0)
    df.index = pd.Index([str(i) for i in df.index])
    return df.astype(float)


def repr_by_drug(
    delta: pd.DataFrame, key: pd.DataFrame, genes: pd.Index
) -> dict[str, pd.DataFrame]:
    """Split a delta source into ``{drug: DataFrame[line x genes]}`` for per-drug regression."""
    d = delta.reindex(columns=genes).fillna(0.0)
    pat = key["patient"].astype(str).to_numpy()
    drg = key["drug"].astype(str).to_numpy()
    out: dict[str, pd.DataFrame] = {}
    for drug in pd.unique(drg):
        m = d[drg == drug]
        m.index = pd.Index(pat[drg == drug])
        out[str(drug)] = m
    return out


def penalized_preds(
    feat: dict[str, pd.DataFrame] | Callable[[str], pd.DataFrame],
    design: pd.DataFrame,
    fold_of: dict[str, int],
    n_folds: int,
    uniq_lines: list[str],
    penalty: str,
    *,
    min_lines: int = 8,
    min_train: int = 5,
) -> pd.DataFrame:
    """Per-drug penalized regression (representation -> AUC), leave-cell-line-out by fold.

    ``feat`` maps a drug to a (line x gene) frame -- a dict for a delta source, or a callable for a
    drug-independent representation (baseline expression). For each drug the model is fit on the
    training-fold lines' features vs AUC and predicts the held-fold lines; the StandardScaler is fit
    on the training lines only, so a single held line (true LOO) is scored leakage-free. All
    representations share one model class, so a difference is the representation, not the model.
    Returns preds (patient, drug, y_true, y_pred); y_pred is an AUC estimate (same sign as y_true).
    """
    auc_by_drug = {
        str(d): dict(zip(g["patient"].astype(str), g["y"], strict=False))
        for d, g in design.groupby("drug")
    }
    rows: list[tuple[str, str, float, float]] = []
    for drug, auc in auc_by_drug.items():
        fdf = feat(drug) if callable(feat) else feat.get(drug)  # type: ignore[union-attr]
        if fdf is None or fdf.empty:
            continue
        fdf = fdf.copy()
        fdf.index = pd.Index([str(i) for i in fdf.index])
        lines_d = [ln for ln in fdf.index if ln in auc]
        if len(lines_d) < min_lines:
            continue
        for f in range(n_folds):
            held = {ln for ln in uniq_lines if fold_of[ln] == f}
            tr = [ln for ln in lines_d if ln not in held]
            te = [ln for ln in lines_d if ln in held]
            if len(tr) < min_train or not te:
                continue
            sc = StandardScaler().fit(fdf.loc[tr].to_numpy(dtype=np.float64))
            # make_penalty returns `object` (RidgeCV/LassoCV/ElasticNetCV share no common
            # base class pyright can see .fit/.predict on) -- both calls need an ignore.
            model = make_penalty(penalty).fit(  # type: ignore[attr-defined]
                sc.transform(fdf.loc[tr].to_numpy(dtype=np.float64)), [auc[ln] for ln in tr]
            )
            te_x = sc.transform(fdf.loc[te].to_numpy(dtype=np.float64))
            pred = model.predict(te_x)  # type: ignore[attr-defined]
            rows.extend(
                (ln, drug, float(auc[ln]), float(p)) for ln, p in zip(te, pred, strict=False)
            )
    cols = ["patient", "drug", "y_true", "y_pred"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
