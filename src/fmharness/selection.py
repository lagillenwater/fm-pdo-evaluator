"""Selection-metric machinery for the check-2 shortlist audit.

`gap@k` in raw AUC rewards ranking broadly-potent compounds, because a pan-cytotoxic drug is
close to the best drug for most cell lines. Scoring in each drug's rank among lines instead
makes every drug's marginal uniform, so breadth of potency carries no advantage and only
line-specific ordering can score. The concentration summary answers the prior question --
whether a representation is producing line-specific shortlists at all, or re-picking the same
few toxic compounds for everyone.
"""

from __future__ import annotations

import pandas as pd


def within_drug_percentile(
    preds: pd.DataFrame, cols: tuple[str, ...] = ("y_true", "y_pred")
) -> pd.DataFrame:
    """Replace each named column with its within-drug percentile rank in ``(0, 1]``.

    Ranking inside each drug removes that drug's location and scale, so a compound that is
    potent on every line no longer sits closer to the per-line optimum than a selective one.
    Order within a drug is preserved. ``preds`` is not modified.
    """
    out = preds.copy()
    for col in cols:
        out[col] = out.groupby("drug")[col].rank(pct=True)
    return out


def broadly_active_drugs(preds: pd.DataFrame, frac: float = 0.5) -> set[str]:
    """Drugs below the line's own median response for more than ``frac`` of lines.

    ``y_true`` is AUC-like (lower is more sensitive), so these are the compounds that work on
    nearly everything. Picking them is partly correct behaviour -- the observed best drug is
    usually one of them -- which is why the shortlist audit reports the observed row as its
    reference rather than treating any such pick as a failure.
    """
    median = preds.groupby("patient")["y_true"].transform("median")
    share = (preds["y_true"] < median).groupby(preds["drug"]).mean()
    return set(share.index[share > frac].astype(str))


def shortlist_concentration(
    preds: pd.DataFrame, pred_col: str = "y_pred"
) -> dict[str, float | str]:
    """How concentrated a representation's top-1 picks are across lines.

    Returns the number of distinct drugs ever ranked first, the most-picked drug and its share
    of lines, and the share of top-1 picks that are broadly active. Call with
    ``pred_col="y_true"`` for the observed reference: the truth is itself concentrated, so a
    model is only collapsed if it is *more* concentrated than that row.
    """
    top1 = preds.loc[preds.groupby("patient")[pred_col].idxmin(), ["patient", "drug"]]
    counts = top1["drug"].value_counts()
    active = broadly_active_drugs(preds)
    return {
        "distinct": float(counts.size),
        "modal_drug": str(counts.index[0]),
        "modal_share": float(counts.iloc[0] / len(top1)),
        "broadly_active_share": float(top1["drug"].isin(active).mean()),
        "n_lines": float(len(top1)),
    }
