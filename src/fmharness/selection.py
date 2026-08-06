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
