"""Classification readouts: top-k hit rate, Brier score, expected calibration error.

Named in the original harness plan and never built (the harness stayed
AUC/interaction-focused). Operate on the same preds[patient, drug, y_true, y_pred]
frame shape ``score_predictions`` uses in ``evaluation.py`` -- ``y_true`` is a
binary responder label (from a ``ThresholdedModality``, ``modality.py``), ``y_pred``
is a predicted probability of response.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def top_k_hit_rate(preds: pd.DataFrame, k: int) -> float:
    """Fraction of true responders (y_true == 1) captured in the top-k by y_pred.

    Ranks all rows by descending predicted probability, takes the top k, and
    reports what share of the panel's actual responders are in that shortlist --
    the same "does the shortlist contain the answer" question regret_norm_at_k
    asks for continuous response, adapted to a binary label.
    """
    n_responders = int((preds["y_true"] == 1.0).sum())
    if n_responders == 0:
        return float("nan")
    top_k = preds.nlargest(k, "y_pred")
    hits = int((top_k["y_true"] == 1.0).sum())
    return hits / n_responders


def brier_score(preds: pd.DataFrame) -> float:
    """Mean squared error between predicted probability and binary outcome.

    0 is perfect, 0.25 is the score of a constant p=0.5 predictor against a
    balanced panel, 1 is maximally wrong (confident and always incorrect).
    """
    y_true = preds["y_true"].to_numpy(dtype=np.float64)
    y_pred = preds["y_pred"].to_numpy(dtype=np.float64)
    return float(np.mean((y_pred - y_true) ** 2))


def expected_calibration_error(preds: pd.DataFrame, n_bins: int = 10) -> float:
    """Mean absolute gap between predicted probability and empirical response rate,
    within equal-width probability bins, weighted by bin size.

    A well-calibrated model's predicted probabilities should match the actual
    fraction of responders among samples given that probability; this is the
    standard ECE definition (Guo et al. 2017).
    """
    y_true = preds["y_true"].to_numpy(dtype=np.float64)
    y_pred = preds["y_pred"].to_numpy(dtype=np.float64)
    n = len(y_true)
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_pred, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        bin_confidence = float(y_pred[mask].mean())
        bin_accuracy = float(y_true[mask].mean())
        ece += (count / n) * abs(bin_confidence - bin_accuracy)
    return ece
