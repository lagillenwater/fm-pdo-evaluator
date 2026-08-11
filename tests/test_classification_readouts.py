"""Tests for classification readouts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmharness.classification_readouts import (
    brier_score,
    expected_calibration_error,
    top_k_hit_rate,
)


def _preds(y_true: list[float], y_pred: list[float]) -> pd.DataFrame:
    n = len(y_true)
    return pd.DataFrame(
        {
            "patient": [f"p{i}" for i in range(n)],
            "drug": ["d1"] * n,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )


def test_top_k_hit_rate_perfect_predictions() -> None:
    # y_true is binary (1 = responder); y_pred is a predicted probability.
    preds = _preds([1.0, 0.0, 1.0, 0.0], [0.9, 0.1, 0.8, 0.2])
    assert np.isclose(top_k_hit_rate(preds, k=2), 1.0)


def test_top_k_hit_rate_worst_case() -> None:
    preds = _preds([1.0, 0.0, 1.0, 0.0], [0.1, 0.9, 0.2, 0.8])
    assert np.isclose(top_k_hit_rate(preds, k=1), 0.0)


def test_brier_score_perfect_predictions_is_zero() -> None:
    preds = _preds([1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0])
    assert np.isclose(brier_score(preds), 0.0)


def test_brier_score_worst_case_is_one() -> None:
    preds = _preds([1.0, 0.0], [0.0, 1.0])
    assert np.isclose(brier_score(preds), 1.0)


def test_brier_score_uninformative_half_probability() -> None:
    preds = _preds([1.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 0.5])
    assert np.isclose(brier_score(preds), 0.25)


def test_expected_calibration_error_perfect_calibration_is_zero() -> None:
    # Every predicted probability exactly matches the empirical response rate
    # within its own bin when there's one sample per bin and pred == true.
    preds = _preds([1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0])
    assert np.isclose(expected_calibration_error(preds, n_bins=2), 0.0)


def test_expected_calibration_error_penalizes_overconfidence() -> None:
    # Predicts near-certain responder for everyone, but only half actually respond.
    preds = _preds([1.0, 0.0, 1.0, 0.0], [0.95, 0.95, 0.95, 0.95])
    ece = expected_calibration_error(preds, n_bins=10)
    assert ece > 0.4  # bin mean confidence ~0.95 vs. empirical rate 0.5
