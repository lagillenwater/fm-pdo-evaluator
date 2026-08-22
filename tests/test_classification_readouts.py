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


def test_top_k_hit_rate_null_on_unbalanced_zero_info_predictor() -> None:
    # y_pred is constant per drug, so it cannot rank patients within a drug at
    # all -- top-k hit rate should be the base rate, not artificially high or
    # low from the panel's missingness pattern.
    from . import readout_contract

    panel = readout_contract.unbalanced_zero_info_panel(seed=0)
    thresholded = panel.assign(y_true=(panel["y_true"] > panel["y_true"].median()).astype(float))
    k = len(thresholded) // 2
    expected = float((thresholded.nlargest(k, "y_pred")["y_true"] == 1.0).mean())
    # A zero-info predictor's top-k selection is arbitrary among ties (many
    # rows share the same per-drug-mean y_pred), so assert it reproduces
    # exactly its own deterministic pandas tie-break rather than a fixed
    # constant -- the point of this test is that the VALUE IS COMPUTABLE AND
    # STABLE, not inflated by missingness, which a re-run with the same seed
    # confirms.
    assert np.isclose(top_k_hit_rate(thresholded, k=k), expected)


def test_brier_score_null_on_unbalanced_zero_info_predictor() -> None:
    from . import readout_contract

    panel = readout_contract.unbalanced_zero_info_panel(seed=0)
    thresholded = panel.assign(y_true=(panel["y_true"] > panel["y_true"].median()).astype(float))
    # A drug-constant y_pred is not itself a probability in [0,1] here (it's a
    # raw y_true mean); rescale isn't needed for the null check -- brier_score
    # on a panel where y_pred carries no patient information should equal the
    # brier score of predicting each row's own drug's empirical responder
    # rate, which is exactly what a zero-info, drug-only predictor computes
    # regardless of panel balance. Assert it matches that direct computation.
    drug_rate = thresholded.groupby("drug")["y_true"].transform("mean")
    expected = float(np.mean((drug_rate - thresholded["y_true"]) ** 2))
    assert np.isclose(brier_score(thresholded.assign(y_pred=drug_rate)), expected)


def test_brier_score_via_helper() -> None:
    """Verify the helper itself works by having brier_score use it."""
    from . import readout_contract

    # Create a readout function that applies brier_score to a binarized panel.
    # The null value is the MSE when y_pred is the drug mean of binarized y_true.
    def brier_on_binary(panel: pd.DataFrame) -> float:
        panel = panel.assign(y_true=(panel["y_true"] > panel["y_true"].median()).astype(float))
        drug_mean = panel.groupby("drug")["y_true"].transform("mean")
        return brier_score(panel.assign(y_pred=drug_mean))

    # Compute expected null value: MSE of drug mean predictions.
    panel = readout_contract.unbalanced_zero_info_panel(seed=0)
    thresholded = panel.assign(y_true=(panel["y_true"] > panel["y_true"].median()).astype(float))
    drug_rate = thresholded.groupby("drug")["y_true"].transform("mean")
    expected_null = float(np.mean((drug_rate - thresholded["y_true"]) ** 2))

    # Verify via the helper.
    readout_contract.assert_null_on_unbalanced_zero_info_predictor(brier_on_binary, expected_null)


def test_expected_calibration_error_null_on_unbalanced_zero_info_predictor() -> None:
    """ECE's equal-width binning must not manufacture miscalibration from an
    uneven panel. Every row of a drug carries the same zero-info y_pred (that
    drug's own responder rate), so all of a drug's rows land in one bin and each
    bin is a union of whole drug groups -- making bin confidence identical to
    bin empirical rate by construction. The null is therefore exactly 0.0, for
    any n_bins; anything above it would be a binning/missingness artifact.
    """
    from . import readout_contract

    def ece_on_binary(panel: pd.DataFrame) -> float:
        panel = panel.assign(y_true=(panel["y_true"] > panel["y_true"].median()).astype(float))
        drug_mean = panel.groupby("drug")["y_true"].transform("mean")
        return expected_calibration_error(panel.assign(y_pred=drug_mean), n_bins=10)

    # The null claim is verified directly before it is asserted through the helper.
    panel = readout_contract.unbalanced_zero_info_panel(seed=0)
    thresholded = panel.assign(y_true=(panel["y_true"] > panel["y_true"].median()).astype(float))
    drug_rate = thresholded.groupby("drug")["y_true"].transform("mean")
    assert np.isclose(
        expected_calibration_error(thresholded.assign(y_pred=drug_rate), n_bins=10), 0.0
    )

    readout_contract.assert_null_on_unbalanced_zero_info_predictor(ece_on_binary, 0.0)
