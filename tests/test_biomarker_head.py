"""Tests for the Estimator-conforming biomarker wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmharness.probe.biomarker_head import BiomarkerEstimator
from fmharness.probe.estimator import Estimator


def _synthetic() -> tuple[pd.DataFrame, list[str], list[str], list[float], list[dict[str, str]], dict[str, dict[str, set[str]]], set[str], dict[str, int]]:
    # 4 patients, 2 drugs. drugA has an "expr" biomarker rule keyed on entrez "100";
    # drugB has a "mut" rule. patients p1/p3 carry the mutation.
    x_log = pd.DataFrame(
        {"100": [2.0, 0.0, 2.0, 0.0]},
        index=pd.Index(["p1", "p2", "p3", "p4"]),
    )
    rows_patient = ["p1", "p2", "p3", "p4"] * 2
    rows_drug = ["drugA"] * 4 + ["drugB"] * 4
    rows_y = [10.0, 60.0, 15.0, 65.0, 20.0, 70.0, 25.0, 75.0]
    biomarkers = [
        {"drug": "drugA", "gene": "GENE100", "kind": "expr", "direction": "sensitize"},
        {"drug": "drugB", "gene": "GENEMUT", "kind": "mut", "direction": "sensitize"},
    ]
    alt = {"mut": {"GENEMUT": {"p1", "p3"}}, "amp": {}, "del": {}}
    wes = {"p1", "p2", "p3", "p4"}
    sym2ent = {"GENE100": 100, "GENEMUT": 999}
    return x_log, rows_patient, rows_drug, rows_y, biomarkers, alt, wes, sym2ent


def test_biomarker_estimator_satisfies_estimator_protocol() -> None:
    _, _, _, _, biomarkers, alt, wes, sym2ent = _synthetic()
    est = BiomarkerEstimator(biomarkers, alt, wes, sym2ent)
    assert isinstance(est, Estimator)


def test_biomarker_estimator_fit_predict_uses_the_matching_rule() -> None:
    x_log, patients, drugs, y, biomarkers, alt, wes, sym2ent = _synthetic()
    est = BiomarkerEstimator(biomarkers, alt, wes, sym2ent).fit(x_log, drugs, np.asarray(y))
    x_log_indexed = x_log.loc[patients]
    base, residual = est.predict_parts(x_log_indexed, drugs)
    assert base.shape == (8,)
    assert residual.shape == (8,)
    # drugA rows: residual should differ between p1/p3 (expr=2.0) and p2/p4 (expr=0.0)
    drug_a_mask = [d == "drugA" for d in drugs]
    resid_a = residual[drug_a_mask]
    assert resid_a[0] != resid_a[1]  # p1 (high expr) differs from p2 (low expr)


def test_biomarker_estimator_zero_residual_for_drug_with_no_rule() -> None:
    x_log, patients, drugs, y, biomarkers, alt, wes, sym2ent = _synthetic()
    est = BiomarkerEstimator(biomarkers, alt, wes, sym2ent).fit(x_log, drugs, np.asarray(y))
    x_log_indexed = x_log.loc[patients]
    _, residual = est.predict_parts(x_log_indexed, ["drugC"] * 4)
    np.testing.assert_array_equal(residual, np.zeros(4))


def test_biomarker_estimator_single_patient_predict_batch_is_finite() -> None:
    # Leave-one-out is SoragniViability.recommended_cv(), so a one-row test fold
    # is the normal case. Normalizing at predict time would give a single-element
    # SD of NaN (and `NaN or 1.0` is NaN, since NaN is truthy), silently poisoning
    # base + residual. With fit-time stats frozen, the residual stays finite.
    x_log, _patients, drugs, y, biomarkers, alt, wes, sym2ent = _synthetic()
    est = BiomarkerEstimator(biomarkers, alt, wes, sym2ent).fit(x_log, drugs, np.asarray(y))
    for patient in ["p1", "p2", "p3", "p4"]:
        one = x_log.loc[[patient]]
        base, residual = est.predict_parts(one, ["drugA"])
        assert np.all(np.isfinite(residual)), f"non-finite residual for {patient}"
        assert np.all(np.isfinite(base + residual))


def test_biomarker_estimator_normalization_is_frozen_at_fit() -> None:
    # Two separate single-patient batches must be normalized by the SAME
    # (fit-time mean, SD), not by each batch's own statistics -- otherwise
    # residuals are not comparable across leave-one-out folds.
    x_log, _patients, drugs, y, biomarkers, alt, wes, sym2ent = _synthetic()
    est = BiomarkerEstimator(biomarkers, alt, wes, sym2ent).fit(x_log, drugs, np.asarray(y))

    # Frozen stats computed by hand from the FIT data: column "100" = [2,0,2,0].
    fit_mean = float(np.mean([2.0, 0.0, 2.0, 0.0]))
    fit_sd = float(np.std([2.0, 0.0, 2.0, 0.0], ddof=1))  # pandas Series.std default
    # "sensitize" direction flips the sign of the z-score.
    expected = {p: -((v - fit_mean) / fit_sd) for p, v in [("p1", 2.0), ("p2", 0.0)]}

    _, resid_p1 = est.predict_parts(x_log.loc[["p1"]], ["drugA"])
    _, resid_p2 = est.predict_parts(x_log.loc[["p2"]], ["drugA"])
    assert np.isclose(resid_p1[0], expected["p1"])
    assert np.isclose(resid_p2[0], expected["p2"])

    # Same rows, batched together: batch composition changes nothing.
    _, resid_both = est.predict_parts(x_log.loc[["p1", "p2"]], ["drugA", "drugA"])
    np.testing.assert_allclose(resid_both, [expected["p1"], expected["p2"]])


def test_biomarker_estimator_skips_expr_rule_with_no_training_spread() -> None:
    # A gene constant across the training patients has SD 0; dividing by it would
    # yield Inf/NaN, so the rule contributes no residual instead.
    x_log, _patients, drugs, y, biomarkers, alt, wes, sym2ent = _synthetic()
    flat = x_log.assign(**{"100": [1.0, 1.0, 1.0, 1.0]})
    est = BiomarkerEstimator(biomarkers, alt, wes, sym2ent).fit(flat, drugs, np.asarray(y))
    _, residual = est.predict_parts(flat.loc[["p1"]], ["drugA"])
    assert residual[0] == 0.0


def test_biomarker_estimator_fit_rejects_bare_array() -> None:
    _x_log, _patients, drugs, y, biomarkers, alt, wes, sym2ent = _synthetic()
    est = BiomarkerEstimator(biomarkers, alt, wes, sym2ent)
    with pytest.raises(TypeError, match="patient_id"):
        est.fit(np.zeros((4, 1)), drugs, np.asarray(y))
