"""Tests for the Estimator-conforming biomarker wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmharness.probe.biomarker_head import BiomarkerEstimator
from fmharness.probe.estimator import Estimator


def _synthetic() -> tuple[pd.DataFrame, list[str], list[str], list[float], list[dict[str, str]], dict[str, dict[str, set[str]]], set[str], dict[str, int]]:  # noqa: E501
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
