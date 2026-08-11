"""Tests for the Estimator-conforming bilinear wrapper."""

from __future__ import annotations

import numpy as np

from fmharness.probe.bilinear_head import BilinearEstimator
from fmharness.probe.estimator import Estimator


def _synthetic(n_patients: int = 20, emb_dim: int = 6, fp_dim: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    drugs = ["d1", "d2", "d3"]
    patients = [f"p{i}" for i in range(n_patients)]
    fingerprints = {d: rng.standard_normal(fp_dim) for d in drugs}
    rows_emb, rows_drug, rows_y, rows_groups = [], [], [], []
    z_by_patient = {p: rng.standard_normal(emb_dim) for p in patients}
    for p in patients:
        for d in drugs:
            z = z_by_patient[p]
            g = fingerprints[d]
            limit = min(emb_dim, fp_dim)
            y = float((z[:limit] * g[:limit]).sum()) + rng.normal(scale=0.01)
            rows_emb.append(z)
            rows_drug.append(d)
            rows_y.append(y)
            rows_groups.append(p)
    return (
        np.asarray(rows_emb, dtype=np.float64),
        rows_drug,
        np.asarray(rows_y, dtype=np.float64),
        rows_groups,
        fingerprints,
    )


def test_bilinear_estimator_satisfies_estimator_protocol() -> None:
    _, _, _, _, fingerprints = _synthetic()
    assert isinstance(BilinearEstimator(fingerprints), Estimator)


def test_bilinear_estimator_fits_and_predicts_shapes() -> None:
    emb, drugs, y, groups, fingerprints = _synthetic()
    est = BilinearEstimator(fingerprints, n_components=3, seed=0).fit(emb, drugs, y, groups)
    base, residual = est.predict_parts(emb, drugs)
    assert base.shape == (len(y),)
    assert residual.shape == (len(y),)


def test_bilinear_estimator_falls_back_to_drug_mean_for_unknown_drug() -> None:
    emb, drugs, y, groups, fingerprints = _synthetic()
    est = BilinearEstimator(fingerprints, n_components=3, seed=0).fit(emb, drugs, y, groups)
    base, residual = est.predict_parts(emb[:1], ["unknown_drug"])
    assert residual[0] == 0.0  # no fingerprint -> no residual term
    assert base[0] != 0.0  # falls back to the global mean, not zero


def test_bilinear_estimator_recovers_signal_better_than_drug_mean_alone() -> None:
    # The synthetic y depends on z . g -- a fitted bilinear estimator should
    # explain materially more residual variance than predicting 0 residual
    # for everyone (the drug-mean-only floor).
    emb, drugs, y, groups, fingerprints = _synthetic(n_patients=40, seed=1)
    est = BilinearEstimator(fingerprints, n_components=4, seed=0).fit(emb, drugs, y, groups)
    base, residual = est.predict_parts(emb, drugs)
    mse_fitted = float(np.mean((base + residual - y) ** 2))
    mse_drug_mean_only = float(np.mean((base - y) ** 2))
    assert mse_fitted < mse_drug_mean_only * 0.5
