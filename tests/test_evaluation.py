"""Tests for shared evaluation metrics."""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest

from fmharness.cv import group_k_fold
from fmharness.evaluation import (
    de_fidelity,
    delta_fidelity,
    grouped_cv_predict,
    regret_norm_at_k,
    score_de_metrics,
    score_delta_sources,
    score_predictions,
)
from fmharness.probe import SimpleProbe
from fmharness.probe.biomarker_head import BiomarkerEstimator


def test_regret_norm_at_k() -> None:
    # y_true / y_pred are AUC-like (lower = better). Patient A's predicted ranking puts
    # its true-best drug first (regret 0); patient B's puts it last (regret 1 at k=1,
    # 0 once k covers all 3). Patients with no spread are skipped.
    preds = pd.DataFrame(
        {
            "patient": ["A", "A", "A", "B", "B", "B", "C", "C"],
            "drug": ["d1", "d2", "d3", "d1", "d2", "d3", "d1", "d2"],
            "y_true": [10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 5.0, 5.0],  # C flat -> skipped
            "y_pred": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 2.0],
        }
    )
    r = regret_norm_at_k(preds, ks=(1, 3))
    assert np.isclose(r[1], 0.5)  # A: 0, B: (30-10)/(30-10)=1 -> mean 0.5
    assert np.isclose(r[3], 0.0)  # top-3 covers every drug for both A and B


def test_delta_fidelity_matches_specific_and_flags_nonspecific() -> None:
    genes = pd.Index(list("abcde"))
    real = pd.DataFrame(
        [
            [3.0, 1.0, -1.0, -2.0, -1.0],  # (P1, d1)
            [-1.0, -2.0, 3.0, 1.0, -1.0],  # (P2, d1) -- a different response shape
        ],
        columns=genes,
    )
    key = pd.DataFrame({"patient": ["P1", "P2"], "drug": ["d1", "d1"]})

    # specific predictor: each pair predicts its own real delta -> matched r = 1, rank = 1,
    # and the matched correlation beats the correlation to the wrong pair.
    spec = delta_fidelity(real.copy(), key.copy(), real, key, n_hvg=None)
    assert np.allclose(spec["r"].to_numpy(), 1.0)
    assert np.allclose(spec["rank"].to_numpy(), 1.0)
    assert (spec["r"].to_numpy() > spec["r_offdiag"].to_numpy()).all()

    # non-specific predictor: BOTH pairs predict the same profile (P1's real delta). P2's
    # matched correlation is then no better than its correlation to the wrong (P1) pair,
    # so its specificity rank collapses -- the smooth-generator failure mode is caught.
    pred = pd.DataFrame([real.iloc[0].to_numpy(), real.iloc[0].to_numpy()], columns=genes)
    nonspec = delta_fidelity(pred, key.copy(), real, key, n_hvg=None)
    p2 = nonspec[nonspec["patient"] == "P2"].iloc[0]
    assert p2["rank"] == 0.0
    assert p2["r"] <= p2["r_offdiag"] + 1e-9


def test_delta_fidelity_restricts_to_hvgs() -> None:
    # only genes a, b vary across the two pairs; c, d, e are constant -> top-2 HVGs = a, b.
    genes = pd.Index(list("abcde"))
    real = pd.DataFrame(np.eye(2, 5) * 3.0 + 1.0, columns=genes)
    key = pd.DataFrame({"patient": ["P1", "P2"], "drug": ["d", "d"]})
    out = delta_fidelity(real.copy(), key, real, key, n_hvg=2)
    assert (out["n_genes"] == 2).all()


def test_score_delta_sources_builds_one_row_per_source() -> None:
    # two sources scored against the same real delta: "good" reconstructs it exactly
    # (matched r = 1 per pair), "bad" is its exact negation (matched r = -1 per pair) --
    # one row per source name, and the near-perfect source scores higher.
    genes = pd.Index(list("abcde"))
    real_delta = pd.DataFrame(
        [
            [3.0, 1.0, -1.0, -2.0, -1.0],  # (P1, d1)
            [-1.0, -2.0, 3.0, 1.0, -1.0],  # (P2, d1)
        ],
        columns=genes,
    )
    real_key = pd.DataFrame({"patient": ["P1", "P2"], "drug": ["d1", "d1"]})

    sources = {
        "good": (real_delta.copy(), real_key.copy()),
        "bad": (-real_delta, real_key.copy()),
    }
    table = score_delta_sources(sources, real_delta, real_key, n_hvg=None)
    assert set(table["source"]) == {"good", "bad"}
    by_source = table.set_index("source")
    assert by_source.loc["good", "r"] > by_source.loc["bad", "r"]
    assert by_source.loc["good", "r"] == 1.0
    assert by_source.loc["bad", "r"] == -1.0


def test_score_delta_sources_restricts_to_common_support_before_scoring() -> None:
    # "wide" covers 3 (patient, drug) pairs against real_key; "narrow" only covers 2 of
    # them. Without a common-support restriction (fmharness.deltas.restrict_common_support,
    # the Path-B fix), "wide" would be scored on 3 matched pairs and "narrow" on 2 --
    # different evaluation sets in the same table, not a fair head-to-head.
    genes = pd.Index(list("ab"))
    real_delta = pd.DataFrame([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], columns=genes)
    real_key = pd.DataFrame({"patient": ["P1", "P2", "P3"], "drug": ["d1", "d1", "d1"]})
    wide_delta = real_delta.copy()
    wide_key = real_key.copy()
    narrow_delta = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], columns=genes)
    narrow_key = pd.DataFrame({"patient": ["P1", "P2"], "drug": ["d1", "d1"]})

    sources = {"wide": (wide_delta, wide_key), "narrow": (narrow_delta, narrow_key)}
    table = score_delta_sources(sources, real_delta, real_key, n_hvg=None)
    by_source = table.set_index("source")
    assert by_source.loc["wide", "n_pairs"] == 2
    assert by_source.loc["narrow", "n_pairs"] == 2


def test_de_fidelity_scores_spearman_lfc_overlap_pr_auc_jaccard() -> None:
    # one (patient, drug) pair, 4 genes; real DE calls: A, B significant (padj<0.05,
    # |log2fc|>0.25), C, D not. Predicted delta ranks A, B highest by |value| -- matches the real
    # top-N=2 exactly -> perfect overlap/Jaccard/PR-AUC, and correlates with the real log2FC on
    # the significant genes -> spearman +1.
    de_calls = pd.DataFrame(
        {
            "patient": ["ACH-1"] * 4,
            "drug": ["100"] * 4,
            "gene": ["A", "B", "C", "D"],
            "log2fc": [3.0, 2.0, 0.1, -0.05],
            "padj": [0.001, 0.01, 0.9, 0.8],
            "significant": [True, True, False, False],
        }
    )
    pred_delta = pd.DataFrame({"A": [3.5], "B": [1.5], "C": [0.2], "D": [-0.1]})
    pred_key = pd.DataFrame({"patient": ["ACH-1"], "drug": ["100"]})

    f = de_fidelity(pred_delta, pred_key, de_calls)

    assert len(f) == 1
    row = f.iloc[0]
    assert row["n_sig_genes"] == 2
    assert row["de_overlap_accuracy"] == 1.0
    assert row["jaccard"] == 1.0
    assert np.isclose(row["de_spearman_lfc"], 1.0)
    assert 0.0 <= row["pr_auc"] <= 1.0


def test_de_fidelity_zero_significant_genes_gives_nan_rank_metrics_but_no_crash() -> None:
    de_calls = pd.DataFrame(
        {
            "patient": ["ACH-1"] * 2,
            "drug": ["100"] * 2,
            "gene": ["A", "B"],
            "log2fc": [0.1, -0.05],
            "padj": [0.9, 0.8],
            "significant": [False, False],
        }
    )
    pred_delta = pd.DataFrame({"A": [0.2], "B": [-0.1]})
    pred_key = pd.DataFrame({"patient": ["ACH-1"], "drug": ["100"]})

    f = de_fidelity(pred_delta, pred_key, de_calls)

    assert f.iloc[0]["n_sig_genes"] == 0
    assert np.isnan(f.iloc[0]["de_spearman_lfc"])
    assert np.isnan(f.iloc[0]["de_overlap_accuracy"])
    assert np.isnan(f.iloc[0]["jaccard"])
    assert np.isnan(f.iloc[0]["pr_auc"])  # only one class present (all non-significant)


def test_score_de_metrics_builds_one_row_per_source() -> None:
    de_calls = pd.DataFrame(
        {
            "patient": ["ACH-1", "ACH-1"],
            "drug": ["100", "100"],
            "gene": ["A", "B"],
            "log2fc": [3.0, 0.1],
            "padj": [0.001, 0.9],
            "significant": [True, False],
        }
    )
    pred_key = pd.DataFrame({"patient": ["ACH-1"], "drug": ["100"]})
    sources = {
        "good": (pd.DataFrame({"A": [3.0], "B": [0.1]}), pred_key),
        "bad": (pd.DataFrame({"A": [-3.0], "B": [0.1]}), pred_key),
    }

    table = score_de_metrics(sources, de_calls)

    assert set(table["source"]) == {"good", "bad"}
    assert list(table.columns) == [
        "source",
        "de_spearman_lfc",
        "pr_auc",
        "de_overlap_accuracy",
        "jaccard",
        "n_pairs",
    ]


def test_score_predictions_reports_interaction_and_null() -> None:
    # perfect predictions (y_pred == y_true) -> interaction rho = 1; the within-drug
    # permutation null almost never reaches 1, so p_label is small.
    rng = np.random.default_rng(0)
    y = rng.normal(size=15)
    preds = pd.DataFrame(
        {
            "patient": [f"P{i}" for i in range(5) for _ in range(3)],
            "drug": ["d1", "d2", "d3"] * 5,
            "y_true": y,
            "y_pred": y,
        }
    )
    s = score_predictions(preds, n_perm=200, seed=0)
    assert s["interaction"] == 1.0
    assert 0.0 <= s["p_label"] <= 0.2
    assert s["n"] == 15.0
    assert "regret@1" in s and "global" in s


def test_grouped_cv_predict_drives_biomarker_estimator() -> None:
    # BiomarkerEstimator.fit/predict_parts require a DataFrame indexed by
    # patient_id (biomarker lookup keys off patient identity, not row
    # position) -- grouped_cv_predict must hand it one, not a bare ndarray.
    patients = [f"p{i}" for i in range(8)]
    x_log = pd.DataFrame(
        {"100": [2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0]}, index=pd.Index(patients)
    )
    rows_patient = patients * 2
    rows_drug = ["drugA"] * 8 + ["drugB"] * 8
    rows_y = list(np.linspace(10.0, 70.0, 16))
    design = pd.DataFrame({"patient": rows_patient, "drug": rows_drug, "y": rows_y})
    biomarkers = [{"drug": "drugA", "gene": "GENE100", "kind": "expr", "direction": "sensitize"}]
    alt: dict[str, dict[str, set[str]]] = {"mut": {}, "amp": {}, "del": {}}
    sym2ent = {"GENE100": 100}

    factory = partial(BiomarkerEstimator, biomarkers, alt, set(patients), sym2ent)
    preds = grouped_cv_predict(factory, x_log, design, n_splits=4, seed=0)
    assert len(preds) == len(design)
    assert np.isfinite(preds["y_pred"]).all()


def _design_and_x(n_patients: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    patients = [f"p{i}" for i in range(n_patients)]
    rng = np.random.default_rng(0)
    x_df = pd.DataFrame(
        rng.standard_normal((n_patients, 4)),
        index=pd.Index(patients),
        columns=pd.Index(list("abcd")),
    )
    rows_patient = [p for p in patients for _ in range(2)]
    rows_drug = ["d1", "d2"] * n_patients
    design = pd.DataFrame(
        {"patient": rows_patient, "drug": rows_drug, "y": rng.standard_normal(n_patients * 2)}
    )
    return x_df, design


def test_grouped_cv_predict_rejects_both_n_splits_and_cv() -> None:
    x_df, design = _design_and_x()
    factory = partial(SimpleProbe, n_components=2)
    with pytest.raises(ValueError, match="exactly one"):
        grouped_cv_predict(factory, x_df, design, n_splits=3, cv=group_k_fold(3))


def test_grouped_cv_predict_rejects_neither_n_splits_nor_cv() -> None:
    x_df, design = _design_and_x()
    factory = partial(SimpleProbe, n_components=2)
    with pytest.raises(ValueError, match="exactly one"):
        grouped_cv_predict(factory, x_df, design)


def test_grouped_cv_predict_with_cv_matches_equivalent_n_splits() -> None:
    # A CVScheme built from group_k_fold(k) must drive grouped_cv_predict to
    # the exact same predictions as the original n_splits=k path -- cv= is a
    # generalization, not a different CV shape, for this equivalent case.
    x_df, design = _design_and_x()
    factory = partial(SimpleProbe, n_components=2, per_drug=True)
    via_n_splits = grouped_cv_predict(factory, x_df, design, n_splits=3, seed=0)
    via_cv = grouped_cv_predict(factory, x_df, design, cv=group_k_fold(3), seed=0)
    pd.testing.assert_frame_equal(
        via_n_splits.sort_values(["patient", "drug"]).reset_index(drop=True),
        via_cv.sort_values(["patient", "drug"]).reset_index(drop=True),
    )


def test_grouped_cv_predict_with_leave_subtype_out_cv() -> None:
    # A non-GroupKFold CVScheme (leave_subtype_out) must also drive
    # grouped_cv_predict correctly -- cv= is not special-cased to GroupKFold.
    from fmharness.cv import leave_subtype_out

    x_df, design = _design_and_x(n_patients=4)
    subtypes = {"p0": "A", "p1": "A", "p2": "B", "p3": "B"}
    factory = partial(SimpleProbe, n_components=2)
    preds = grouped_cv_predict(
        factory, x_df, design, cv=leave_subtype_out(subtypes, seed=0), seed=0
    )
    assert len(preds) == len(design)
