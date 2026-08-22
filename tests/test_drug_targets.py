"""Tests for the known-biology (drug-target-expression) positive control."""

from __future__ import annotations

import pandas as pd

from fmharness.drug_targets import DRUG_TARGET_GENES, score_target_gene_predictors


def test_score_target_gene_predictors_detects_a_planted_dependency() -> None:
    # target expression falls monotonically while AUC (less sensitive = worse) rises --
    # a clean planted "higher target expression -> more sensitive" relationship.
    # interaction_rho requires >=2 drugs per patient to be non-degenerate (it removes
    # each patient's own cross-drug mean first, per its own docstring) -- with one drug
    # per patient here, "global" (plain Spearman y_true vs y_pred) is the right check.
    patients = [f"p{i}" for i in range(6)]
    design = pd.DataFrame(
        {"patient": patients, "drug": ["D1"] * 6, "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    )
    baseline = pd.DataFrame({"G1": [10.0, 8.0, 6.0, 4.0, 2.0, 0.0]}, index=pd.Index(patients))

    s = score_target_gene_predictors(design, baseline, targets={"D1": "G1"}, n_perm=200, seed=0)

    assert s["n"] == 6
    assert s["global"] > 0.9


def test_score_target_gene_predictors_returns_empty_when_no_drug_covered() -> None:
    design = pd.DataFrame({"patient": ["p1"], "drug": ["UNKNOWN"], "y": [1.0]})
    baseline = pd.DataFrame({"G1": [1.0]}, index=pd.Index(["p1"]))

    s = score_target_gene_predictors(design, baseline, targets={"D1": "G1"})

    assert s["n"] == 0


def test_score_target_gene_predictors_skips_patients_missing_a_baseline() -> None:
    design = pd.DataFrame({"patient": ["p1", "p2"], "drug": ["D1", "D1"], "y": [1.0, 2.0]})
    baseline = pd.DataFrame({"G1": [5.0]}, index=pd.Index(["p1"]))  # p2 has no baseline row

    s = score_target_gene_predictors(design, baseline, targets={"D1": "G1"})

    assert s["n"] == 1


def test_drug_target_genes_covers_every_soragni_drug_cid() -> None:
    # the 26 CIDs actually screened in Soragni (confirmed via build_sample_design against
    # the sarcoma tranche, 2026-08-22) -- this map must not silently drop one.
    soragni_cids = {
        "10113978", "11222830", "11442891", "11556711", "11626560", "11640390",
        "11707110", "123631", "135398510", "148124", "208908", "216239",
        "23725625", "25102847", "25126798", "3062316", "5284616", "5311497",
        "5330286", "54761306", "60700", "60750", "6442177", "6918837",
        "9823820", "9865515",
    }
    assert soragni_cids <= set(DRUG_TARGET_GENES)
