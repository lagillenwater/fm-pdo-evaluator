"""Tests for the GDSC mechanism-of-action join."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from fmharness.moa import load_moa, moa_hit_rate_at_k, normalize_drug, pathway_map


def test_normalize_drug_strips_case_and_punctuation() -> None:
    assert normalize_drug("AZD-8055") == "azd8055"
    assert normalize_drug("crizotinib") == "crizotinib"
    assert normalize_drug("5-Fluorouracil") == "5fluorouracil"
    assert normalize_drug("Trametinib (DMSO_TF solvate)") == "trametinibdmsotfsolvate"


def test_load_moa_and_pathway_map(tmp_path: Path) -> None:
    src = tmp_path / "compounds.csv"
    pd.DataFrame(
        {
            "DRUG_ID": [1, 2, 3],
            "SCREENING_SITE": ["MGH", "MGH", "WTSI"],
            "DRUG_NAME": ["Crizotinib", "AZD8055", "Trametinib"],
            "SYNONYMS": ["PF-02341066", "-", "GSK1120212"],
            "TARGET": ["MET, ALK", "mTORC1, mTORC2", "MEK1, MEK2"],
            "TARGET_PATHWAY": ["RTK signalling", "PI3K/MTOR signalling", "ERK MAPK signalling"],
        }
    ).to_csv(src, index=False)

    moa = load_moa(src)
    assert moa.loc["crizotinib", "target_pathway"] == "RTK signalling"
    assert moa.loc["azd8055", "target"] == "mTORC1, mTORC2"

    # Tahoe writes these names differently; the normalized join still lands them.
    pw = pathway_map(moa, ["crizotinib", "AZD-8055", "Trametinib", "LJI308"])
    assert pw["crizotinib"] == "RTK signalling"
    assert pw["AZD-8055"] == "PI3K/MTOR signalling"
    assert pw["Trametinib"] == "ERK MAPK signalling"
    assert "LJI308" not in pw  # unmatched drugs are omitted, not mapped to a sentinel


def test_load_moa_detects_pathway_disagreements(tmp_path: Path) -> None:
    """Compounds screened at multiple sites can have conflicting pathway annotations."""
    src = tmp_path / "compounds.csv"
    pd.DataFrame(
        {
            "DRUG_ID": [1, 2],
            "SCREENING_SITE": ["MGH", "SANGER"],
            "DRUG_NAME": ["Dasatinib", "Dasatinib"],
            "SYNONYMS": ["-", "-"],
            "TARGET": ["ABL", "ABL"],
            "TARGET_PATHWAY": ["RTK signaling", "Other, kinases"],
        }
    ).to_csv(src, index=False)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        moa = load_moa(src)
        assert len(w) == 1
        assert "dasatinib" in str(w[0].message).lower()
        assert "differing target_pathway" in str(w[0].message)

    # Result should be deterministic (sorted by drug name, then index)
    assert len(moa) == 1
    assert moa.index[0] == "dasatinib"
    # After deduplication with sort determinism, first drug name alphabetically wins
    assert pd.notna(moa.loc["dasatinib", "target_pathway"])


def test_load_moa_preserves_missing_values(tmp_path: Path) -> None:
    """Missing TARGET or TARGET_PATHWAY values are preserved as NaN, not stringified."""
    src = tmp_path / "compounds.csv"
    pd.DataFrame(
        {
            "DRUG_ID": [1, 2],
            "SCREENING_SITE": ["MGH", "MGH"],
            "DRUG_NAME": ["Crizotinib", "UnknownDrug"],
            "SYNONYMS": ["-", "-"],
            "TARGET": ["MET, ALK", None],
            "TARGET_PATHWAY": ["RTK signalling", None],
        }
    ).to_csv(src, index=False)

    moa = load_moa(src)
    assert moa.loc["crizotinib", "target_pathway"] == "RTK signalling"
    # Missing value should be NaN, not the string "nan"
    assert pd.isna(moa.loc["unknowndrug", "target_pathway"])
    assert not isinstance(moa.loc["unknowndrug", "target_pathway"], str)


def test_pathway_map_omits_missing_pathways(tmp_path: Path) -> None:
    """Drugs with missing pathway annotations are omitted from pathway_map output."""
    src = tmp_path / "compounds.csv"
    pd.DataFrame(
        {
            "DRUG_ID": [1, 2, 3],
            "SCREENING_SITE": ["MGH", "MGH", "MGH"],
            "DRUG_NAME": ["Crizotinib", "UnknownDrug", "Trametinib"],
            "SYNONYMS": ["-", "-", "-"],
            "TARGET": ["MET, ALK", None, "MEK1, MEK2"],
            "TARGET_PATHWAY": ["RTK signalling", None, "ERK MAPK signalling"],
        }
    ).to_csv(src, index=False)

    moa = load_moa(src)
    pw = pathway_map(moa, ["Crizotinib", "UnknownDrug", "Trametinib"])
    assert "Crizotinib" in pw
    assert "Trametinib" in pw
    assert "UnknownDrug" not in pw  # missing pathway, not included
    assert pw["Crizotinib"] == "RTK signalling"
    assert pw["Trametinib"] == "ERK MAPK signalling"


def test_load_moa_invariant_to_row_order(tmp_path: Path) -> None:
    """Deduplication result is the same regardless of CSV row order (invariance test).

    This is the critical test: it verifies that when a compound has conflicting pathway
    annotations across screening sites, load_moa always selects the same pathway whether
    the CSV is presented in forward or reversed order. This proves the deduplication is
    deterministic, not row-order-dependent.
    """
    src = tmp_path / "compounds.csv"
    src_reversed = tmp_path / "compounds_reversed.csv"

    # Fixture: Dasatinib at two sites with different pathways
    df = pd.DataFrame(
        {
            "DRUG_ID": [1, 2],
            "SCREENING_SITE": ["MGH", "SANGER"],
            "DRUG_NAME": ["Dasatinib", "Dasatinib"],
            "SYNONYMS": ["-", "-"],
            "TARGET": ["ABL", "ABL"],
            "TARGET_PATHWAY": ["RTK signaling", "Other, kinases"],
        }
    )
    df.to_csv(src, index=False)

    # Reversed version: same rows but in opposite order
    df.iloc[::-1].to_csv(src_reversed, index=False)

    # Load both versions
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        moa_forward = load_moa(src)
        moa_reversed = load_moa(src_reversed)

    # Both should have exactly one row for dasatinib
    assert len(moa_forward) == 1
    assert len(moa_reversed) == 1

    # Both should return the SAME pathway (alphabetically-first: "Other, kinases" < "RTK")
    pathway_forward = moa_forward.loc["dasatinib", "target_pathway"]
    pathway_reversed = moa_reversed.loc["dasatinib", "target_pathway"]
    assert pathway_forward == pathway_reversed, (
        f"Invariance violated: forward={pathway_forward}, reversed={pathway_reversed}"
    )
    # Verify it's the alphabetically-first one
    assert pathway_forward == "Other, kinases"


def test_moa_hit_rate_at_k_counts_pathway_not_compound() -> None:
    # Line A's true best is d_mek1 (MEK). The model ranks d_mek2 first -- wrong compound,
    # right pathway -- so it is a hit at k=1. Line B's true best is d_mek1 too, but the
    # model ranks the RTK compound first, so B only hits once k reaches 2.
    preds = pd.DataFrame(
        {
            "patient": ["A", "A", "A", "B", "B", "B"],
            "drug": ["d_mek1", "d_mek2", "d_rtk"] * 2,
            "y_true": [0.1, 0.5, 0.9, 0.1, 0.5, 0.9],
            "y_pred": [0.5, 0.1, 0.9, 0.5, 0.9, 0.1],
        }
    )
    pathway = {"d_mek1": "ERK MAPK", "d_mek2": "ERK MAPK", "d_rtk": "RTK signalling"}
    hits = moa_hit_rate_at_k(preds, pathway, ks=(1, 2))
    assert np.isclose(hits[1], 0.5)  # A hits, B misses
    assert np.isclose(hits[2], 1.0)  # both hit


def test_moa_hit_rate_at_k_skips_unannotated_best_drug() -> None:
    preds = pd.DataFrame(
        {
            "patient": ["A", "A", "B", "B"],
            "drug": ["d_x", "d_mek1", "d_mek1", "d_mek2"],
            "y_true": [0.1, 0.9, 0.1, 0.5],
            "y_pred": [0.1, 0.9, 0.1, 0.5],
        }
    )
    # A's true best (d_x) has no pathway -> A is skipped; only B counts, and B hits.
    hits = moa_hit_rate_at_k(preds, {"d_mek1": "ERK MAPK", "d_mek2": "ERK MAPK"}, ks=(1,))
    assert np.isclose(hits[1], 1.0)
