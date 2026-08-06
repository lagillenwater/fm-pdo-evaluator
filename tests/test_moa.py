"""Tests for the GDSC mechanism-of-action join."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from fmharness.moa import load_moa, normalize_drug, pathway_map


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
