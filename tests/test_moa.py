"""Tests for the GDSC mechanism-of-action join."""

from __future__ import annotations

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
