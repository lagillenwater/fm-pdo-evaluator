"""Tests for the Modality registry."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fmharness.modality import (
    Gdsc2Auc,
    Modality,
    SoragniViability,
    ThresholdedModality,
)


class _FakeAucModality:
    """A minimal regression Modality for testing the wrapper, no real data."""

    def load(self, repo: Path) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "patient": ["p1", "p1", "p2", "p2"],
                "drug": ["d1", "d2", "d1", "d2"],
                "y": [10.0, 60.0, 30.0, 80.0],
            }
        )

    def direction(self) -> str:
        return "lower_is_better"

    def recommended_cv(self) -> str:
        return "5fold"

    def task_type(self) -> str:
        return "regression"

    def name(self) -> str:
        return "fake_auc"


def test_fake_modality_satisfies_protocol() -> None:
    assert isinstance(_FakeAucModality(), Modality)


def test_thresholded_modality_emits_binary_y_below() -> None:
    wrapped = ThresholdedModality(_FakeAucModality(), threshold=50.0, responder_is="below")
    design = wrapped.load(Path("."))
    assert wrapped.task_type() == "classification"
    assert design.set_index(["patient", "drug"])["y"].to_dict() == {
        ("p1", "d1"): 1.0,
        ("p1", "d2"): 0.0,
        ("p2", "d1"): 1.0,
        ("p2", "d2"): 0.0,
    }


def test_thresholded_modality_emits_binary_y_above() -> None:
    wrapped = ThresholdedModality(_FakeAucModality(), threshold=50.0, responder_is="above")
    design = wrapped.load(Path("."))
    assert design.set_index(["patient", "drug"])["y"].to_dict() == {
        ("p1", "d1"): 0.0,
        ("p1", "d2"): 1.0,
        ("p2", "d1"): 0.0,
        ("p2", "d2"): 1.0,
    }


def test_thresholded_modality_delegates_direction_and_cv() -> None:
    wrapped = ThresholdedModality(_FakeAucModality(), threshold=50.0, responder_is="below")
    assert wrapped.direction() == "lower_is_better"
    assert wrapped.recommended_cv() == "5fold"


def test_thresholded_modality_satisfies_protocol() -> None:
    wrapped = ThresholdedModality(_FakeAucModality(), threshold=50.0, responder_is="below")
    assert isinstance(wrapped, Modality)


_REPO = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(
    not (_REPO / "data/raw/gdsc2_sarcoma/gdsc2/GDSC2_fitted_dose_response_27Oct23.xlsx").exists(),
    reason="requires local GDSC2 raw data",
)
def test_gdsc2_auc_loads_real_data() -> None:
    design = Gdsc2Auc().load(_REPO)
    assert {"patient", "drug", "y"} <= set(design.columns)
    assert design["patient"].nunique() > 100  # full pan-cancer panel, not a small subset


@pytest.mark.skipif(
    not (_REPO / "data/raw/soragni").exists(),
    reason="requires local Soragni raw data",
)
def test_soragni_viability_loads_real_data() -> None:
    design = SoragniViability().load(_REPO)
    assert {"patient", "drug", "y"} <= set(design.columns)
    assert design["patient"].nunique() == 17
