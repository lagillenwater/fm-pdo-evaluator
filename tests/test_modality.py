"""Tests for the Modality registry."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fmharness.modality import (
    CtrpAuc,
    Direction,
    Gdsc2Auc,
    Modality,
    PrismAuc,
    SoragniViability,
    TaskType,
    ThresholdedModality,
)


class _FakeAucModality:
    """A minimal regression Modality for testing the wrapper, no real data."""

    def load_with_features(self, repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        x_df = pd.DataFrame({"gene1": [1.0, 2.0]}, index=pd.Index(["p1", "p2"]))
        design = pd.DataFrame(
            {
                "patient": ["p1", "p1", "p2", "p2"],
                "drug": ["d1", "d2", "d1", "d2"],
                "y": [10.0, 60.0, 30.0, 80.0],
            }
        )
        return x_df, design

    def load(self, repo: Path) -> pd.DataFrame:
        return self.load_with_features(repo)[1]

    def direction(self) -> Direction:
        return "lower_is_better"

    def recommended_cv(self) -> str:
        return "5fold"

    def task_type(self) -> TaskType:
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


def test_thresholded_modality_direction_is_higher_is_better_and_cv_delegates() -> None:
    # Thresholding absorbs the base's sign convention into the label (y == 1 is
    # the responder either way), so a higher predicted probability is always
    # better -- delegating direction() would flip every metric on a
    # lower_is_better base. recommended_cv, in contrast, is a property of the
    # cohort's size and rightly delegates.
    for responder_is in ("below", "above"):
        wrapped = ThresholdedModality(
            _FakeAucModality(), threshold=50.0, responder_is=responder_is
        )
        assert wrapped.direction() == "higher_is_better"
        assert wrapped.recommended_cv() == "5fold"
        assert wrapped.task_type() == "classification"
    # Also true when wrapping a real modality whose own direction is the opposite.
    assert Gdsc2Auc().direction() == "lower_is_better"
    assert (
        ThresholdedModality(Gdsc2Auc(), threshold=0.8, responder_is="below").direction()
        == "higher_is_better"
    )


def test_thresholded_modality_name_distinguishes_responder_tail() -> None:
    # Same base and threshold, opposite tails: semantically opposite targets that
    # must not collide in a results table keyed by name.
    below = ThresholdedModality(_FakeAucModality(), threshold=50.0, responder_is="below")
    above = ThresholdedModality(_FakeAucModality(), threshold=50.0, responder_is="above")
    assert below.name() != above.name()
    assert below.name() == "fake_auc_responder_below_50"
    assert above.name() == "fake_auc_responder_above_50"


def test_concrete_modality_metadata_without_data() -> None:
    # The two tests that load real data both skip without local raw files, which
    # left every concrete Modality's metadata unexecuted -- exactly how a wrong
    # direction() or a colliding name() goes unnoticed. These need no repo access.
    expected: list[tuple[Modality, Direction, str, TaskType, str]] = [
        (Gdsc2Auc(), "lower_is_better", "5fold", "regression", "gdsc2_auc"),
        (CtrpAuc(), "lower_is_better", "5fold", "regression", "ctrp_auc"),
        (PrismAuc(), "lower_is_better", "5fold", "regression", "prism_auc"),
        (
            SoragniViability(),
            "lower_is_better",
            "loo",  # n=17 organoids: leave-one-out, not 5-fold
            "regression",
            "sarcoma_organoids_2024_viability_tumor",
        ),
    ]
    for modality, direction, cv, task, name in expected:
        assert isinstance(modality, Modality)
        assert modality.direction() == direction, name
        assert modality.recommended_cv() == cv, name
        assert modality.task_type() == task, name
        assert modality.name() == name
    # The rna_source is the substrate, and it must show up in the name so tumor-
    # and organoid-RNA runs against the same target stay distinguishable.
    assert SoragniViability(rna_source="organoid").name() == "sarcoma_organoids_2024_viability_organoid"


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
    not (_REPO / "data/raw/sarcoma_organoids_2024").exists(),
    reason="requires local Soragni raw data",
)
def test_sarcoma_organoids_2024_viability_loads_real_data() -> None:
    design = SoragniViability().load(_REPO)
    assert {"patient", "drug", "y"} <= set(design.columns)
    assert design["patient"].nunique() == 17


@pytest.mark.skipif(
    not (_REPO / "data/raw/gdsc2_sarcoma/gdsc2/GDSC2_fitted_dose_response_27Oct23.xlsx").exists(),
    reason="requires local GDSC2 raw data",
)
def test_gdsc2_auc_load_with_features_matches_load() -> None:
    modality = Gdsc2Auc(cancer_type_filter=["sarcoma"])
    x_df, design = modality.load_with_features(_REPO)
    design_only = modality.load(_REPO)
    pd.testing.assert_frame_equal(design, design_only)
    # x_df is indexed by patient; build_sample_design filters design down to
    # patients with an assay of the chosen metric but does not filter x_df
    # back, so design's patients are a subset of x_df's index, not necessarily
    # equal (a patient can have expression with no matching drug-response row).
    assert set(design["patient"]) <= set(x_df.index.astype(str))


@pytest.mark.skipif(
    not (_REPO / "data/raw/sarcoma_organoids_2024").exists(),
    reason="requires local Soragni raw data",
)
def test_sarcoma_organoids_2024_viability_load_with_features_matches_load() -> None:
    modality = SoragniViability()
    x_df, design = modality.load_with_features(_REPO)
    design_only = modality.load(_REPO)
    pd.testing.assert_frame_equal(design, design_only)
    assert set(x_df.index.astype(str)) == set(design["patient"])


@pytest.mark.skipif(
    not (_REPO / "data/raw/sarcoma_organoids_2024").exists(),
    reason="requires local Soragni raw data",
)
def test_sarcoma_organoids_2024_viability_load_with_features_loads_the_tranche_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The bug this fixes: a caller needing both design and features used to
    # call Modality.load() (one full tranche load) and then reconstruct
    # features by hand (a second, redundant tranche load). load_with_features
    # must get both out of a single underlying load.
    import fmharness.modality as modality_module

    real_load_tranche = modality_module.load_tranche
    call_count = 0

    def counting_load_tranche(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        return real_load_tranche(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(modality_module, "load_tranche", counting_load_tranche)
    SoragniViability().load_with_features(_REPO)
    assert call_count == 1


def test_thresholded_modality_load_with_features_delegates_and_thresholds_y_only() -> None:
    wrapped = ThresholdedModality(_FakeAucModality(), threshold=50.0, responder_is="below")
    x_df, design = wrapped.load_with_features(Path("."))
    # x_df is untouched -- thresholding only transforms the label.
    pd.testing.assert_frame_equal(x_df, _FakeAucModality().load_with_features(Path("."))[0])
    assert design.set_index(["patient", "drug"])["y"].to_dict() == {
        ("p1", "d1"): 1.0,
        ("p1", "d2"): 0.0,
        ("p2", "d1"): 1.0,
        ("p2", "d2"): 0.0,
    }
    # load() and load_with_features() agree.
    pd.testing.assert_frame_equal(design, wrapped.load(Path(".")))
