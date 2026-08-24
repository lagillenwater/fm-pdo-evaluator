"""Tests for leakage filtering."""

from __future__ import annotations

from datetime import date

import pandas as pd

from fmharness.leakage import LeakageQueryable, filter_leakage
from fmharness.model_protocols import MockGenerator
from fmharness.models.adapter import KnownCorpusAdapter, MockAdapter
from fmharness.schema import ModelMetadata


class _KnownCorpusModel:
    """Test double: declares an exact pretraining line/drug set."""

    def __init__(
        self,
        lines: set[str] | None,
        drugs: set[str] | None,
        task_signal: str = "adjacent",
    ) -> None:
        self._lines = lines
        self._drugs = drugs
        self._task_signal = task_signal

    def version(self) -> str:
        return "known_corpus@v1"

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            pretraining_corpus="synthetic",
            pretraining_cutoff_date=date(2026, 1, 1),
            task_signal_in_pretrain=self._task_signal,  # type: ignore[arg-type]
            expected_input="log1p_cpm",
        )

    def embed(self, adata: object) -> object:
        raise NotImplementedError

    def pretraining_lines(self) -> set[str] | None:
        return self._lines

    def pretraining_drugs(self) -> set[str] | None:
        return self._drugs


def _design() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient": ["L1", "L1", "L2", "L2", "L3", "L3"],
            "drug": ["d1", "d2", "d1", "d2", "d1", "d2"],
            "y": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )


def test_known_corpus_model_satisfies_leakage_queryable() -> None:
    m = _KnownCorpusModel(lines=set(), drugs=set())
    assert isinstance(m, LeakageQueryable)
    assert not isinstance(MockAdapter(), LeakageQueryable)  # doesn't expose these methods


def test_filter_leakage_drops_doubly_exposed_pairs_always() -> None:
    # L1 and d1 are both in the pretraining corpus; only (L1, d1) is doubly exposed.
    model = _KnownCorpusModel(lines={"L1"}, drugs={"d1"}, task_signal="adjacent")
    # _KnownCorpusModel satisfies Encoder structurally but pyright is conservative.
    filtered, profile = filter_leakage(_design(), model)  # type: ignore[arg-type]
    assert not ((filtered["patient"] == "L1") & (filtered["drug"] == "d1")).any()
    assert len(filtered) == 5  # one row dropped
    assert profile.basis == "measured"


def test_filter_leakage_adjacent_signal_keeps_single_axis_overlap() -> None:
    # L1 overlaps (line only, no drug overlap) with task_signal "adjacent" --
    # single-axis rows must NOT be dropped, or a broadly-pretrained model
    # becomes untestable on almost any cohort.
    model = _KnownCorpusModel(lines={"L1"}, drugs=set(), task_signal="adjacent")
    # _KnownCorpusModel satisfies Encoder structurally but pyright is conservative.
    filtered, _ = filter_leakage(_design(), model)  # type: ignore[arg-type]
    assert len(filtered) == 6  # nothing dropped -- no doubly-exposed pairs exist


def test_filter_leakage_direct_signal_drops_single_axis_overlap_too() -> None:
    model = _KnownCorpusModel(lines={"L1"}, drugs=set(), task_signal="direct")
    # _KnownCorpusModel satisfies Encoder structurally but pyright is conservative.
    filtered, _ = filter_leakage(_design(), model)  # type: ignore[arg-type]
    assert not (filtered["patient"] == "L1").any()
    assert len(filtered) == 4  # both L1 rows dropped


def test_filter_leakage_unknown_basis_when_model_cannot_expose_corpus() -> None:
    design = _design()
    filtered, profile = filter_leakage(design, MockAdapter())
    pd.testing.assert_frame_equal(filtered, design)
    assert profile.basis == "unknown"
    assert profile.line_overlap_frac is None


def test_filter_leakage_works_for_generator_too() -> None:
    filtered, profile = filter_leakage(_design(), MockGenerator())
    pd.testing.assert_frame_equal(filtered, _design())
    assert profile.basis == "unknown"


def test_filter_leakage_unknown_basis_when_implements_protocol_but_returns_none() -> None:
    # Model structurally satisfies LeakageQueryable (has the methods) but
    # returns None from pretraining_lines, indicating corpus was not exposed.
    design = _design()
    model = _KnownCorpusModel(lines=None, drugs={"d1"})
    # _KnownCorpusModel satisfies Encoder structurally but pyright is conservative.
    filtered, profile = filter_leakage(design, model)  # type: ignore[arg-type]
    pd.testing.assert_frame_equal(filtered, design)
    assert profile.basis == "unknown"
    assert profile.line_overlap_frac is None
    assert profile.doubly_exposed_frac is None


def test_known_corpus_adapter_drives_filter_leakage_end_to_end() -> None:
    # KnownCorpusAdapter is a real (non-test-file) class a driver script can
    # construct -- proving filter_leakage has a genuine, non-test caller path,
    # not just the test-only _KnownCorpusModel double above.
    adapter = KnownCorpusAdapter(
        pretraining_lines={"L1"}, pretraining_drugs={"d1"}, task_signal_in_pretrain="adjacent"
    )
    filtered, profile = filter_leakage(_design(), adapter)
    # (L1, d1) is the only doubly-exposed pair -- always dropped, regardless
    # of task_signal_in_pretrain.
    assert not ((filtered["patient"] == "L1") & (filtered["drug"] == "d1")).any()
    assert len(filtered) == len(_design()) - 1
    assert profile.basis == "measured"
    assert profile.doubly_exposed_frac == 1 / 6
