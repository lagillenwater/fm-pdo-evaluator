"""Tests for the modular viability adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmharness.adapters import (
    ALL_METHODS,
    PenalizedRegressionAdapter,
    SignatureAdapter,
    build_adapters,
    build_hallmark_breakout,
)

SIGS: dict[str, tuple[tuple[str, ...], int]] = {"death": (("A", "B", "C"), 1)}


def test_build_default_is_all_methods() -> None:
    adapters = build_adapters(signatures=SIGS)
    assert [a.name for a in adapters] == list(ALL_METHODS)
    assert all(a.citation for a in adapters)  # every method carries a citation


def test_build_subset_selects_methods() -> None:
    adapters = build_adapters(["l1"], signatures=SIGS)
    assert len(adapters) == 1
    assert adapters[0].name == "l1" and adapters[0].supervised


def test_hallmark_requires_signatures() -> None:
    with pytest.raises(ValueError, match="signatures"):
        build_adapters(["hallmark"])


def test_hallmark_scores_induced_death_most_sensitive() -> None:
    cols = ["A", "B", "C", "N1", "N2"]
    rng = np.random.default_rng(0)
    delta = pd.DataFrame(
        rng.normal(0, 0.1, (6, 5)),
        columns=pd.Index(cols),
        index=pd.Index([f"s{i}" for i in range(6)]),
    )
    delta.loc["s0", ["A", "B", "C"]] += 5.0  # strong death induction in s0
    scores = SignatureAdapter(SIGS).predict(delta)
    assert int(np.argmax(scores)) == 0


@pytest.mark.parametrize("penalty", ["l1", "l2"])
def test_penalized_regression_transfers_direction_to_heldout_cohort(penalty: str) -> None:
    rng = np.random.default_rng(1)
    cols = pd.Index(list("abcd"))
    x_tr = pd.DataFrame(rng.normal(size=(80, 4)), columns=cols)
    via_tr = x_tr["a"].to_numpy() * 2 + rng.normal(0, 0.1, 80)  # viability tracks gene a
    adapter = PenalizedRegressionAdapter(penalty).fit(x_tr, via_tr)
    assert adapter.name == penalty and adapter.supervised
    x_te = pd.DataFrame(rng.normal(size=(40, 4)), columns=cols)
    via_te = x_te["a"].to_numpy() * 2
    sens = adapter.predict(x_te)  # higher = more sensitive = lower viability
    assert float(np.corrcoef(sens, via_te)[0, 1]) < -0.5


def test_penalized_regression_rejects_unknown_penalty() -> None:
    with pytest.raises(ValueError, match="unknown penalty"):
        PenalizedRegressionAdapter("bogus")


def test_build_hallmark_breakout_returns_one_adapter_per_signature() -> None:
    # two disjoint-gene signatures, each with its own perturbed sample; the combined
    # SignatureAdapter would average both into one number per sample, hiding which
    # signature actually moved -- per-signature adapters must not cross-talk.
    sigs: dict[str, tuple[tuple[str, ...], int]] = {
        "up": (("A", "B"), 1),
        "down": (("C", "D"), -1),
    }
    cols = ["A", "B", "C", "D"]
    rng = np.random.default_rng(2)
    delta = pd.DataFrame(
        rng.normal(0, 0.1, (6, 4)), columns=pd.Index(cols), index=pd.Index([f"s{i}" for i in range(6)])
    )
    delta.loc["s0", ["A", "B"]] += 5.0  # only "up"'s genes move (up) in s0
    delta.loc["s1", ["C", "D"]] -= 5.0  # only "down"'s genes move (down) in s1

    adapters = build_hallmark_breakout(sigs)
    names = {a.name for a in adapters}
    assert names == {"up", "down"}
    assert all(not a.supervised and a.citation for a in adapters)

    scores = {a.name: a.predict(delta) for a in adapters}
    assert int(np.argmax(scores["up"])) == 0  # "up" fires on s0, not s1
    assert int(np.argmax(scores["down"])) == 1  # "down" fires on s1, not s0 -- no cross-talk
