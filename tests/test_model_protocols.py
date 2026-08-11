"""Tests for the Encoder/Generator model protocols."""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from fmharness.model_protocols import Encoder, Generator, MockGenerator, PerturbationNotInContext
from fmharness.models import LinearBaselineAdapter, MockAdapter


def test_existing_adapters_satisfy_encoder() -> None:
    # Encoder is a strict subset of ModelAdapter's surface -- every current
    # adapter already implements it, with no code changes to those classes.
    assert isinstance(MockAdapter(), Encoder)
    assert isinstance(LinearBaselineAdapter(), Encoder)


def test_mock_generator_satisfies_generator_and_encoder() -> None:
    gen = MockGenerator()
    assert isinstance(gen, Generator)
    assert isinstance(gen, Encoder)  # has embed too, so it's both


def test_mock_generator_context_coverage_matches_generate() -> None:
    gen = MockGenerator(known_perturbations=frozenset({"drugA", "drugB"}))
    assert gen.context_coverage(["drugA", "drugC", "drugB"]) == {"drugA", "drugB"}


def test_mock_generator_generate_is_deterministic_and_row_aligned() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 5)).astype(np.float32)
    baseline = AnnData(X=x)
    baseline.obs_names = ["s0", "s1", "s2"]

    gen = MockGenerator(known_perturbations=frozenset({"drugA"}), seed=0)
    out1 = gen.generate(baseline, "drugA")
    out2 = gen.generate(baseline, "drugA")
    assert out1.obs_names.tolist() == baseline.obs_names.tolist()
    np.testing.assert_array_equal(np.asarray(out1.X), np.asarray(out2.X))


def test_mock_generator_raises_on_unknown_perturbation() -> None:
    rng = np.random.default_rng(0)
    baseline = AnnData(X=rng.normal(size=(2, 4)).astype(np.float32))
    gen = MockGenerator(known_perturbations=frozenset({"drugA"}))
    with pytest.raises(PerturbationNotInContext):
        gen.generate(baseline, "drugZ")
