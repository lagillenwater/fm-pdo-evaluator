"""Model-level capability protocols: ``Encoder`` and ``Generator``.

Split rather than a single protocol with optional methods, so a model's type
signature says exactly what it can do -- no ``None``-checking, and
``isinstance(model, Generator)`` is how the harness detects capability.
``Encoder`` is a strict subset of the existing ``ModelAdapter``
(``models/adapter.py``): every current adapter already satisfies it, with no
changes to those classes. ``Generator`` is new -- it predicts a transcriptome
profile, distinct from ``ModelAdapter.predict_native``, which predicts a
scalar response value directly.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Protocol, runtime_checkable

import numpy as np
from anndata import AnnData

from fmharness.models.adapter import as_dense_f32, seeded_projection
from fmharness.schema import ModelMetadata


class PerturbationNotInContext(ValueError):
    """Raised by ``Generator.generate`` when the perturbation is not representable.

    A model must refuse loudly rather than silently extrapolate to something
    that looks like a real prediction but reflects the nearest thing it has
    actually seen -- confident-looking noise, worse than a null result.
    """


@runtime_checkable
class Encoder(Protocol):
    """Produces a per-sample representation vector."""

    def embed(self, adata: AnnData) -> np.ndarray:
        """(n_obs, embedding_dim), row-aligned to adata.obs_names. Deterministic."""
        ...

    def metadata(self) -> ModelMetadata:
        """Pretraining provenance for the leakage scan."""
        ...

    def version(self) -> str:
        """Stable identifier, e.g. ``encoder@v1.0.0``. Embedded in PredictionRecord."""
        ...


@runtime_checkable
class Generator(Protocol):
    """Produces a predicted post-perturbation profile."""

    def generate(self, baseline: AnnData, perturbation: str) -> AnnData:
        """Predicted profile for each row of ``baseline`` under ``perturbation``.

        Same obs/var contract as ``baseline``; full profile or delta, declared
        in ``metadata()``. Raises ``PerturbationNotInContext`` if
        ``perturbation`` cannot be represented.
        """
        ...

    def context_coverage(self, perturbations: Iterable[str]) -> set[str]:
        """Subset of ``perturbations`` this model can actually represent.

        Computed once per model, before generation runs -- not discovered as
        failures partway through an expensive GPU job.
        """
        ...

    def metadata(self) -> ModelMetadata:
        """Pretraining provenance for the leakage scan."""
        ...

    def version(self) -> str:
        """Stable identifier, e.g. ``generator@v1.0.0``. Embedded in PredictionRecord."""
        ...


class MockGenerator:
    """Deterministic stand-in Generator for tests.

    ``known_perturbations`` is a fixed, small set -- ``context_coverage()``
    and ``generate()`` agree on what's representable, mirroring how a real
    model's context corpus bounds what it can generate for. Also implements
    ``embed`` (a seeded random projection, same recipe as ``MockAdapter``) so
    it satisfies both ``Generator`` and ``Encoder`` -- the dual-capability
    case ``scFoundation``/Stack-aligned will eventually occupy.
    """

    def __init__(
        self,
        known_perturbations: frozenset[str] = frozenset({"drugA", "drugB"}),
        embedding_dim: int = 8,
        seed: int = 0,
    ) -> None:
        self.known_perturbations = known_perturbations
        self.embedding_dim = embedding_dim
        self.seed = seed

    def version(self) -> str:
        return f"mock_generator@v1.0.0-s{self.seed}"

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            pretraining_corpus="none",
            pretraining_cutoff_date=date(1970, 1, 1),
            task_signal_in_pretrain="none",
        )

    def embed(self, adata: AnnData) -> np.ndarray:
        x = as_dense_f32(adata)
        return seeded_projection(x, self.seed, self.embedding_dim)

    def context_coverage(self, perturbations: Iterable[str]) -> set[str]:
        return {p for p in perturbations if p in self.known_perturbations}

    def generate(self, baseline: AnnData, perturbation: str) -> AnnData:
        if perturbation not in self.known_perturbations:
            raise PerturbationNotInContext(
                f"{perturbation!r} not in this model's context: {sorted(self.known_perturbations)}"
            )
        rng = np.random.default_rng(self.seed)
        x = as_dense_f32(baseline)
        shift = rng.standard_normal(x.shape[1]).astype(np.float32) * 0.1
        out = baseline.copy()
        out.X = x + shift
        return out
