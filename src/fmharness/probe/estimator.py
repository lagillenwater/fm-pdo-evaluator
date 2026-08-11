"""``Estimator`` protocol: the shared ``fit``/``predict_parts`` contract every
probe head and every wrapped standalone model (bilinear, biomarker) satisfies.

``SimpleProbe`` and ``KernelProbe`` (``probe/simple.py``, ``probe/kernel.py``)
already implement this shape; this module makes it an explicit, checkable
Protocol rather than an implicit convention, per the design spec's decision to
wrap bilinear and biomarker into this same contract rather than leave them
standalone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@runtime_checkable
class Estimator(Protocol):
    def fit(
        self,
        embeddings: ArrayLike,
        drug_ids: Sequence[str],
        y: ArrayLike,
        groups: Sequence[str] | None = None,
    ) -> Estimator: ...

    def predict_parts(
        self, embeddings: ArrayLike, drug_ids: Sequence[str]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
