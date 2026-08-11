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
    """Fit a response model, then predict it as ``(base, residual)``.

    ``fit(embeddings, drug_ids, y, groups)`` learns the model; ``groups`` (patient
    ids, when supplied) keeps any inner CV from splitting one patient's rows
    across folds. ``predict_parts(embeddings, drug_ids)`` returns two arrays, one
    per row:

    - ``base``: the per-drug mean ``a_d`` learned at fit (the global mean for a
      drug unseen at fit), i.e. what you would predict knowing only the drug.
    - ``residual``: the embedding-driven term added on top, so the full
      prediction is ``base + residual``.

    They are returned separately, rather than pre-summed, because the scientific
    question is whether the *representation* carries patient-level signal, and
    the drug mean is a large, representation-independent term that would swamp
    it. Metrics like ``interaction_rho`` score the residual alone; handing them
    ``base + residual`` would let the leave-one-out drug-mean baseline
    contaminate a Stack-vs-expression comparison that is supposed to isolate the
    embedding. Keeping the split in the contract also enforces graceful
    degradation: an uninformative representation shrinks ``residual`` toward 0
    and the estimator reduces to the drug-mean baseline rather than injecting
    noise -- which is what makes a null result a real null.
    """

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
