"""Estimator-conforming wrapper around ``bilinear_features``.

``bilinear.py``'s ``AUC(s, d) = ridge([z_s, g_d, z_s (x) g_d])`` needs a drug
fingerprint lookup that a plain ``embeddings`` array can't carry, so this
wraps it: fingerprints are injected at construction, and ``fit``/``predict_parts``
match the same signature every other probe head uses. Drugs missing a
fingerprint fall back to the drug-mean base with zero residual, matching
``SimpleProbe``'s graceful-degradation contract -- an uninformative model
reduces to the drug mean rather than injecting noise.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from fmharness.bilinear import bilinear_features
from fmharness.probe.base import ALPHAS, ProbeBase


class BilinearEstimator(ProbeBase):
    """Per-drug mean + ridge on ``[z, g, z (x) g]``, reusing ProbeBase's PCA
    reduction of ``z`` and per-drug-mean bookkeeping."""

    def __init__(
        self,
        drug_fingerprints: dict[str, NDArray[np.float64]],
        *,
        n_components: int = 10,
        alphas: Sequence[float] = ALPHAS,
        seed: int = 0,
    ) -> None:
        super().__init__(n_components=n_components, seed=seed)
        self.drug_fingerprints = drug_fingerprints
        self.alphas = tuple(alphas)
        self._reduce: Pipeline | None = None
        self._ridge: RidgeCV | None = None

    def _stack_fingerprints(
        self, drug_ids: Sequence[str]
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        dim = 0 if not self.drug_fingerprints else len(next(iter(self.drug_fingerprints.values())))
        g = np.zeros((len(drug_ids), dim), dtype=np.float64)
        known = np.zeros(len(drug_ids), dtype=bool)
        for i, d in enumerate(drug_ids):
            fp = self.drug_fingerprints.get(d)
            if fp is not None:
                g[i] = fp
                known[i] = True
        return g, known

    def fit(
        self,
        embeddings: ArrayLike,
        drug_ids: Sequence[str],
        y: ArrayLike,
        groups: Sequence[str] | None = None,
    ) -> BilinearEstimator:
        emb, _drug_arr, residual, k = self._prepare_fit(embeddings, drug_ids, y)
        g, known = self._stack_fingerprints(drug_ids)
        self._reduce = self._ridge = None
        # _prepare_fit caps k against ALL rows, but the reducer below is fit on
        # emb[known] alone, so the rank ceiling is that subset's size. Without
        # this recap, sparse fingerprint coverage (a few known drugs in a large
        # panel) asks PCA for more components than the subset can support.
        k = min(k, max(0, int(known.sum()) - 1))
        if known.any() and k > 0:
            self._reduce = Pipeline(self._reducer_steps(k))
            z = self._reduce.fit_transform(emb[known])
            feats = bilinear_features(z, g[known])
            cv = None
            if groups is not None:
                grp = np.asarray(groups)[known]
                n_g = len(np.unique(grp))
                if n_g >= 2:
                    cv = list(
                        GroupKFold(n_splits=min(5, n_g)).split(
                            feats, residual[known], groups=grp
                        )
                    )
            self._ridge = RidgeCV(alphas=np.asarray(self.alphas, dtype=np.float64), cv=cv)
            self._ridge.fit(feats, residual[known])
        return self

    def predict_parts(
        self, embeddings: ArrayLike, drug_ids: Sequence[str]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if not self._drug_means:
            raise RuntimeError("estimator is not fitted; call fit() before predict_parts()")
        base = self._base(drug_ids)
        emb = np.asarray(embeddings, dtype=np.float64)
        residual = np.zeros(len(drug_ids), dtype=np.float64)
        if self._reduce is not None and self._ridge is not None:
            g, known = self._stack_fingerprints(drug_ids)
            if known.any():
                z = self._reduce.transform(emb[known])
                feats = bilinear_features(z, g[known])
                residual[known] = self._ridge.predict(feats)
        return base, residual
