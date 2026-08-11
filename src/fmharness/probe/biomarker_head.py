"""Estimator-conforming wrapper around the biomarker rule table.

``biomarker_series`` is ``scripts/biomarker_anchored.py``'s ``_biomarker_series``,
moved here verbatim so it's importable without executing the script's
``__main__`` (which reads real WES/expression files from disk on import). Rules
are pre-specified (a fixed list of drug/gene/kind/direction dicts), not learned,
so ``fit`` only learns the per-drug mean base; ``predict_parts`` looks up each
row's matching rule and applies it directly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from fmharness.probe.base import ProbeBase


def biomarker_series(
    bm: dict[str, str],
    x_log: pd.DataFrame,
    alt: dict[str, dict[str, set[str]]],
    wes: set[str],
    sym2ent: dict[str, int],
) -> pd.Series | None:
    """Per-patient biomarker value (index = patient_id); None if unavailable."""
    if bm["kind"] == "expr":
        ent = sym2ent.get(bm["gene"])
        col = str(ent) if ent is not None else None
        if col is None or col not in x_log.columns:
            return None
        v = x_log[col]
        return (v - v.mean()) / (v.std() or 1.0)
    positive = alt[bm["kind"]].get(bm["gene"], set())
    return pd.Series({p: float(p in positive) for p in sorted(wes)})


class BiomarkerEstimator(ProbeBase):
    """Wraps the biomarker rule table into the Estimator contract.

    ``features`` must be a DataFrame indexed by patient_id (not a bare ndarray)
    since biomarker lookup keys off patient identity, not row position -- a
    deliberate deviation from the plain-embedding convention other heads use,
    documented here rather than forced into an artificial fit. This remains a
    valid Estimator: the Protocol checks method presence structurally, not
    parameter types.
    """

    def __init__(
        self,
        biomarkers: list[dict[str, str]],
        alterations: dict[str, dict[str, set[str]]],
        wes_patients: set[str],
        sym2entrez: dict[str, int],
        seed: int = 0,
    ) -> None:
        super().__init__(n_components=0, seed=seed)
        self.biomarkers = biomarkers
        self.alterations = alterations
        self.wes_patients = wes_patients
        self.sym2entrez = sym2entrez

    def fit(
        self,
        embeddings: ArrayLike,
        drug_ids: Sequence[str],
        y: ArrayLike,
        groups: Sequence[str] | None = None,
    ) -> BiomarkerEstimator:
        # Only the per-drug-mean bookkeeping from _prepare_fit is used here;
        # the "embeddings" arg (a features DataFrame) never reaches PCA/NMF
        # since n_components=0 forces k=0 in _prepare_fit's reduction-rank calc.
        placeholder = np.zeros((len(np.asarray(y, dtype=np.float64)), 1))
        self._prepare_fit(placeholder, drug_ids, y)
        return self

    def predict_parts(
        self, embeddings: ArrayLike, drug_ids: Sequence[str]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if not self._drug_means:
            raise RuntimeError("estimator is not fitted; call fit() before predict_parts()")
        if not isinstance(embeddings, pd.DataFrame):
            raise TypeError(
                "BiomarkerEstimator.predict_parts requires a DataFrame indexed by "
                "patient_id, not a bare array -- biomarker lookup keys off patient identity"
            )
        base = self._base(drug_ids)
        residual = np.zeros(len(drug_ids), dtype=np.float64)
        for bm in self.biomarkers:
            rows = [i for i, d in enumerate(drug_ids) if d == bm["drug"]]
            if not rows:
                continue
            # Extract unique patients and subset embeddings to avoid index issues
            embeddings_unique = embeddings[~embeddings.index.duplicated(keep="first")]
            assert isinstance(embeddings_unique, pd.DataFrame)
            b = biomarker_series(
                bm, embeddings_unique, self.alterations, self.wes_patients, self.sym2entrez
            )
            if b is None:
                continue
            sign = -1.0 if bm["direction"] == "sensitize" else 1.0
            for i in rows:
                patient = embeddings.index[i]
                if patient in b.index:
                    residual[i] = sign * float(b.loc[patient])
        return base, residual
