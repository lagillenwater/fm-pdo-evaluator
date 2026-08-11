"""Estimator-conforming wrapper around the biomarker rule table.

``biomarker_series`` is ``scripts/biomarker_anchored.py``'s ``_biomarker_series``,
moved here verbatim so it's importable without executing the script's
``__main__`` (which reads real WES/expression files from disk on import). Rules
are pre-specified (a fixed list of drug/gene/kind/direction dicts), not learned,
so ``fit`` only learns the per-drug mean base and, for ``expr`` rules, the
expression mean/SD used to z-score that gene; ``predict_parts`` looks up each
row's matching rule and applies it directly.

``biomarker_series`` normalizes an ``expr`` rule against whatever patients are in
the frame handed to it, which is correct for the script (one call on the whole
cohort) but wrong inside a CV loop: a single-patient test fold has an undefined
SD, and any fold-specific mean/SD both leaks test-fold statistics into the
feature and makes residuals incomparable across folds. ``BiomarkerEstimator``
therefore freezes those statistics on the training patients at ``fit`` time and
applies the frozen values in ``predict_parts``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from fmharness.probe.base import ProbeBase


def _expr_column(bm: dict[str, str], sym2ent: dict[str, int]) -> str | None:
    """Expression-matrix column holding this rule's gene, or None if unmapped."""
    ent = sym2ent.get(bm["gene"])
    return str(ent) if ent is not None else None


def biomarker_series(
    bm: dict[str, str],
    x_log: pd.DataFrame,
    alt: dict[str, dict[str, set[str]]],
    wes: set[str],
    sym2ent: dict[str, int],
) -> pd.Series | None:
    """Per-patient biomarker value (index = patient_id); None if unavailable.

    ``expr`` rules are z-scored against the patients present in ``x_log``, so the
    caller owns the normalization batch. ``scripts/biomarker_anchored.py`` calls
    this once on the full cohort, where that is exactly right; inside a CV loop
    use ``BiomarkerEstimator``, which freezes the statistics at fit time.
    """
    if bm["kind"] == "expr":
        col = _expr_column(bm, sym2ent)
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

    ``fit`` learns two things: the per-drug mean base, and -- for ``expr`` rules
    -- the mean/SD used to z-score that gene, taken from the training patients
    and then frozen. ``predict_parts`` applies the frozen values, so a residual
    means the same thing in every fold and no test-fold statistic leaks into the
    feature. This is what makes the estimator safe under leave-one-out, the CV
    scheme ``SoragniViability`` recommends.
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
        # index into self.biomarkers -> (mean, sd) frozen on the training patients
        self._expr_stats: dict[int, tuple[float, float]] = {}

    @staticmethod
    def _unique_patients(embeddings: pd.DataFrame) -> pd.DataFrame:
        """One row per patient: a design frame repeats a patient once per drug."""
        unique = embeddings[~embeddings.index.duplicated(keep="first")]
        assert isinstance(unique, pd.DataFrame)
        return unique

    def _freeze_expr_stats(self, embeddings: pd.DataFrame) -> dict[int, tuple[float, float]]:
        """Mean/SD of each ``expr`` rule's gene over the training patients.

        A rule whose gene is absent, or whose training SD is 0 or non-finite
        (e.g. one training patient), carries no usable spread and is simply
        omitted -- ``predict_parts`` then leaves its rows at residual 0 rather
        than dividing by zero and emitting NaN/Inf.
        """
        unique = self._unique_patients(embeddings)
        stats: dict[int, tuple[float, float]] = {}
        # A loop over the fixed, pre-specified rule table (a handful of dicts),
        # not over samples or genes -- the per-patient math below is vectorized.
        for i, bm in enumerate(self.biomarkers):
            if bm["kind"] != "expr":
                continue
            col = _expr_column(bm, self.sym2entrez)
            if col is None or col not in unique.columns:
                continue
            v = unique[col].astype(np.float64)
            sd = float(v.std())
            if not np.isfinite(sd) or sd == 0.0:
                continue
            stats[i] = (float(v.mean()), sd)
        return stats

    def _values(self, index: int, bm: dict[str, str], unique: pd.DataFrame) -> pd.Series | None:
        """Per-patient value for one rule, using the frozen stats for ``expr``."""
        if bm["kind"] != "expr":
            # Binary alteration indicator over self.wes_patients -- batch-independent.
            return biomarker_series(
                bm, unique, self.alterations, self.wes_patients, self.sym2entrez
            )
        frozen = self._expr_stats.get(index)
        col = _expr_column(bm, self.sym2entrez)
        if frozen is None or col is None or col not in unique.columns:
            return None
        mean, sd = frozen
        return (unique[col].astype(np.float64) - mean) / sd

    def fit(
        self,
        embeddings: ArrayLike,
        drug_ids: Sequence[str],
        y: ArrayLike,
        groups: Sequence[str] | None = None,
    ) -> BiomarkerEstimator:
        if not isinstance(embeddings, pd.DataFrame):
            raise TypeError(
                "BiomarkerEstimator.fit requires a DataFrame indexed by patient_id, "
                "not a bare array -- biomarker lookup keys off patient identity"
            )
        # Only the per-drug-mean bookkeeping from _prepare_fit is used here;
        # the "embeddings" arg (a features DataFrame) never reaches PCA/NMF
        # since n_components=0 forces k=0 in _prepare_fit's reduction-rank calc.
        # It is read directly, below, for the frozen expression statistics.
        placeholder = np.zeros((len(np.asarray(y, dtype=np.float64)), 1))
        self._prepare_fit(placeholder, drug_ids, y)
        self._expr_stats = self._freeze_expr_stats(embeddings)
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
        unique = self._unique_patients(embeddings)
        row_patients = pd.Index(embeddings.index)
        drug_arr = np.asarray(drug_ids, dtype=object)
        # Loop over the fixed rule table; every per-row operation inside is vectorized.
        for i, bm in enumerate(self.biomarkers):
            rows = np.flatnonzero(drug_arr == bm["drug"])
            if rows.size == 0:
                continue
            b = self._values(i, bm, unique)
            if b is None:
                continue
            sign = -1.0 if bm["direction"] == "sensitize" else 1.0
            matched = b.reindex(row_patients[rows]).to_numpy(dtype=np.float64)
            # Patients absent from the rule's index (and any non-finite value)
            # keep residual 0, matching the drug-mean fallback elsewhere.
            residual[rows] = sign * np.where(np.isfinite(matched), matched, 0.0)
        return base, residual
