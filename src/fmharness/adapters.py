"""Viability adapters: map a treated-minus-control transcriptome delta to a
per-sample drug-sensitivity score (higher = more sensitive). Selected with
``build_adapters(methods=...)``; the default is all of ``ALL_METHODS``.

- ``"hallmark"`` -- fixed signature scoring: the direction-signed mean of z-scored
  signature genes (apoptosis / p53 up, proliferation down), averaged across sets.
  No training. MSigDB Hallmark gene sets (Liberzon et al., Cell Systems 2015);
  single-sample scoring in the spirit of ssGSEA (Barbie et al., Nature 2009).
- ``"l1"`` / ``"l2"`` -- CV-tuned penalized linear regression (LassoCV / RidgeCV,
  ``make_penalty``) from the delta to viability, fit on real perturbation->viability
  pairs and applied to the target delta. Replaces the earlier Szalai/WRFEN-XGBoost
  adapters (2026-08-20): both fit dense, weakly-regularized models over thousands of
  genes on a few hundred real training pairs, which transfers poorly under domain
  shift onto a generated delta with very different covariance structure -- an
  overfitting-under-distribution-shift failure mode a properly CV-tuned, optionally
  sparse penalty is less prone to.

The supervised adapters (l1, l2) are trained on a perturbation->viability cohort
(e.g. real L1000 deltas vs GDSC2 AUC) and applied to a held-out delta (e.g.
Stack-generated patient deltas). Each cohort is z-scored by its own per-gene
statistics, so ``predict`` must be given a cohort (many samples), not one row, and
the learned coefficients transfer across the platform gap in standardized units.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

ALL_METHODS: tuple[str, ...] = ("hallmark", "l1", "l2")
PENALTY_NAMES: tuple[str, ...] = ("l2", "l1", "en")  # every penalty make_penalty knows, not
# just the two ALL_METHODS exposes as adapters -- "en" stays available for fmharness.check2's
# own 3-way representation-controlled grid, which imports both from here.


def make_penalty(name: str) -> object:
    """A fresh ALPHA-CV-TUNED penalized model: l2=RidgeCV (efficient GCV), l1=LassoCV, en=
    ElasticNetCV (both inner 3-fold on the training lines). Tuning the penalty per
    representation/cohort makes a comparison model-fair -- a fixed alpha over-/under-
    regularizes some inputs and flips the ranking (Kurilov 2020)."""
    if name == "l2":
        return RidgeCV(alphas=np.logspace(-2, 3, 12))
    if name == "l1":
        return LassoCV(n_alphas=30, cv=3, max_iter=20000, random_state=0)  # type: ignore[arg-type]
    if name == "en":
        return ElasticNetCV(l1_ratio=0.5, n_alphas=30, cv=3, max_iter=20000, random_state=0)  # type: ignore[arg-type]
    raise ValueError(f"unknown penalty {name!r}")


class ViabilityAdapter(Protocol):
    """A predictor of per-sample sensitivity (higher = more sensitive) from a delta."""

    name: str
    citation: str
    supervised: bool

    def fit(self, delta: pd.DataFrame, viability: np.ndarray) -> ViabilityAdapter: ...
    def predict(self, delta: pd.DataFrame) -> np.ndarray: ...


def _zscore(delta: pd.DataFrame) -> pd.DataFrame:
    """Z-score each gene (column) by this cohort's own mean / std."""
    arr = delta.to_numpy(dtype=np.float64)
    sd = arr.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return pd.DataFrame((arr - arr.mean(axis=0)) / sd, index=delta.index, columns=delta.columns)


class SignatureAdapter:
    """Fixed signature scoring (no training); the signed signature means, combined."""

    name = "hallmark"
    supervised = False
    citation = "Liberzon et al., Cell Systems 2015 (Hallmark); Barbie et al., Nature 2009 (ssGSEA)"

    def __init__(self, signatures: dict[str, tuple[tuple[str, ...], int]]) -> None:
        self._sigs = signatures

    def fit(self, delta: pd.DataFrame, viability: np.ndarray) -> SignatureAdapter:
        del delta, viability  # unsupervised
        return self

    def predict(self, delta: pd.DataFrame) -> np.ndarray:
        z = _zscore(delta)
        parts: list[np.ndarray] = []
        for genes, direction in self._sigs.values():
            present = [g for g in genes if g in z.columns]
            if present:
                parts.append(direction * z[present].to_numpy().mean(axis=1))
        if not parts:
            return np.zeros(len(delta), dtype=np.float64)
        return np.asarray(parts, dtype=np.float64).mean(axis=0)


class PenalizedRegressionAdapter:
    """CV-tuned penalized linear regression, delta -> viability (``make_penalty``, Kurilov
    2020). ``penalty`` is ``"l1"`` (LassoCV, sparse) or ``"l2"`` (RidgeCV, dense)."""

    supervised = True
    citation = "Kurilov et al., iScience 2020 (CV-tuned penalized regression)"

    def __init__(self, penalty: str) -> None:
        self._model: Any = make_penalty(penalty)  # raises ValueError on an unknown penalty
        self.name = penalty
        self._genes: list[str] = []

    def fit(self, delta: pd.DataFrame, viability: np.ndarray) -> PenalizedRegressionAdapter:
        self._genes = [str(c) for c in delta.columns]
        self._model.fit(_zscore(delta).to_numpy(), np.asarray(viability, dtype=np.float64))
        return self

    def predict(self, delta: pd.DataFrame) -> np.ndarray:
        z = _zscore(delta.reindex(columns=self._genes, fill_value=0.0)).to_numpy()
        return -np.asarray(self._model.predict(z), dtype=np.float64)  # higher = more sensitive


def build_hallmark_breakout(
    signatures: dict[str, tuple[tuple[str, ...], int]],
) -> list[ViabilityAdapter]:
    """One ``SignatureAdapter`` per individual signature, each named after it.

    The combined ``hallmark`` method (``SignatureAdapter`` over the whole dict) averages
    every signature into one blended score -- on Tahoe, only the proliferation sets
    (E2F/G2M) cleared the gate's random-gene-set control while P53/apoptosis added only
    noise (docs/tahoe_generation_results.md's Gate table), so a blended score can dilute a
    real signal with a null one. This reports each signature's score separately (still
    z-scored / averaged over its own genes only, via ``SignatureAdapter``), so a
    proliferation-only or per-pathway breakdown falls out for free -- pair with
    ``--hallmark-sets`` to restrict which signatures are even computed.
    """
    out: list[ViabilityAdapter] = []
    for name, sig in signatures.items():
        a = SignatureAdapter({name: sig})
        a.name = name
        out.append(a)
    return out


def build_adapters(
    methods: list[str] | None = None,
    *,
    signatures: dict[str, tuple[tuple[str, ...], int]] | None = None,
) -> list[ViabilityAdapter]:
    """Construct the selected viability adapters (default: all of ``ALL_METHODS``).

    ``signatures`` is required when "hallmark" is among the methods.
    """
    chosen = list(ALL_METHODS) if methods is None else methods
    out: list[ViabilityAdapter] = []
    for m in chosen:
        if m == "hallmark":
            if signatures is None:
                raise ValueError("the hallmark adapter requires signatures=")
            out.append(SignatureAdapter(signatures))
        elif m in ("l1", "l2"):
            out.append(PenalizedRegressionAdapter(m))
        else:
            raise ValueError(f"unknown method {m!r}; choose from {ALL_METHODS}")
    return out
