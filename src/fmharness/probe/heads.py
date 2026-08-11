"""Probe-head registry: build a probe factory by name.

The transfer and per-patient scripts already take a zero-arg ``probe_factory``
(``grouped_cv_predict``, ``transfer_predict``). ``make_head`` returns such a
factory for a named head so a ``--head`` flag swaps the linear ridge for the
nonlinear kernel ridge without touching the harness. Both heads share the
``fit``/``predict_parts`` contract and the same PCA/NMF reduction, so the only
thing that changes is the residual model -- the comparison stays apples-to-apples.

The bilinear and biomarker models DO satisfy the same contract --
``BilinearEstimator`` (``probe/bilinear_head.py``) and ``BiomarkerEstimator``
(``probe/biomarker_head.py``) both implement the ``Estimator`` protocol
(``probe/estimator.py``) -- but are intentionally absent from ``HEADS``/
``make_head`` here. The registry's shape is a zero-arg factory configured by one
uniform kwarg set (``n_components``, ``std_floor``, ``reducer``, ``per_drug``),
and neither model fits it: bilinear additionally requires a drug-fingerprint
lookup, biomarker a pre-specified rule table plus WES alteration calls and a
gene-symbol map, none of which a ``--head`` flag can supply. Wiring them into a
by-name factory is driver-integration work, explicitly out of scope for this
plan; until then they are constructed directly and contribute their rows to the
head-invariance table from their own scripts.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from fmharness.probe.kernel import KernelProbe
from fmharness.probe.simple import SimpleProbe

__all__ = ["HEADS", "make_head"]

HEADS = ("linear", "kernel")


def make_head(
    name: str,
    *,
    n_components: int = 10,
    std_floor: float = 0.0,
    reducer: str = "pca",
    per_drug: bool = True,
) -> Callable[[], SimpleProbe | KernelProbe]:
    """Return a zero-arg factory for the named head (``"linear"`` or ``"kernel"``)."""
    kwargs = dict(
        n_components=n_components,
        std_floor=std_floor,
        reducer=reducer,
        per_drug=per_drug,
    )
    if name == "linear":
        return partial(SimpleProbe, **kwargs)
    if name == "kernel":
        return partial(KernelProbe, **kwargs)
    raise ValueError(f"unknown head {name!r}; choose from {HEADS}")
