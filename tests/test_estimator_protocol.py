"""Tests that the existing probe heads conform to the (now explicit) Estimator protocol."""

from __future__ import annotations

from fmharness.probe.estimator import Estimator
from fmharness.probe.kernel import KernelProbe
from fmharness.probe.simple import SimpleProbe


def test_simple_probe_satisfies_estimator() -> None:
    assert isinstance(SimpleProbe(), Estimator)


def test_kernel_probe_satisfies_estimator() -> None:
    assert isinstance(KernelProbe(), Estimator)
