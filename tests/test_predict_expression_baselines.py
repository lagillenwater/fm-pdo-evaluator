"""Tests for scripts/predict_expression_baselines.py's method-support restriction.

Mirrors tests/test_deltas.py's restrict_common_support tests and
tests/test_baselines_soragni.py's sibling fix: control/mean broadcast every mapped
drug to every organoid, but pca/nmf additionally require >= min_lines profiled L1000
cell lines per drug (the conditional() helper in predict_expression_baselines.py) and
silently drop low-coverage drugs -- so control/mean's key is a strict superset of
pca/nmf's whenever a drug has too few profiled lines. Scoring each method's native key
against `design` independently would match the four methods to different (patient,
drug) pairs in the same printed table; restrict_common_support (wired into main()
right before the scoring loop) must narrow every method to the pairs they all share
AND that carry a real label, so the printed "n" is identical and correct across rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmharness.deltas import restrict_common_support
from predict_expression_baselines import broadcast


def test_methods_restricted_to_common_support_when_pca_nmf_drop_low_coverage_drugs() -> None:
    genes = pd.Index(["g1", "g2"])
    orgs = ["p1", "p2"]
    # drug "lowcov" is mapped for control/mean (every drug is broadcast) but dropped by
    # pca/nmf because it had < min_lines profiled L1000 cell lines (the conditional()
    # guard) -- the exact scenario the bug report describes.
    control_delta, control_key = broadcast(
        {"d1": np.zeros(2), "lowcov": np.zeros(2)}, orgs, genes
    )
    mean_delta, mean_key = broadcast(
        {"d1": np.array([1.0, 1.0]), "lowcov": np.array([2.0, 2.0])}, orgs, genes
    )
    # pca/nmf only produced rows for d1 (lowcov was skipped by the min_lines guard).
    pca_delta, pca_key = broadcast({"d1": np.array([0.5, 0.5])}, orgs, genes)
    nmf_delta, nmf_key = broadcast({"d1": np.array([0.3, 0.3])}, orgs, genes)

    methods = {
        "control": (control_delta, control_key),
        "mean": (mean_delta, mean_key),
        "pca": (pca_delta, pca_key),
        "nmf": (nmf_delta, nmf_key),
    }
    # design labels both drugs for both patients -- "lowcov" is a real, scoreable drug;
    # it is only pca/nmf's own coverage gap that excludes it, not a labeling gap.
    design = pd.DataFrame(
        {
            "patient": ["p1", "p1", "p2", "p2"],
            "drug": ["d1", "lowcov", "d1", "lowcov"],
            "y": [0.1, 0.2, 0.3, 0.4],
        }
    )

    native_n = {name: len(key) for name, (_, key) in methods.items()}
    assert native_n == {"control": 4, "mean": 4, "pca": 2, "nmf": 2}  # the pre-fix mismatch

    restricted = restrict_common_support(methods, design)

    # every method is narrowed to exactly d1 x {p1, p2} -- lowcov is dropped everywhere,
    # not just from pca/nmf, so the four rows are scored on IDENTICAL support.
    want_pairs = {("p1", "d1"), ("p2", "d1")}
    ns = set()
    for name, (delta, key) in restricted.items():
        pairs = set(zip(key["patient"], key["drug"], strict=True))
        assert pairs == want_pairs, name
        assert len(delta) == len(key) == 2, name
        ns.add(len(key))
    assert ns == {2}  # r["n"] = len(key) is now identical across every method


def test_broadcast_pairs_every_drug_with_every_organoid() -> None:
    genes = pd.Index(["g1", "g2", "g3"])
    delta, key = broadcast({"d1": np.array([1.0, 2.0, 3.0])}, ["p1", "p2"], genes)
    assert list(delta.columns) == list(genes)
    assert list(zip(key["patient"], key["drug"], strict=True)) == [("p1", "d1"), ("p2", "d1")]
    assert np.allclose(delta.to_numpy(), [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
