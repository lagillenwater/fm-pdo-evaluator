"""Known-answer tests for the derangement-based exact permutation null (SPEC rule 4).

This is rung 0's final verification step (docs/tasks/rung0-replicate-ceiling/design.md,
decision history), added to carry -- by construction -- the dependence the bootstrapped
p-values in `scripts/delta_reproducibility.py` had to assume away: the stratified null draws
in that script reuse the same half-profiles across many mismatched-pair comparisons, so they
are not an exchangeable i.i.d. pool even though the bootstrap treats them as one
(docs/tasks/rung0-replicate-ceiling/verification.md, "Write-up caveat"). Derangements of the
pairing preserve every row exactly once while eliminating the possibility of a matched pair,
so the resulting null distribution of the mean carries the real dependence structure rather
than an assumed one. Every test here runs the REAL functions from
`scripts/derangement_null.py` (which itself reuses, not reimplements, the measurement core in
`scripts/delta_reproducibility.py`) on synthetic replicate pools with planted, known answers.
"""

# pandas ships no PEP-561 type stubs in this environment; under strict mode that turns every
# pandas call site into a cascade of reportUnknown* noise about *pandas'* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# `_write_fixture_pool` lives in tests/test_rung0_controls.py; the leading underscore marks it
# as private to that module, not a boundary this sibling test file needs to respect -- both
# test the same rung 0 task, and re-declaring the fixture builder here would be the duplication
# PROCESS §3 tells tests not to reintroduce. The suppression covers importing a sibling test
# module's private helper for exactly that reason.
from tests.test_rung0_controls import _write_fixture_pool  # pyright: ignore[reportPrivateUsage]

pytestmark = pytest.mark.known_answer

_SPEC = importlib.util.spec_from_file_location(
    "derangement_null",
    Path(__file__).resolve().parents[1] / "scripts" / "derangement_null.py",
)
assert _SPEC is not None and _SPEC.loader is not None
dn = importlib.util.module_from_spec(_SPEC)
sys.modules["derangement_null"] = dn
_SPEC.loader.exec_module(dn)


def test_sample_derangement_has_no_fixed_points() -> None:
    """A rejection-sampled derangement has zero fixed points, at several sizes and seeds."""
    for n in (2, 3, 5, 10, 50):
        for seed in range(5):
            rng = np.random.default_rng(seed * 1000 + n)
            sigma = dn.sample_derangement(rng, n)
            assert sigma.shape == (n,)
            assert sorted(sigma.tolist()) == list(range(n)), f"not a permutation: {sigma}"
            assert not np.any(sigma == np.arange(n)), f"n={n} seed={seed}: fixed point in {sigma}"


def test_planted_signal_pool_clears_the_derangement_null(tmp_path: Path) -> None:
    # signal_sd = noise_sd = 1, 8 plates -> expected split-half r ~ 0.8 (same planted shape as
    # test_score_positive_planted_reliability_is_recovered in tests/test_rung0_controls.py).
    # Mismatching the pairing via a derangement destroys that pair-specific signal entirely
    # (drug_sd = 0.0, the default -- no drug-shared component survives a mismatch either), so
    # the derangement null should sit near zero while the observed mean sits near 0.8.
    path = _write_fixture_pool(tmp_path, signal_sd=1.0, noise_sd=1.0)
    de, _ = dn.dr.build_split_half_frame(
        [str(path)], ["D0", "D1", "D2"], None, tmp_path / "duck", memory_limit="2GB"
    )
    de = de.dropna(subset=["lfc0", "lfc1"])
    r, piv0, piv1 = dn.dr.score_split_half(de, set(de["gene_name"].unique()))
    summary, _ = dn.derangement_null(piv0, piv1, r, min_genes=50, n_perm=99, seed=0)
    assert summary["p_exact"] < 0.05, (
        f"planted signal must clear the null, got p={summary['p_exact']}"
    )
    assert summary["observed_mean"] > summary["perm_mean_mean"] + 0.1, (
        f"observed {summary['observed_mean']} not far above perm null mean "
        f"{summary['perm_mean_mean']}"
    )


def test_zero_signal_pool_is_not_significant(tmp_path: Path) -> None:
    # No planted signal at all: matched and mismatched pairs are drawn from the same
    # generative process, so the derangement null and the observed mean should sit at the
    # same (near-zero) place. Deterministic under the pinned seed; if this ever flips the
    # failure message below records the actual p rather than a bare assertion.
    path = _write_fixture_pool(tmp_path, signal_sd=0.0, noise_sd=1.0)
    de, _ = dn.dr.build_split_half_frame(
        [str(path)], ["D0", "D1", "D2"], None, tmp_path / "duck", memory_limit="2GB"
    )
    de = de.dropna(subset=["lfc0", "lfc1"])
    r, piv0, piv1 = dn.dr.score_split_half(de, set(de["gene_name"].unique()))
    summary, _ = dn.derangement_null(piv0, piv1, r, min_genes=50, n_perm=200, seed=0)
    assert summary["p_exact"] > 0.01, (
        f"zero-signal pool must not clear the derangement null, got p={summary['p_exact']} "
        f"(observed_mean={summary['observed_mean']}, perm_mean_mean={summary['perm_mean_mean']})"
    )


def test_derangement_null_rejects_too_few_pairs_or_too_few_permutations() -> None:
    """A single finite pair has no derangement (rejection sampling can never satisfy "no
    fixed points" for n=1), and n_perm=1 gives a null distribution with no spread -- both
    are degenerate inputs the function should refuse before entering the permutation loop,
    rather than failing obscurely inside `sample_derangement` or downstream variance math."""
    import pandas as pd

    one_pair = pd.DataFrame([[1.0, 2.0, 3.0]], index=["a"])
    with pytest.raises(ValueError, match="at least 2"):
        dn.derangement_null(one_pair, one_pair, np.array([0.9]), min_genes=1, n_perm=99, seed=0)

    two_pairs = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["a", "b"])
    with pytest.raises(ValueError, match="n_perm"):
        dn.derangement_null(
            two_pairs, two_pairs, np.array([0.9, 0.8]), min_genes=1, n_perm=1, seed=0
        )


def test_design_effect_sits_in_a_sane_band_on_near_independent_rows(tmp_path: Path) -> None:
    """The design effect the exchangeable-pool bootstrap ignores, measured on data with no
    shared drug or line component (drug_sd = 0.0, the default -- rows are as close to
    independent as this fixture can make them).

    This is a known answer for "no pathological inflation," not a precise theoretical
    prediction: the mean-r statistic still shares gene-panel structure across rows through the
    finite gene count, and only 12 (line, drug) pairs are available (4 lines x 3 drugs, the
    fixture default), so the derangement-null variance is itself estimated noisily at
    n_perm=200. [0.2, 5.0] is a loose band that would catch an order-of-magnitude inflation
    factor -- the failure mode the write-up caveat in verification.md was worried about --
    while tolerating ordinary Monte Carlo noise at this sample size.
    """
    path = _write_fixture_pool(tmp_path, signal_sd=0.0, noise_sd=1.0)
    de, _ = dn.dr.build_split_half_frame(
        [str(path)], ["D0", "D1", "D2"], None, tmp_path / "duck", memory_limit="2GB"
    )
    de = de.dropna(subset=["lfc0", "lfc1"])
    r, piv0, piv1 = dn.dr.score_split_half(de, set(de["gene_name"].unique()))
    summary, _ = dn.derangement_null(piv0, piv1, r, min_genes=50, n_perm=200, seed=0)
    assert 0.2 <= summary["design_effect"] <= 5.0, (
        f"design effect {summary['design_effect']} outside the sane band [0.2, 5.0]"
    )
