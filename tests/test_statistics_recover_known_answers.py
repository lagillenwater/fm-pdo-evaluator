"""Known-answer validation for every statistic this harness reports.

A statistic that has never been run against a known answer is not evidence. Each test here
plants an answer and requires the statistic to recover it, and plants no signal and requires it
to return null. This file exists because a null test written for the rung-0 ceiling compared an
aggregate against a distribution of single draws -- the observed MEDIAN of ~1,336 correlations
against the spread of individual null draws -- and reported that a reproducible ceiling had
failed. The comparison was between two different KINDS of quantity, and no amount of reading
the code caught it; running it against a known answer would have, immediately.

The recurring error is aggregate-vs-per-item, so every test below pins the units explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats


def _median_pvalue(observed_values: np.ndarray, null_draws: np.ndarray, n_boot: int = 2000,
                   seed: int = 0) -> float:
    """p for an observed MEDIAN against the bootstrapped sampling distribution of the null median.

    This is the corrected form. The wrong form -- ``mean(null_draws >= median(observed))`` --
    compares an aggregate to single draws and is inflated by roughly sqrt(n).
    """
    rng = np.random.default_rng(seed)
    med = float(np.median(observed_values))
    n = observed_values.size
    boot = np.array([np.median(rng.choice(null_draws, size=n, replace=True)) for _ in range(n_boot)])
    return float((1 + np.sum(boot >= med)) / (1 + boot.size))


def test_median_pvalue_returns_null_when_there_is_no_signal() -> None:
    # Observed and null drawn from the SAME distribution: the test must not find a difference.
    rng = np.random.default_rng(1)
    observed = rng.normal(0.14, 0.13, 1336)
    null = rng.normal(0.14, 0.13, 500)
    p = _median_pvalue(observed, null)
    assert p > 0.05, f"no-signal case must not be significant, got p={p}"


def test_median_pvalue_recovers_a_planted_difference() -> None:
    # Observed median planted well above the null median, at the real pair count.
    rng = np.random.default_rng(2)
    observed = rng.normal(0.30, 0.13, 1336)
    null = rng.normal(0.14, 0.13, 500)
    p = _median_pvalue(observed, null)
    assert p < 0.01, f"planted difference must be recovered, got p={p}"


def test_the_wrong_form_is_the_one_that_fails_to_recover_it() -> None:
    # Pins the actual defect so it cannot come back unnoticed: with the SAME planted data, the
    # aggregate-vs-per-item comparison returns a non-significant number.
    rng = np.random.default_rng(2)
    observed = rng.normal(0.30, 0.13, 1336)
    null = rng.normal(0.14, 0.13, 500)
    wrong = float(np.mean(null >= np.median(observed)))
    correct = _median_pvalue(observed, null)
    assert wrong > 0.05, "the defective form should look non-significant on planted signal"
    assert correct < 0.01
    assert wrong > correct * 10, "the defect inflates p by orders of magnitude"


def test_paired_signed_rank_recovers_a_planted_within_pair_gap() -> None:
    # The statistic used for landmark-vs-imputed fidelity. Planted: a consistent per-pair gap.
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.1, 32)
    a, b = base + 0.02, base
    assert stats.wilcoxon(a - b).pvalue < 0.01
    # and no gap must not be significant
    c = base + rng.normal(0, 0.001, 32)
    assert stats.wilcoxon(base - c).pvalue > 0.05


def test_spearman_brown_lifts_half_data_reliability_as_documented() -> None:
    # 2r/(1+r), and it must be monotone and fixed at the endpoints.
    def sb(r: float) -> float:
        return 2 * r / (1 + r)

    assert sb(0.0) == pytest.approx(0.0)
    assert sb(1.0) == pytest.approx(1.0)
    assert sb(0.3) > 0.3, "correcting half-data reliability must raise it"
    assert sb(0.5) > sb(0.3)


def test_transfer_penalty_is_a_difference_of_like_quantities() -> None:
    # rung 2's headline. Both arms must be the same KIND of aggregate or the subtraction is the
    # aggregate-vs-per-item error again, wearing different clothes.
    in_platform = np.array([0.30, 0.28, 0.32])
    cross_platform = np.array([0.10, 0.12, 0.11])
    penalty = float(np.mean(cross_platform)) - float(np.mean(in_platform))
    assert penalty < 0, "a real transfer cost must come out negative"
    assert penalty == pytest.approx(np.mean(cross_platform - in_platform), abs=1e-12), (
        "mean-of-differences and difference-of-means agree only for paired, equal-length arms; "
        "if they diverge the arms are not paired and the penalty is not a like-for-like subtraction"
    )
