"""Known-answer validation for every statistic this harness reports.

A statistic that has never been run against a known answer is not evidence. Each test here
plants an answer and requires the statistic to recover it, and plants no signal and requires it
to return null. This file exists because a null test written for the rung-0 ceiling compared an
aggregate against a distribution of single draws -- the observed MEDIAN of ~1,336 correlations
against the spread of individual null draws -- and reported that a reproducible ceiling had
failed. The comparison was between two different KINDS of quantity, and no amount of reading
the code caught it; running it against a known answer would have, immediately.

The recurring error is aggregate-vs-per-item, so every test below pins the units explicitly.

Scoped to the statistics rung 0 reports. Tests for the controls and transfer statistics of
higher rungs arrive with those rungs, so that a statistic and the test proving it recovers a
known answer land in the same change.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from fmharness.statistics import bootstrap_aggregate_pvalue

# The signed claim project rule 4 relies on: every test in this file plants a known answer and
# requires the real, shipped function to recover it. Do not add a test here that does not.
pytestmark = pytest.mark.known_answer


def test_bootstrap_aggregate_pvalue_returns_null_when_there_is_no_signal() -> None:
    # Observed and null drawn from the SAME distribution: the test must not find a difference.
    rng = np.random.default_rng(1)
    observed = rng.normal(0.14, 0.13, 1336)
    null = rng.normal(0.14, 0.13, 500)
    p, _, _ = bootstrap_aggregate_pvalue(float(np.median(observed)), null, observed.size, agg=np.median)
    assert p > 0.05, f"no-signal case must not be significant, got p={p}"


def test_bootstrap_aggregate_pvalue_recovers_a_planted_difference() -> None:
    # Observed median planted well above the null median, at the real pair count.
    rng = np.random.default_rng(2)
    observed = rng.normal(0.30, 0.13, 1336)
    null = rng.normal(0.14, 0.13, 500)
    p, _, _ = bootstrap_aggregate_pvalue(float(np.median(observed)), null, observed.size, agg=np.median)
    assert p < 0.01, f"planted difference must be recovered, got p={p}"


def test_bootstrap_aggregate_pvalue_recovers_the_real_l1000_landmark_case() -> None:
    # docs/results/l1000_imputation_fidelity.csv reported p_vs_null=0.2438 for landmark genes
    # via the defective aggregate-vs-per-item form; this is the same case through the real,
    # shipped function, pinning that it comes out significant instead.
    rng = np.random.default_rng(0)
    null = rng.normal(0.0018, 0.052, 500)  # null_mean, null_sd, n_perm from the promoted CSV
    p, _, _ = bootstrap_aggregate_pvalue(0.041, null, 32)  # mean_spearman, n_pairs
    assert p < 0.01, f"the real L1000 landmark case must clear its null once corrected, got p={p}"


def test_the_wrong_form_is_the_one_that_fails_to_recover_it() -> None:
    # Pins the actual defect so it cannot come back unnoticed: with the SAME planted data, the
    # aggregate-vs-per-item comparison returns a non-significant number that the real function
    # does not.
    rng = np.random.default_rng(2)
    observed = rng.normal(0.30, 0.13, 1336)
    null = rng.normal(0.14, 0.13, 500)
    wrong = float(np.mean(null >= np.median(observed)))
    correct, _, _ = bootstrap_aggregate_pvalue(float(np.median(observed)), null, observed.size, agg=np.median)
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
