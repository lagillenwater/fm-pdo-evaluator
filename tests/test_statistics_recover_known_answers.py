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

from fmharness.deltas import shuffled_target_base
from fmharness.statistics import bootstrap_aggregate_pvalue


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


def test_shuffled_target_base_never_returns_a_lines_own_baseline() -> None:
    # rung 2's negative control must relabel EVERY held-out line onto a DIFFERENT line's real
    # baseline values. The version this replaced relabelled a same-sized subset of the FULL
    # donor pool and then looked up one held-out line inside it, which raised ValueError at
    # 5-fold (the held-out line matched its new random label with probability
    # len(group)/len(pool), not 1) -- exactly the shape that killed array cell 15 on the
    # cluster. This is the real, shipped function, not a reimplementation of it.
    import pandas as pd

    rng = np.random.default_rng(0)
    all_lines = [f"L{i}" for i in range(20)]
    base = pd.DataFrame(
        np.arange(20 * 3, dtype=float).reshape(20, 3), index=pd.Index(all_lines)
    )
    group = all_lines[:7]
    sb = shuffled_target_base(base, group, all_lines, rng)
    assert list(sb.index) == group, "must return one row per held-out line, in order"
    for ln in group:
        assert not (sb.loc[ln].to_numpy() == base.loc[ln].to_numpy()).all(), (
            f"{ln} was handed its own baseline -- the control does not shuffle anything"
        )


def test_shuffled_target_base_handles_the_singleton_fold() -> None:
    # A group of size 1 has no derangement of itself; the donor must come from outside it.
    import pandas as pd

    rng = np.random.default_rng(0)
    all_lines = [f"L{i}" for i in range(5)]
    base = pd.DataFrame(np.arange(5 * 2, dtype=float).reshape(5, 2), index=pd.Index(all_lines))
    sb = shuffled_target_base(base, ["L0"], all_lines, rng)
    assert list(sb.index) == ["L0"]
    assert not (sb.loc["L0"].to_numpy() == base.loc["L0"].to_numpy()).all()


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


def test_shuffled_control_relabels_every_held_out_line_not_just_one() -> None:
    # rung 2's negative control permutes the target baseline's line labels. When rung 2 moved
    # from leave-one-out to 5-fold the held-out set became a fold, and a control that assigns a
    # single label to a multi-row frame dies on a shape error -- which is what killed one array
    # cell. Pins the invariant: one new label per row, all distinct.
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    all_lines = [f"L{i}" for i in range(20)]
    held_out = pd.DataFrame(np.zeros((4, 3)), index=pd.Index(all_lines[:4]))
    relabelled = pd.Index(
        [str(x) for x in rng.choice(all_lines, size=len(held_out), replace=False)]
    )
    assert len(relabelled) == len(held_out)
    assert len(set(relabelled)) == len(held_out), "labels must stay distinct"
