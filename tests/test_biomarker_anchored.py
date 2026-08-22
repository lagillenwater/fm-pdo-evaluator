"""Tests for scripts/biomarker_anchored.py's head-to-head support restriction."""

from __future__ import annotations

import pandas as pd

from biomarker_anchored import restrict_biomarker_support


def test_restrict_biomarker_support_keeps_only_shared_pairs_and_dedupes() -> None:
    # p3/d1 is a WES-covered patient anchored by TWO biomarkers to the same drug (e.g.
    # CDK4-amp AND RB1-del both -> Palbociclib) -- bm_all carries a duplicate (patient,
    # drug) row for it, which must be deduped (kept: the first, y_pred=0.9) so p3 doesn't
    # get double weight in the pooled within-drug correlation.
    #
    # p4/d1 is biomarker-covered but NOT in glob's support (glob is grouped_cv_predict
    # over every actionable-drug row; a real gap would be a data/fold quirk) -- it must
    # be dropped, not scored against a missing global prediction.
    #
    # p1/d2 is in glob's support but has no biomarker defined for it (e.g. no WES) --
    # it must be dropped from the global side too, so both rows are scored on the
    # identical (patient, drug) set.
    bm_all = pd.DataFrame(
        {
            "patient": ["p1", "p2", "p3", "p3", "p4"],
            "drug": ["d1", "d1", "d1", "d1", "d1"],
            "y_true": [0.1, 0.2, 0.3, 0.3, 0.4],
            "y_pred": [1.1, 1.2, 0.9, -5.0, 1.4],
        }
    )
    glob = pd.DataFrame(
        {
            "patient": ["p1", "p2", "p3", "p1"],
            "drug": ["d1", "d1", "d1", "d2"],
            "y_true": [0.1, 0.2, 0.3, 0.5],
            "y_resid": [2.1, 2.2, 2.3, 2.5],
        }
    )

    bm_common, gl_common = restrict_biomarker_support(bm_all, glob)

    assert list(zip(bm_common["patient"], bm_common["drug"])) == [
        ("p1", "d1"),
        ("p2", "d1"),
        ("p3", "d1"),
    ]
    # p3/d1's duplicate is resolved by keeping the FIRST row (y_pred=0.9), not -5.0.
    assert bm_common["y_pred"].tolist() == [1.1, 1.2, 0.9]

    assert list(zip(gl_common["patient"], gl_common["drug"])) == [
        ("p1", "d1"),
        ("p2", "d1"),
        ("p3", "d1"),
    ]
    assert gl_common["y_resid"].tolist() == [2.1, 2.2, 2.3]


def test_restrict_biomarker_support_empty_when_no_shared_pairs() -> None:
    bm_all = pd.DataFrame(
        {"patient": ["p1"], "drug": ["d1"], "y_true": [0.1], "y_pred": [1.0]}
    )
    glob = pd.DataFrame(
        {"patient": ["p2"], "drug": ["d1"], "y_true": [0.2], "y_resid": [2.0]}
    )

    bm_common, gl_common = restrict_biomarker_support(bm_all, glob)

    assert bm_common.empty
    assert gl_common.empty
