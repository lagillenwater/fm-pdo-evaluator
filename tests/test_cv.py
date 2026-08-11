"""Tests for the CV registry's leave-subtype-out bridge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmharness.cv import CVScheme, leave_subtype_out


def _design() -> pd.DataFrame:
    # 4 patients, 2 subtypes (A: p1,p2 -- B: p3,p4), 2 drugs each.
    return pd.DataFrame(
        {
            "patient": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
            "drug": ["d1", "d2"] * 4,
            "y": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
    )


def test_leave_subtype_out_satisfies_cv_scheme() -> None:
    subtypes = {"p1": "A", "p2": "A", "p3": "B", "p4": "B"}
    scheme = leave_subtype_out(subtypes, seed=0)
    assert isinstance(scheme, CVScheme)


def test_leave_subtype_out_yields_one_fold_per_subtype() -> None:
    subtypes = {"p1": "A", "p2": "A", "p3": "B", "p4": "B"}
    design = _design()
    scheme = leave_subtype_out(subtypes, seed=0)
    folds = list(scheme.splits(design))
    assert len(folds) == 2  # one fold per subtype (A held out, then B held out)

    for train_idx, test_idx in folds:
        train_idx_arr = np.asarray(train_idx)
        test_idx_arr = np.asarray(test_idx)
        assert set(train_idx_arr) & set(test_idx_arr) == set()
        train_patients = set(design.iloc[train_idx_arr]["patient"])
        test_patients = set(design.iloc[test_idx_arr]["patient"])
        assert train_patients & test_patients == set()  # no patient in both


def test_leave_subtype_out_test_fold_covers_exactly_the_held_out_subtype_rows() -> None:
    subtypes = {"p1": "A", "p2": "A", "p3": "B", "p4": "B"}
    design = _design()
    scheme = leave_subtype_out(subtypes, seed=0)
    folds = list(scheme.splits(design))
    test_patient_sets = [set(design.iloc[np.asarray(te)]["patient"]) for _, te in folds]
    assert {"p1", "p2"} in test_patient_sets
    assert {"p3", "p4"} in test_patient_sets


def test_leave_subtype_out_raises_on_missing_patient_subtype() -> None:
    subtypes = {"p1": "A", "p2": "A"}  # p3, p4 missing
    with pytest.raises(KeyError):
        list(leave_subtype_out(subtypes, seed=0).splits(_design()))
