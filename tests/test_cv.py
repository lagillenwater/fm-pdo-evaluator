"""Tests for the CV registry's leave-subtype-out bridge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmharness.cv import CVScheme, group_k_fold, leave_subtype_out, resolve_cv


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


def _six_patient_design() -> pd.DataFrame:
    # 6 patients, 2 drugs each -- enough rows to check a 3-fold split lands
    # exactly 2 patients per test fold, and enough patients to distinguish
    # "n_splits capped at n_patients" from "n_splits honored as given".
    patients = [f"p{i}" for i in range(6)]
    rows_patient = [p for p in patients for _ in range(2)]
    rows_drug = ["d1", "d2"] * 6
    return pd.DataFrame(
        {"patient": rows_patient, "drug": rows_drug, "y": np.arange(12, dtype=float)}
    )


def test_group_k_fold_satisfies_cv_scheme() -> None:
    assert isinstance(group_k_fold(n_splits=5), CVScheme)


def test_group_k_fold_yields_n_splits_folds_disjoint_by_patient() -> None:
    design = _six_patient_design()
    scheme = group_k_fold(n_splits=3)
    folds = list(scheme.splits(design))
    assert len(folds) == 3
    seen_test_patients: set[str] = set()
    for train_idx, test_idx in folds:
        train_idx_arr = np.asarray(train_idx)
        test_idx_arr = np.asarray(test_idx)
        assert set(train_idx_arr) & set(test_idx_arr) == set()
        train_patients = set(design.iloc[train_idx_arr]["patient"])
        test_patients = set(design.iloc[test_idx_arr]["patient"])
        assert train_patients & test_patients == set()
        seen_test_patients |= test_patients
    # every patient appears in exactly one test fold across the 3 folds
    assert seen_test_patients == set(design["patient"])


def test_group_k_fold_caps_at_n_patients_when_n_splits_exceeds_it() -> None:
    # Only 4 patients in leave_subtype_out's _design(); asking for 10 splits
    # must not raise sklearn's "n_splits > n_groups" error -- grouped_cv_predict's
    # own min(n_splits, n_unique) capping is the precedent this mirrors.
    design = _design()
    scheme = group_k_fold(n_splits=10)
    folds = list(scheme.splits(design))
    assert len(folds) == 4


def test_group_k_fold_none_is_true_leave_one_patient_out() -> None:
    design = _six_patient_design()
    scheme = group_k_fold(n_splits=None)
    folds = list(scheme.splits(design))
    assert len(folds) == 6  # one fold per patient
    for _train_idx, test_idx in folds:
        test_patients = set(design.iloc[np.asarray(test_idx)]["patient"])
        assert len(test_patients) == 1


def test_resolve_cv_5fold_and_loo() -> None:
    design = _six_patient_design()
    five_fold = resolve_cv("5fold")
    assert isinstance(five_fold, CVScheme)
    assert len(list(five_fold.splits(design))) == 5

    loo = resolve_cv("loo")
    assert isinstance(loo, CVScheme)
    assert len(list(loo.splits(design))) == 6


def test_resolve_cv_raises_on_unknown_key() -> None:
    with pytest.raises(ValueError, match="unknown"):
        resolve_cv("some_unknown_scheme")


def test_resolve_cv_returns_a_fresh_instance_each_call() -> None:
    # A shared singleton is safe only while every CVScheme is stateless past
    # construction -- resolve_cv must not hand two callers the same object,
    # so a future stateful scheme (a cached fold list, a per-call seed) can't
    # leak between them.
    assert resolve_cv("5fold") is not resolve_cv("5fold")
    assert resolve_cv("loo") is not resolve_cv("loo")
