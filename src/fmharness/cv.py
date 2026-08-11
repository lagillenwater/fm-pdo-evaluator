"""CV registry, bridging the pydantic Splitter system to the pandas design-frame
pattern the working pipeline (build_sample_design, grouped_cv_predict,
score_generation_eval.py) actually uses.

``src/fmharness/splits/`` operates on pydantic ``Patient`` objects and yields
``SplitFold`` (train/test patient-id tuples). Every real working script operates
on flat ``design[patient, drug, y]`` DataFrames instead. Rather than migrate the
working pipeline onto the heavier pydantic system, this module runs the real
splitter logic (``LeaveSubtypeOut``) and translates its output into the
``design -> Iterator[(train_idx, test_idx)]`` shape already in use everywhere.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import NamedTuple, Protocol, cast, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fmharness.splits.base import SplittablePatient
from fmharness.splits.lso import LeaveSubtypeOut, LSOGranularity


@runtime_checkable
class CVScheme(Protocol):
    """Turn a design frame into train/test folds: design in, index pairs out.

    A scheme takes the flat ``design[patient, drug, y]`` frame and yields
    ``(train_idx, test_idx)`` pairs that must be disjoint *by patient*, not
    merely by row -- one patient contributes many rows (one per drug), so a
    row-wise split would put the same organoid on both sides and leak.

    The indices are POSITIONAL (0..len(design)-1), matching sklearn's splitter
    convention, not design-frame labels. Positional is the safe choice because
    the frames these schemes see have usually been through
    ``filter_leakage``, which returns a ``reset_index(drop=True)`` frame: its
    labels are freshly renumbered and carry no meaning, so a label-based
    contract would silently mean different rows before and after filtering.
    """

    def splits(self, design: pd.DataFrame) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Yield (train_row_idx, test_row_idx) into ``design``, disjoint by patient."""
        ...


class _PatientRow(NamedTuple):
    """Satisfies fmharness.splits.base.SplittablePatient."""

    patient_id: str
    subtype: str | None


class _LeaveSubtypeOutScheme:
    def __init__(
        self,
        patient_subtypes: dict[str, str],
        *,
        seed: int,
        granularity: LSOGranularity = "fine",
        subtype_map: dict[str, str] | None = None,
    ) -> None:
        self.patient_subtypes = patient_subtypes
        self._lso = LeaveSubtypeOut(seed=seed, granularity=granularity, subtype_map=subtype_map)

    def splits(
        self, design: pd.DataFrame
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        patient_ids = sorted(design["patient"].unique())
        missing = [p for p in patient_ids if p not in self.patient_subtypes]
        if missing:
            raise KeyError(f"no subtype declared for patient(s): {missing}")
        patients = cast(
            Sequence[SplittablePatient],
            [_PatientRow(p, self.patient_subtypes[p]) for p in patient_ids],
        )
        patient_pos = pd.Series(np.arange(len(design)), index=design["patient"]).groupby(
            level=0
        )
        for fold in self._lso.split(patients):
            train_idx = np.concatenate(
                [patient_pos.get_group(p).to_numpy() for p in fold.train_patient_ids]
            )
            test_idx = np.concatenate(
                [patient_pos.get_group(p).to_numpy() for p in fold.test_patient_ids]
            )
            yield np.sort(train_idx), np.sort(test_idx)


def leave_subtype_out(
    patient_subtypes: dict[str, str],
    *,
    seed: int,
    granularity: LSOGranularity = "fine",
    subtype_map: dict[str, str] | None = None,
) -> CVScheme:
    """One fold per unique subtype, bridged from ``fmharness.splits.lso.LeaveSubtypeOut``.

    ``patient_subtypes`` maps every patient_id appearing in the design frame to its
    subtype label; missing patients raise ``KeyError`` rather than being silently
    dropped, since a silently-shrunk cohort would understate the fold's true size.
    """
    return _LeaveSubtypeOutScheme(
        patient_subtypes, seed=seed, granularity=granularity, subtype_map=subtype_map
    )
