"""Tests for scripts/baselines_soragni.py's shared-drug-set row construction.

Mirrors tests/test_deltas.py's restrict_common_support tests: a drug that is in the
"ref" set (the L1000-matched drugs, or Soragni's own drugs when no L1000 context is
given) but was never screened in GDSC2 must be dropped from EVERY row -- drug-mean and
each l1000:<sig> row alike, not just from the pca/nmf transfer inputs -- so the printed
comparison table is a fair head-to-head on identical drug support.
"""

from __future__ import annotations

import pandas as pd

from baselines_soragni import build_reference_rows


def _ds() -> pd.DataFrame:
    # drug "C" is in Soragni but was never screened in GDSC2.
    return pd.DataFrame(
        {
            "patient": ["p1", "p1", "p1", "p2", "p2", "p2"],
            "drug": ["A", "B", "C", "A", "B", "C"],
            "y": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )


def _l1000_rows() -> list[tuple[str, pd.DataFrame]]:
    # direct_l1000's own table defines "ref" here, and it includes C -- the
    # L1000 context was mappable for C even though GDSC2 never screened it.
    tbl = pd.DataFrame(
        {
            "patient": ["p1", "p1", "p1", "p2", "p2", "p2"],
            "drug": ["A", "B", "C", "A", "B", "C"],
            "y_true": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "y_pred": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )
    return [("l1000:sig1", tbl)]


def test_build_reference_rows_keeps_only_shared_drugs_with_l1000_context() -> None:
    gdsc_drugs = {"A", "B"}  # C never screened in GDSC2

    ref, shared, rows = build_reference_rows(_ds(), gdsc_drugs, _l1000_rows())

    assert ref == {"A", "B", "C"}  # ref is the wider, L1000-matched set (includes C)
    assert shared == ["A", "B"]  # narrowed to what GDSC2 also screened

    names = [name for name, _ in rows]
    assert names == ["drug-mean", "l1000:sig1"]
    # every row -- drug-mean AND l1000:sig1 -- must be narrowed to "shared", not "ref",
    # so it lines up with the pca/nmf transfer support (built from "shared" too).
    for name, tbl in rows:
        assert set(tbl["drug"].astype(str)) == {"A", "B"}, name


def test_build_reference_rows_falls_back_to_ds_drugs_without_l1000_context() -> None:
    gdsc_drugs = {"A", "B"}

    ref, shared, rows = build_reference_rows(_ds(), gdsc_drugs, [])

    # no --l1000-context: ref falls back to ds's own drugs intersected with GDSC2's,
    # so C (never in gdsc_drugs) is excluded from ref already.
    assert ref == {"A", "B"}
    assert shared == ["A", "B"]
    assert set(rows[0][1]["drug"].astype(str)) == {"A", "B"}
