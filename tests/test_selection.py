"""Tests for selection-metric machinery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmharness.selection import (
    broadly_active_drugs,
    shortlist_concentration,
    within_drug_percentile,
)


def test_within_drug_percentile_makes_each_drug_uniform() -> None:
    # d_toxic is on a completely different scale from d_mild. In raw AUC, d_toxic dominates
    # any cross-drug comparison; after the transform each drug spans the same (0, 1] ranks,
    # so scale carries no advantage.
    preds = pd.DataFrame(
        {
            "patient": ["A", "B", "C", "D"] * 2,
            "drug": ["d_toxic"] * 4 + ["d_mild"] * 4,
            "y_true": [0.01, 0.02, 0.03, 0.04, 0.90, 0.92, 0.94, 0.96],
            "y_pred": [0.04, 0.03, 0.02, 0.01, 0.96, 0.94, 0.92, 0.90],
        }
    )
    out = within_drug_percentile(preds)
    for drug in ("d_toxic", "d_mild"):
        g = out.loc[out["drug"] == drug, "y_true"].to_numpy()
        assert np.allclose(np.sort(g), [0.25, 0.5, 0.75, 1.0])
    # Order within a drug is preserved; only the scale changes.
    toxic = out.loc[out["drug"] == "d_toxic", "y_true"].to_numpy()
    assert np.allclose(toxic, [0.25, 0.5, 0.75, 1.0])
    # The input frame is untouched.
    assert np.isclose(preds["y_true"].iloc[0], 0.01)


def test_within_drug_percentile_only_transforms_named_columns() -> None:
    preds = pd.DataFrame(
        {
            "patient": ["A", "B"],
            "drug": ["d1", "d1"],
            "y_true": [10.0, 20.0],
            "y_pred": [1.0, 2.0],
            "y_prior": [5.0, 5.0],
        }
    )
    out = within_drug_percentile(preds, cols=("y_true",))
    assert np.allclose(out["y_true"].to_numpy(), [0.5, 1.0])
    assert np.allclose(out["y_pred"].to_numpy(), [1.0, 2.0])
    assert np.allclose(out["y_prior"].to_numpy(), [5.0, 5.0])


def _panel() -> pd.DataFrame:
    # Four lines x three drugs. d_tox is below each line's median everywhere (broadly active);
    # d_sel is the true best for exactly one line; d_weak is never good.
    return pd.DataFrame(
        {
            "patient": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D"],
            "drug": ["d_tox", "d_sel", "d_weak"] * 4,
            "y_true": [
                0.20,
                0.10,
                0.90,  # A: d_sel best
                0.20,
                0.50,
                0.90,  # B: d_tox best
                0.20,
                0.60,
                0.90,  # C: d_tox best
                0.20,
                0.70,
                0.90,  # D: d_tox best
            ],
        }
    )


def test_broadly_active_drugs_flags_the_pan_potent_compound() -> None:
    preds = _panel()
    assert broadly_active_drugs(preds) == {"d_tox"}


def test_shortlist_concentration_collapsed_model() -> None:
    # A model that ignores the line and always ranks d_tox first: 1 distinct pick, modal
    # share 1.0, and every pick is a broadly-active compound.
    preds = _panel()
    preds["y_pred"] = preds["drug"].map({"d_tox": 0.0, "d_sel": 1.0, "d_weak": 2.0})  # type: ignore[arg-type]
    c = shortlist_concentration(preds)
    assert c["distinct"] == 1.0
    assert c["modal_drug"] == "d_tox"
    assert np.isclose(float(c["modal_share"]), 1.0)
    assert np.isclose(float(c["broadly_active_share"]), 1.0)
    assert c["n_lines"] == 4.0


def test_shortlist_concentration_observed_reference() -> None:
    # Scoring y_true against itself gives the observed reference row: d_sel wins for A,
    # d_tox for B/C/D -> 2 distinct, modal share 3/4, and 3 of 4 picks broadly active.
    c = shortlist_concentration(_panel(), pred_col="y_true")
    assert c["distinct"] == 2.0
    assert c["modal_drug"] == "d_tox"
    assert np.isclose(float(c["modal_share"]), 0.75)
    assert np.isclose(float(c["broadly_active_share"]), 0.75)
