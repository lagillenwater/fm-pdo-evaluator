"""Tests for selection-metric machinery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmharness.selection import within_drug_percentile


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
