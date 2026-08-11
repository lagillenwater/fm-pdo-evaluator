"""Shared test contract every Readout must satisfy: null behavior on an
unbalanced, zero-information panel. See tests/test_evaluation.py's
test_interaction_rho_ignores_drug_only_signal_on_an_unbalanced_panel for the
precedent -- interaction_rho's missingness artifact shipped because no test
like this existed before it was found and fixed.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def unbalanced_zero_info_panel(seed: int = 0) -> pd.DataFrame:
    """4 patients x 5 drugs, ~30% of cells missing at random, y_pred carries
    zero patient-level information (constant per drug -- the drug's own mean
    y_true), the exact shape of predictor that exposed the interaction_rho bug.
    """
    rng = np.random.default_rng(seed)
    patients = [f"p{i}" for i in range(4)]
    drugs = [f"d{i}" for i in range(5)]
    rows = [(p, d) for p in patients for d in drugs]
    keep = rng.random(len(rows)) > 0.3
    rows = [r for r, k in zip(rows, keep, strict=True) if k]
    y_true_vals = rng.uniform(0.0, 1.0, size=len(rows))
    df = pd.DataFrame(
        {
            "patient": [r[0] for r in rows],
            "drug": [r[1] for r in rows],
            "y_true": y_true_vals,
        }
    )
    df["y_pred"] = df.groupby("drug")["y_true"].transform("mean")  # zero patient info
    return df


def assert_null_on_unbalanced_zero_info_predictor(
    readout_fn: Callable[[pd.DataFrame], float],
    expected_null: float,
    *,
    seed: int = 0,
    atol: float = 1e-9,
) -> None:
    """A predictor with zero patient-level information must score at its null
    value even on an unbalanced panel -- the exact failure mode that let
    interaction_rho's missingness artifact ship undetected.
    """
    panel = unbalanced_zero_info_panel(seed=seed)
    observed = readout_fn(panel)
    assert np.isclose(observed, expected_null, atol=atol), (
        f"expected null {expected_null}, got {observed} on an unbalanced zero-info panel -- "
        "this readout may have a missingness artifact"
    )
