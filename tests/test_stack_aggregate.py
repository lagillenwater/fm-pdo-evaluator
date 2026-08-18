"""Tests for confidence-filtered aggregation of Stack's per-query-cell generated output."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from fmharness.stack_aggregate import aggregate_generated_replicates


@pytest.fixture
def generated_dir(tmp_path: Path) -> Path:
    genes = ["A", "B"]
    gen = ad.AnnData(
        X=np.array(
            [
                [10.0, 0.0],  # L1 rep1: confidently resolved (kept)
                [12.0, 0.0],  # L1 rep2: confidently resolved (kept)
                [100.0, 0.0],  # L1 rep3: still masked/low-confidence (dropped)
                [5.0, 5.0],  # L2 rep1: low-confidence (dropped)
                [6.0, 6.0],  # L2 rep2: low-confidence (dropped)
            ],
            dtype=np.float32,
        )
    )
    gen.var_names = genes
    gen.obs["cell_line_id"] = ["L1", "L1", "L1", "L2", "L2"]
    gen.obs["gen_logit"] = [-1.0, -0.5, 3.0, 2.0, 4.0]
    d = tmp_path / "generated"
    d.mkdir()
    gen.write_h5ad(d / "drugX.h5ad")
    return d


def test_aggregate_generated_replicates_filters_by_confidence_before_averaging(
    tmp_path: Path, generated_dir: Path
) -> None:
    out_dir = tmp_path / "aggregated"

    summary = aggregate_generated_replicates(generated_dir, out_dir, threshold=0.0)

    reduced = ad.read_h5ad(out_dir / "drugX.h5ad")
    assert list(reduced.obs_names) == ["L1"]  # L2 dropped: zero replicates survive the filter
    assert np.allclose(np.asarray(reduced.X), [[11.0, 0.0]])  # mean of the two KEPT reps, not all three
    naive_mean_gene_a = (10.0 + 12.0 + 100.0) / 3
    assert not np.isclose(float(np.asarray(reduced.X)[0, 0]), naive_mean_gene_a)

    l2 = summary[(summary["pert_id"] == "drugX") & (summary["cell_line_id"] == "L2")].iloc[0]
    assert l2["n_replicates"] == 2
    assert l2["n_kept"] == 0
    assert bool(l2["dropped"])
    l1 = summary[(summary["pert_id"] == "drugX") & (summary["cell_line_id"] == "L1")].iloc[0]
    assert l1["n_replicates"] == 3
    assert l1["n_kept"] == 2
    assert not bool(l1["dropped"])


def test_aggregate_generated_replicates_raises_without_required_obs_columns(
    tmp_path: Path,
) -> None:
    gen = ad.AnnData(X=np.zeros((2, 1), dtype=np.float32))
    gen.var_names = ["A"]
    d = tmp_path / "generated"
    d.mkdir()
    gen.write_h5ad(d / "drugY.h5ad")

    with pytest.raises(ValueError, match="gen_logit"):
        aggregate_generated_replicates(d, tmp_path / "out", threshold=2.5)
