"""Tests for confidence-filtered aggregation of Stack's per-query-cell generated output."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from fmharness.deltas import build_generated_deltas
from fmharness.stack_aggregate import aggregate_generated_replicates, collapse_query_baseline


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


def test_collapse_query_baseline_closes_the_build_generated_deltas_index_gap(
    tmp_path: Path, generated_dir: Path
) -> None:
    """End-to-end regression test for the reviewer-found bug: tahoe_query.h5ad
    (03_stack_context.sbatch) is cell-position indexed (obs_names "0".."N-1", the line id
    living in obs["cell_line_id"]), but fmharness.deltas.build_generated_deltas joins its
    --query-baseline argument on the AnnData index directly
    (base_df.index.intersection(g.index)). Passing tahoe_query.h5ad straight through -- as
    the plan originally did -- silently produces an empty intersection; reproduced below
    first, then closed by collapse_query_baseline (which re-indexes the baseline BY LINE, the
    same shape aggregate_generated_replicates already reduces the generated side to).
    """
    # a tahoe_query.h5ad-shaped fixture: 4 real single control cells, 2/line, cell-position
    # obs_names, cell_line_id as an obs COLUMN (not the index) -- 03's exact shape.
    genes = ["A", "B"]
    query = ad.AnnData(
        X=np.array([[5.0, 0.0], [7.0, 0.0], [1.0, 1.0], [3.0, 3.0]], dtype=np.float32)
    )
    query.var_names = genes
    query.obs_names = ["0", "1", "2", "3"]
    query.obs["cell_line_id"] = ["L1", "L1", "L2", "L2"]
    query_path = tmp_path / "tahoe_query.h5ad"
    query.write_h5ad(query_path)

    # Task 3's aggregation of the generated_dir fixture: threshold=0.0 keeps only L1's two
    # confidently-resolved replicates (mean [11, 0]); L2's replicates are both dropped, so the
    # aggregated output has exactly one line, L1 -- indexed BY LINE, not by cell position.
    agg_dir = tmp_path / "aggregated"
    aggregate_generated_replicates(generated_dir, agg_dir, threshold=0.0)
    assert list(ad.read_h5ad(agg_dir / "drugX.h5ad").obs_names) == ["L1"]

    # Reproduce the reviewer's bug: passing tahoe_query.h5ad straight through as
    # --query-baseline joins cell-position ids ("0".."3") against the aggregated output's line
    # ids ("L1") -- an empty intersection every time, so build_generated_deltas finds zero
    # delta rows for every file and raises.
    with pytest.raises(ValueError, match="no generated files matched a drug"):
        build_generated_deltas(agg_dir, query_path, {"drugX": "D1"}, use_logcpm=False)

    # The fix: collapse the query baseline to one mean row per line, indexed by cell_line_id,
    # first.
    baseline_path = tmp_path / "tahoe_query_baseline.h5ad"
    audit = collapse_query_baseline(query_path, baseline_path)
    assert dict(zip(audit["cell_line_id"], audit["n_cells"], strict=True)) == {"L1": 2, "L2": 2}
    baseline = ad.read_h5ad(baseline_path)
    assert list(baseline.obs_names) == ["L1", "L2"]
    assert np.allclose(np.asarray(baseline.X), [[6.0, 0.0], [2.0, 2.0]])  # per-line means

    delta, key = build_generated_deltas(agg_dir, baseline_path, {"drugX": "D1"}, use_logcpm=False)
    assert delta.shape == (1, 2)
    assert list(key["patient"]) == ["L1"]
    assert list(key["drug"]) == ["D1"]
    assert np.allclose(delta.to_numpy(), [[5.0, 0.0]])  # generated 11.0 - baseline 6.0, 0 - 0
