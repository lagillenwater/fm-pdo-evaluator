"""Reduce Stack's per-query-cell generated output to one row per (line, drug).

Stack's true in-context generation (04_stack_generate.sbatch, Change 1) writes one predicted
row PER QUERY CELL, cell-indexed -- with a 400-real-cell query (8/line), that means multiple
rows per line per drug, each carrying obs["gen_logit"] (Stack's own confidence classifier: high
= still-masked/unresolved, low = confidently-resolved). fmharness.deltas.build_generated_deltas
expects exactly one row per (line, drug); aggregate_generated_replicates is the step between
Stack's raw output and that function.

The QUERY side has the same shape problem: tahoe_query.h5ad (03_stack_context.sbatch) is
cell-indexed too (400 real cells, 8/line, obs["cell_line_id"] carrying the line), because that
is the shape Stack's --test-adata generation input needs -- but build_generated_deltas's
--query-baseline argument must be indexed BY LINE. collapse_query_baseline is that same
reduction applied to the query side, run once up front (Task 7 Step 3) rather than once per
generated file.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from fmharness.deltas import dense


def aggregate_generated_replicates(
    generated_dir: Path,
    out_dir: Path,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Filter-then-average Stack's per-query-cell generated replicates down to one row per line.

    Only replicates with ``gen_logit < threshold`` (Stack's own confidence classifier judging the
    cell confidently-resolved, not still-masked) are averaged per line. A naive unfiltered mean
    is a bias problem, not a variance one: a low-confidence replicate is mechanistically pulled
    toward the query baseline under weak context support, so averaging in more of them just
    estimates that bias more precisely. A (drug, line) with zero surviving replicates is dropped,
    not silently backfilled with the unfiltered mean, which would reintroduce exactly the bias
    being filtered out.

    Writes one reduced ``<pert_id>.h5ad`` (one row per line) per input file to ``out_dir`` --
    the shape ``build_generated_deltas`` already requires, so it needs no changes. Returns a
    summary DataFrame (``pert_id, cell_line_id, n_replicates, n_kept, dropped``) for auditing how
    much each drug's generation was affected by the filter.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[pd.DataFrame] = []
    for f in sorted(Path(generated_dir).glob("*.h5ad")):
        gen = ad.read_h5ad(f)
        if "gen_logit" not in gen.obs or "cell_line_id" not in gen.obs:
            raise ValueError(
                f"{f}: missing obs['gen_logit']/obs['cell_line_id'] -- was this generated "
                "with --mode mdm (not vanilla) against a query baseline carrying cell_line_id?"
            )
        logit = gen.obs["gen_logit"].to_numpy(dtype=float)
        line = gen.obs["cell_line_id"].astype(str).to_numpy()
        x = dense(gen.X)
        keep = logit < threshold

        # per-line mean of the KEPT replicates only, via an indicator-matmul (no explicit
        # per-line loop -- mirrors fmharness.deltas._group_mean's own indicator-matmul pattern,
        # restricted here to the kept subset).
        codes, uniq = pd.factorize(line)
        n_lines = len(uniq)
        n_total = np.bincount(codes, minlength=n_lines)
        n_kept = np.bincount(codes[keep], minlength=n_lines)
        ind = np.zeros((n_lines, len(codes)), dtype=np.float64)
        ind[codes[keep], np.flatnonzero(keep)] = 1.0
        denom = np.where(n_kept == 0, 1.0, n_kept.astype(np.float64))
        means = (ind @ x) / denom[:, None]

        have = n_kept > 0
        if have.any():
            reduced = ad.AnnData(X=means[have].astype(np.float32))
            reduced.obs_names = [str(u) for u in uniq[have]]
            reduced.var_names = [str(v) for v in gen.var_names]
            reduced.var["feature_name"] = list(reduced.var_names)
            reduced.write_h5ad(out_dir / f.name)

        summaries.append(
            pd.DataFrame(
                {
                    "pert_id": f.stem,
                    "cell_line_id": [str(u) for u in uniq],
                    "n_replicates": n_total,
                    "n_kept": n_kept,
                    "dropped": n_kept == 0,
                }
            )
        )
    if not summaries:
        return pd.DataFrame(
            columns=["pert_id", "cell_line_id", "n_replicates", "n_kept", "dropped"]  # type: ignore[arg-type]
        )
    return pd.concat(summaries, ignore_index=True)


def collapse_query_baseline(query_path: Path, out_path: Path) -> pd.DataFrame:
    """Collapse tahoe_query.h5ad's multiple real cells per line down to one mean row per line,
    indexed by cell_line_id.

    tahoe_query.h5ad (03_stack_context.sbatch) holds 8 real single control cells per line,
    indexed by cell position (obs_names "0".."399") with the line id as obs["cell_line_id"] --
    the shape Stack's --test-adata generation input needs. fmharness.deltas.build_generated_deltas
    expects its --query-baseline argument indexed BY LINE (base_df.index.intersection(g.index)
    joins on that index directly). Passing tahoe_query.h5ad straight through silently joins on an
    empty intersection (cell-position ids never match line ids) -- this collapse is a required
    separate step, not the same file serving double duty the way the pre-Task-2 pseudobulk query
    file did.

    Returns an audit DataFrame (cell_line_id, n_cells) alongside writing the collapsed baseline
    to out_path.
    """
    query = ad.read_h5ad(query_path)
    line = query.obs["cell_line_id"].astype(str).to_numpy()
    x = dense(query.X)
    codes, uniq = pd.factorize(line)
    n_lines = len(uniq)
    counts = np.bincount(codes, minlength=n_lines).astype(np.float64)
    ind = np.zeros((n_lines, len(codes)), dtype=np.float64)
    ind[codes, np.arange(len(codes))] = 1.0
    means = (ind @ x) / counts[:, None]
    baseline = ad.AnnData(X=means.astype(np.float32))
    baseline.obs_names = [str(u) for u in uniq]
    baseline.var_names = [str(v) for v in query.var_names]
    baseline.var["feature_name"] = list(baseline.var_names)
    baseline.write_h5ad(out_path)
    return pd.DataFrame({"cell_line_id": [str(u) for u in uniq], "n_cells": counts.astype(int)})


def build_synthetic_replicate_pool(
    baseline: ad.AnnData,
    *,
    n_replicates: int,
    library_size: float,
    seed: int,
) -> ad.AnnData:
    """Expand a one-row-per-sample CPM baseline into a synthetic single-cell-like pool.

    The inverse problem to ``collapse_query_baseline``: bulk data (one CPM row per
    patient) has too few rows to satisfy Stack's ``--mode mdm`` scheduled draw (up to
    ~281 cells from the whole query pool, independent of the true sample count -- see
    04_stack_generate.sbatch/03's real-cell-pool comment), and unlike Tahoe's single-cell
    panel there are no additional real cells to draw from. Poisson-resampling each
    sample's CPM profile at a nominal single-cell ``library_size`` injects a count-level
    noise magnitude in the range Stack was pretrained on (real, sparse single-cell data),
    then renormalizes each replicate back to CPM so the output matches the query file's
    own scale convention. These are NOT real cells -- a documented, seeded approximation,
    not Tahoe's real-cell pool. ``library_size`` trades off noise magnitude: smaller is
    noisier/more single-cell-like, larger converges toward re-feeding the bulk profile.
    """
    x = dense(baseline.X).astype(np.float64)
    mean_counts = x / 1e6 * library_size
    rng = np.random.default_rng(seed)
    tiled_mean = np.repeat(mean_counts, n_replicates, axis=0)
    counts = rng.poisson(tiled_mean)
    row_sum = counts.sum(axis=1, keepdims=True).astype(np.float64)
    row_sum[row_sum == 0] = 1.0
    cpm = counts / row_sum * 1e6

    sample_ids = np.asarray([str(s) for s in baseline.obs_names])
    ids = np.repeat(sample_ids, n_replicates)
    pool = ad.AnnData(X=cpm.astype(np.float32))
    pool.obs_names = [str(i) for i in range(pool.n_obs)]
    pool.obs["cell_line_id"] = ids
    pool.var_names = [str(v) for v in baseline.var_names]
    pool.var["feature_name"] = list(pool.var_names)
    return pool
