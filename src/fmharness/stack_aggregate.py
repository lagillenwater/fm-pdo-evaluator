"""Reduce Stack's per-query-cell generated output to one row per (line, drug).

Stack's true in-context generation (04_stack_generate.sbatch, Change 1) writes one predicted
row PER QUERY CELL, cell-indexed -- with a 400-real-cell query (8/line), that means multiple
rows per line per drug, each carrying obs["gen_logit"] (Stack's own confidence classifier: high
= still-masked/unresolved, low = confidently-resolved). fmharness.deltas.build_generated_deltas
expects exactly one row per (line, drug); aggregate_generated_replicates is the step between
Stack's raw output and that function.
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
