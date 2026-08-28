"""Exact derangement-based permutation null for rung 0's split-half headline.

Rung 0's promoted p-values (`scripts/delta_reproducibility.py`, `summarize`) bootstrap the
observed mean split-half correlation against a mismatched-pair null pool, but that pool's
draws are not exchangeable: `stratified_null_draws` reuses each (line, drug) pair's
half-profiles across many mismatched-pair comparisons, so treating them as an i.i.d. pool
understates their true dependence. `verification.md`'s "Write-up caveat" bounded that
exposure theoretically (roughly 100 bootstrap standard errors of headroom, so the dependence
would need to inflate the null's variance ~3,000-fold to change the conclusion) but did not
measure it directly. This script carries the dependence by construction instead of assuming
it away: it deranges the pairing between the two half-profile pivots -- a permutation with no
fixed points, so every mismatched draw uses a real profile but never the correct partner --
and builds the null distribution of the MEAN mismatched correlation directly from n_perm such
derangements. The ratio of that derangement-null variance to the i.i.d.-pool bootstrap's
assumed variance is the design effect the exchangeable-pool treatment ignores.

Reuses the measurement core in `scripts/delta_reproducibility.py` (`build_split_half_frame`,
`score_split_half`, `stratified_null_draws`, `masked_rowwise_pearson`) rather than
reimplementing it, loaded the same way `tests/test_rung0_controls.py` loads it.

  python scripts/derangement_null.py --local-dir /scratch/alpine/$USER/tahoe_pseudobulk_de \\
      --drug-names-file <a file of Tahoe drug names, one per line> \\
      --panel-file results/rung1_panel/common_panel.txt --out-dir rung0_outputs
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "delta_reproducibility",
    Path(__file__).resolve().parents[1] / "scripts" / "delta_reproducibility.py",
)
assert _SPEC is not None and _SPEC.loader is not None
dr = importlib.util.module_from_spec(_SPEC)
sys.modules["delta_reproducibility"] = dr
_SPEC.loader.exec_module(dr)


def sample_derangement(rng: np.random.Generator, n: int, max_tries: int = 1000) -> np.ndarray:
    """A permutation of ``range(n)`` with no fixed points, by rejection sampling.

    A uniform random permutation is a derangement with probability ~1/e (inclusion-exclusion),
    so ``max_tries`` is generous headroom for the expected ~e attempts, not a real limit --
    except at n=1, where no derangement exists and every attempt fails by construction.
    """
    identity = np.arange(n)
    for _ in range(max_tries):
        perm = rng.permutation(n)
        if not np.any(perm == identity):
            return perm
    raise RuntimeError(f"failed to sample a derangement of size {n} within {max_tries} tries")


def derangement_null(
    piv0: pd.DataFrame,
    piv1: pd.DataFrame,
    r: np.ndarray,
    min_genes: int,
    n_perm: int,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    """The exact-permutation null for the split-half mean, carrying the half-profile-sharing
    dependence by construction instead of treating the mismatched-pair pool as exchangeable.

    ``piv0``/``piv1``/``r`` are exactly what `score_split_half` returns: ``r`` may carry NaNs
    (rows below ``min_genes`` shared finite entries, or zero variance), which are dropped here
    together with the corresponding pivot rows before deranging. For each of ``n_perm``
    derangements sigma of the row order, every row ``i`` is paired with half-1's row
    ``sigma(i)`` -- never its own match, by construction -- and the mean Pearson r over that
    mismatched pairing is one null draw. ``design_effect`` is the ratio of the actual sampling
    variance of the mean under derangement resampling to the variance an i.i.d. pool of the
    same size would have (``stratified_null_draws``'s ``any_pair`` stratum, the pool a
    derangement's composition matches): the number the write-up caveat in
    `docs/tasks/rung0-replicate-ceiling/verification.md` could only bound theoretically.

    Returns the summary dict (one row's worth of columns) and the raw array of ``n_perm``
    permutation means.
    """
    finite = np.isfinite(r)
    piv0_f, piv1_f = piv0.loc[finite], piv1.loc[finite]
    a = piv0_f.to_numpy(dtype=float)
    b = piv1_f.to_numpy(dtype=float)
    n = a.shape[0]
    observed_mean = float(np.mean(r[finite]))

    rng = np.random.default_rng(seed)
    perm_means = np.empty(n_perm, dtype=float)
    for k in range(n_perm):
        sigma = sample_derangement(rng, n)
        perm_means[k] = float(np.nanmean(dr.masked_rowwise_pearson(a, b[sigma], min_genes)))

    pool = dr.stratified_null_draws(piv0_f, piv1_f, n_perm=n_perm, seed=seed, min_genes=min_genes)[
        "any_pair"
    ]
    var_pool = float(np.var(pool, ddof=1))
    se_iid = float(np.sqrt(var_pool / n))
    design_effect = float(np.var(perm_means, ddof=1) / (var_pool / n))
    p_exact = float((1 + np.sum(perm_means >= observed_mean)) / (1 + n_perm))
    perm_mean_mean = float(np.mean(perm_means))
    perm_mean_sd = float(np.std(perm_means, ddof=1))
    z_derangement = (
        (observed_mean - perm_mean_mean) / perm_mean_sd if perm_mean_sd > 0 else float("nan")
    )

    summary = {
        "n_pairs": int(n),
        "n_perm": int(n_perm),
        "observed_mean": round(observed_mean, 4),
        "perm_mean_mean": round(perm_mean_mean, 4),
        "perm_mean_sd": round(perm_mean_sd, 4),
        "p_exact": round(p_exact, 4),
        "se_iid_pool": round(se_iid, 5),
        "design_effect": round(design_effect, 3),
        "z_derangement": round(z_derangement, 2),
    }
    return summary, perm_means


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local-dir", required=True, help="dir with the Tahoe DE parquet (on scratch)")
    ap.add_argument("--drugs-cid-file", default="data/static/tahoe_target_cids.txt")
    ap.add_argument(
        "--drug-names-file",
        default=None,
        help="one Tahoe drug name per line; bypasses the HuggingFace name lookup so fixtures "
        "and offline runs need no `datasets` import.",
    )
    ap.add_argument(
        "--panel-file", default=None, help="one gene per line; same panel the ceiling was scored on"
    )
    ap.add_argument("--replicate-col", default=None, help="plate/replicate column (auto-detected)")
    ap.add_argument("--min-genes", type=int, default=50, help="min shared genes to score a pair")
    ap.add_argument("--n-perm", type=int, default=500, help="derangement null draws")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="rung0_outputs")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent

    local = Path(args.local_dir)
    local = local if local.is_absolute() else repo / local
    paths = sorted(str(p) for p in local.rglob("*.parquet") if dr.DE in str(p))
    if not paths:
        raise SystemExit(f"no {dr.DE} parquet under {local}")
    if args.drug_names_file:
        names = sorted(
            {ln.strip() for ln in Path(args.drug_names_file).read_text().splitlines() if ln.strip()}
        )
    else:
        cid_file = Path(args.drugs_cid_file)
        cid_file = cid_file if cid_file.is_absolute() else repo / cid_file
        names = dr._target_names(repo, cid_file)
    print(f"{len(names)} target drugs; reading {len(paths)} DE parquet files ...")

    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    de, repl = dr.build_split_half_frame(
        paths, names, args.replicate_col, local.parent / "duckdb_tmp"
    )
    de = de.dropna(subset=["lfc0", "lfc1"])
    if de.empty:
        raise SystemExit("no (line, drug, gene) had both plate halves -- too few plates per pair?")

    if args.panel_file:
        panel_all = {
            ln.strip() for ln in Path(args.panel_file).read_text().splitlines() if ln.strip()
        }
        panel = panel_all & set(de["gene_name"].unique())
        print(f"scoring on the supplied panel: {len(panel)} of {len(panel_all)} genes present")
    else:
        panel = set(de["gene_name"].unique())
        print(f"no --panel-file given; scoring on all {len(panel)} genes present in the data")

    r, piv0, piv1 = dr.score_split_half(de, panel, min_genes=args.min_genes)
    if not np.any(np.isfinite(r)):
        raise SystemExit("no (line, drug) pair had enough shared panel genes to score")

    summary, perm_means = derangement_null(piv0, piv1, r, args.min_genes, args.n_perm, args.seed)
    summary_row = {"replicate_col": repl, "n_genes": len(panel), **summary}

    summary_path = out_dir / "rung0_derangement_summary.csv"
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    dr._write_params_sidecar(summary_path, args, extra={"n_pairs": summary["n_pairs"]})

    pd.DataFrame({"perm_mean": perm_means}).to_csv(
        out_dir / "rung0_derangement_perm_means.csv", index=False
    )

    print("\n=== derangement-based exact permutation null (rung 0, final verification step) ===")
    for k, v in summary_row.items():
        print(f"  {k:22s} {v}")
    print(
        f"\nobserved mean r = {summary['observed_mean']:.4f}, derangement null mean = "
        f"{summary['perm_mean_mean']:.4f} +/- {summary['perm_mean_sd']:.4f} (sd), "
        f"p_exact = {summary['p_exact']:.4f}, design effect = {summary['design_effect']:.3f} "
        f"(se_iid_pool = {summary['se_iid_pool']:.5f}).\nwrote {summary_path}"
    )


if __name__ == "__main__":
    main()
