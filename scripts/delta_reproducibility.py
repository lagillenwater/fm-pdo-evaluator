"""Delta reproducibility ceiling: how noisy is the real Tahoe delta itself?

Check 1 asks how faithfully a source reproduces the real (line, drug) delta and finds r ~ 0.2.
Whether 0.2 is "near the achievable ceiling" or "leaves room for Stack" depends on how reproducible
the real delta is to begin with. This measures that noise floor the same way the label-ceiling
script measured viability reproducibility: split each (line, drug)'s replicate plates into two
halves, aggregate each half's delta, and correlate the two halves over the same top-HVG genes
Check 1 uses. That split-half correlation is the delta's own test-retest reliability -- the most any
predictor, Stack included, could score on Check 1. Spearman-Brown then lifts the half-data number to
the full-data delta the Check-1 sources actually target.

Reuses the DuckDB-over-local-parquet path (the Tahoe DE config is already on scratch from the
pseudobulk shortcut), so it needs no GPU and runs alongside the context build.

  python scripts/delta_reproducibility.py --local-dir /scratch/alpine/$USER/tahoe_pseudobulk_de \\
      --drugs-cid-file data/static/tahoe_target_cids.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fmharness.statistics import bootstrap_aggregate_pvalue

TAHOE = "tahoebio/Tahoe-100M"
DE = "pseudobulk_differential_expression"
REPL_CANDIDATES = ("plate", "Plate", "plate_barcode", "plate_id", "replicate", "batch")


def _target_names(repo: Path, cid_file: Path) -> list[str]:
    """Tahoe drug NAMES for the target PubChem CIDs (drug_metadata), matching the shortcut."""
    from datasets import load_dataset  # type: ignore  # Alpine-only

    cids = {t for t in cid_file.read_text().split() if t}
    dm = load_dataset(TAHOE, "drug_metadata", split="train").to_pandas()
    dm = dm[dm["pubchem_cid"].notna()].copy()
    dm["cid"] = dm["pubchem_cid"].map(lambda c: str(int(c)))
    return sorted(dm[dm["cid"].isin(cids)]["drug"].astype(str).unique())


def build_split_half_frame(
    paths: list[str],
    target_names: list[str],
    repl_col: str | None,
    tmp: Path,
    memory_limit: str = "36GB",
) -> tuple[pd.DataFrame, str]:
    """Per (line, drug, gene), the mean log2FoldChange in each of two plate halves, via DuckDB.

    Splits plates by ``hash(repl_col) % 2`` (deterministic, no RNG) and aggregates each half
    IN-ENGINE, so raw rows are never materialized. Returns the long frame plus the chosen
    replicate column.
    """
    import duckdb  # type: ignore  # Alpine-only

    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{tmp}'")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute("SET preserve_insertion_order=false")
    schema = con.execute("DESCRIBE SELECT * FROM read_parquet(?) LIMIT 0", [paths]).df()
    cols = list(schema["column_name"])
    print(f"DE columns: {cols}")
    candidates = ([repl_col] if repl_col else []) + list(REPL_CANDIDATES)
    chosen = next((c for c in candidates if c in cols), None)
    if chosen is None:
        raise SystemExit(f"no replicate column in {cols}; pass --replicate-col")
    print(f"splitting plates by hash({chosen}) % 2")
    de = con.execute(
        f"""SELECT Cell_ID_DepMap AS patient, drug, gene_name,
                   avg(log2FoldChange) FILTER (WHERE hash({chosen}) % 2 = 0) AS lfc0,
                   avg(log2FoldChange) FILTER (WHERE hash({chosen}) % 2 = 1) AS lfc1
            FROM read_parquet(?)
            WHERE drug IN (SELECT unnest(?)) AND {chosen} IS NOT NULL
            GROUP BY Cell_ID_DepMap, drug, gene_name""",
        [paths, target_names],
    ).df()
    return de, chosen


def masked_rowwise_pearson(a: np.ndarray, b: np.ndarray, min_genes: int) -> np.ndarray:
    """Pearson r per row between ``a`` and ``b``, over entries finite in both.

    Vectorized across rows; rows with fewer than ``min_genes`` shared finite entries or
    zero variance come back NaN.
    """
    ok = np.isfinite(a) & np.isfinite(b)
    n = ok.sum(axis=1)
    a0 = np.where(ok, a, 0.0)
    b0 = np.where(ok, b, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = a0.sum(axis=1) / n
        mb = b0.sum(axis=1) / n
        ac = np.where(ok, a - ma[:, None], 0.0)
        bc = np.where(ok, b - mb[:, None], 0.0)
        cov = (ac * bc).sum(axis=1)
        va = (ac**2).sum(axis=1)
        vb = (bc**2).sum(axis=1)
        r = cov / np.sqrt(va * vb)
    r[(n < min_genes) | (va <= 0) | (vb <= 0)] = np.nan
    return r


def score_split_half(
    de: pd.DataFrame, panel: set[str], min_genes: int = 50
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Per-(line, drug) split-half Pearson over the panel genes, plus the half pivots."""
    d = de[de["gene_name"].isin(panel)]
    piv0 = d.pivot_table(index=["patient", "drug"], columns="gene_name", values="lfc0")
    piv1 = d.pivot_table(index=["patient", "drug"], columns="gene_name", values="lfc1")
    common = piv0.index.intersection(piv1.index)
    piv0, piv1 = piv0.loc[common], piv1.loc[common]
    # pivot_table drops all-NaN columns per half, so the two halves can carry different gene
    # sets even after the index intersection above -- callers that skip main()'s dropna could
    # otherwise silently correlate misaligned genes whenever the column COUNTS happen to match.
    cols = piv0.columns.intersection(piv1.columns)
    piv0, piv1 = piv0[cols], piv1[cols]
    r = masked_rowwise_pearson(piv0.to_numpy(dtype=float), piv1.to_numpy(dtype=float), min_genes)
    return r, piv0, piv1


def stratified_null_draws(
    piv0: pd.DataFrame,
    piv1: pd.DataFrame,
    n_perm: int = 500,
    seed: int = 0,
    min_genes: int = 50,
) -> dict[str, np.ndarray]:
    """Mismatched-pair null correlations per stratum.

    any_pair: two different pairs (continuity with the archived lineage's first run).
    diff_drug: different line AND drug -- the generic-structure floor the ceiling clears.
    same_drug: same drug, different line -- the line-specificity floor.
    """
    lines = piv0.index.get_level_values(0).to_numpy(dtype=str)
    drugs = piv0.index.get_level_values(1).to_numpy(dtype=str)
    n = len(piv0)
    ii, jj = np.divmod(np.arange(n * n), n)
    off = ii != jj
    ii, jj = ii[off], jj[off]
    same_drug = drugs[ii] == drugs[jj]
    same_line = lines[ii] == lines[jj]
    strata = {
        "any_pair": np.ones(ii.size, dtype=bool),
        "diff_drug": ~same_drug & ~same_line,
        "same_drug": same_drug & ~same_line,
    }
    rng = np.random.default_rng(seed)
    a, b = piv0.to_numpy(dtype=float), piv1.to_numpy(dtype=float)
    out: dict[str, np.ndarray] = {}
    for name, mask in strata.items():
        avail = np.flatnonzero(mask)
        if avail.size == 0:
            out[name] = np.array([])
            continue
        pick = rng.choice(avail, size=min(n_perm, avail.size), replace=False)
        r = masked_rowwise_pearson(a[ii[pick]], b[jj[pick]], min_genes)
        out[name] = r[np.isfinite(r)]
    return out


def _write_params_sidecar(result_path, args_ns, extra=None) -> None:
    """Record the git sha and every resolved argument beside the result.

    A ceiling used as a denominator has to be checkable against a rerun; a bare CSV is a number
    with no way back to the code and parameters that produced it. This script had none, which is
    how its value lived in doc prose for weeks with nothing behind it.
    """
    import json as _json
    import subprocess as _sp
    from pathlib import Path as _P

    try:
        sha = _sp.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        sha = "unknown"
    import os as _os

    side = _P(str(result_path)).with_suffix(".params.json")
    side.write_text(
        _json.dumps(
            {
                "result": _P(str(result_path)).name,
                "git_sha": sha,
                "slurm_job_id": _os.environ.get("SLURM_JOB_ID", "local"),
                "args": {k: str(v) for k, v in vars(args_ns).items()},
                **(extra or {}),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {side}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local-dir", required=True, help="dir with the Tahoe DE parquet (on scratch)")
    ap.add_argument("--drugs-cid-file", default="data/static/tahoe_target_cids.txt")
    ap.add_argument("--replicate-col", default=None, help="plate/replicate column (auto-detected)")
    ap.add_argument("--n-hvg", type=int, default=2000, help="top HVGs, matching check 1")
    ap.add_argument("--min-genes", type=int, default=50, help="min shared genes to score a pair")
    ap.add_argument(
        "--panel-file",
        default=None,
        help="one gene per line; pins the ceiling to the SAME panel it will be a denominator "
        "for. Without it the ceiling is top-HVG and not comparable to a panel-scored rung.",
    )
    ap.add_argument("--n-perm", type=int, default=500, help="mismatched-pair null draws")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/delta_reproducibility.csv")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent

    local = Path(args.local_dir)
    local = local if local.is_absolute() else repo / local
    paths = sorted(str(p) for p in local.rglob("*.parquet") if DE in str(p))
    if not paths:
        raise SystemExit(f"no {DE} parquet under {local}")
    cid_file = Path(args.drugs_cid_file)
    cid_file = cid_file if cid_file.is_absolute() else repo / cid_file
    names = _target_names(repo, cid_file)
    print(f"{len(names)} target drugs; reading {len(paths)} DE parquet files ...")

    de, repl = build_split_half_frame(paths, names, args.replicate_col, local.parent / "duckdb_tmp")
    de = de.dropna(subset=["lfc0", "lfc1"])
    if de.empty:
        raise SystemExit("no (line, drug, gene) had both plate halves -- too few plates per pair?")
    de["mean"] = (de["lfc0"].to_numpy() + de["lfc1"].to_numpy()) / 2.0

    # The gene set has to match whatever this ceiling will be a DENOMINATOR for. Scoring the
    # ceiling on top-2000 HVG while rung 1 scores on the 14,121-gene common panel makes
    # "fraction of achievable" a ratio between two different measurements -- the same class of
    # error as the panel bug itself. --panel-file pins it to the panel actually used.
    if args.panel_file:
        panel = {ln.strip() for ln in Path(args.panel_file).read_text().splitlines() if ln.strip()}
        hvg = panel & set(de["gene_name"].unique())
        print(
            f"scoring the ceiling on the supplied panel: {len(hvg)} of {len(panel)} genes present"
        )
    else:
        gene_var = de.groupby("gene_name")["mean"].var()
        hvg = set(gene_var.sort_values(ascending=False).index[: args.n_hvg])
        print(f"scoring the ceiling on top-{args.n_hvg} HVG (no --panel-file given)")

    # per (line, drug): split-half Pearson r between the two plate halves over the HVG genes.
    # `r` stays aligned with piv0/piv1's rows (NaNs and all) -- Task 4's tercile/per-gene
    # diagnostics need that alignment; `r_fin` is the finite-only view for aggregate stats.
    r, piv0, piv1 = score_split_half(de, hvg, min_genes=args.min_genes)
    r_fin = r[np.isfinite(r)]
    if r_fin.size == 0:
        raise SystemExit("no (line, drug) pair had enough shared HVG genes to score")

    # NEGATIVE CONTROL, STRATIFIED. A split-half correlation has a nonzero floor because genes
    # share structure whether or not two halves come from the same perturbation. But a single
    # "mismatched pair" null conflates two very different floors, and the first run showed why:
    # it drew two random pairs, so whenever they happened to share a DRUG the correlation was
    # high -- drug effects dominate the delta -- and the null came back at median 0.139 with 23%
    # of draws exceeding the observed 0.299. That null is inflated by same-drug matches and
    # cannot be read as a floor for reproducibility.
    #
    # Three strata, the same distinction Check 1b draws between shuffle_all and within_drug:
    #   any_pair      two random pairs. Mixed; reported only for continuity with the first run.
    #   diff_drug     different line AND different drug -- the floor from generic gene structure.
    #                 This is the one the CEILING must clear to be a ceiling at all.
    #   same_drug     same drug, different line -- the floor for LINE specificity. A split-half
    #                 above this says the pair's delta is reproducible beyond its drug's effect.
    nulls = stratified_null_draws(
        piv0, piv1, n_perm=args.n_perm, seed=args.seed, min_genes=args.min_genes
    )
    for k, v in nulls.items():
        med_k = float(np.median(v)) if v.size else float("nan")
        print(f"null[{k:<10}] median r = {med_k:+.3f} over {v.size} draws")
    nl = nulls["diff_drug"] if nulls["diff_drug"].size else nulls["any_pair"]
    null_med = float(np.median(nl)) if nl.size else float("nan")

    med = float(np.median(r_fin))

    # Bootstrap the null MEDIAN at the observed pair count, so the comparison is like-for-like.
    p_boot, boot_lo, boot_hi = bootstrap_aggregate_pvalue(
        med,
        nl,
        int(r_fin.size),
        agg=np.median,
        seed=args.seed,
    )
    # Spearman-Brown lifts the half-data reliability to the full (all-plate) delta check 1 targets.
    sb = 2 * med / (1 + med) if med > -1 else float("nan")
    summary = {
        "replicate_col": repl,
        "n_pairs": int(r_fin.size),
        "n_genes": len(hvg),
        "splithalf_median_r": round(med, 3),
        "splithalf_mean_r": round(float(np.mean(r_fin)), 3),
        "splithalf_q1_r": round(float(np.quantile(r_fin, 0.25)), 3),
        "splithalf_q3_r": round(float(np.quantile(r_fin, 0.75)), 3),
        "spearman_brown_full": round(sb, 3),
        "frac_pos": round(float(np.mean(r_fin > 0)), 3),
        "null_median_r": round(null_med, 3) if np.isfinite(null_med) else float("nan"),
        "null_n_draws": int(nl.size),
        "null_any_pair_r": round(float(np.median(nulls["any_pair"])), 3)
        if nulls["any_pair"].size
        else float("nan"),
        "null_diff_drug_r": round(float(np.median(nulls["diff_drug"])), 3)
        if nulls["diff_drug"].size
        else float("nan"),
        "null_same_drug_r": round(float(np.median(nulls["same_drug"])), 3)
        if nulls["same_drug"].size
        else float("nan"),
        # Same fix as p_vs_null below: the observed MEDIAN against the bootstrapped sampling
        # distribution of the same_drug null's MEDIAN, not against individual same_drug draws.
        # The two forms previously disagreed within this same summary -- p_vs_null used the
        # corrected form while this one used the defective one two lines away.
        "p_vs_same_drug": round(
            bootstrap_aggregate_pvalue(
                float(np.median(r_fin)),
                nulls["same_drug"],
                int(r_fin.size),
                agg=np.median,
                seed=args.seed,
            )[0],
            4,
        ),
        "lift_over_null": round(med - null_med, 3) if np.isfinite(null_med) else float("nan"),
        # p compares the observed MEDIAN against the bootstrapped sampling distribution of the
        # NULL MEDIAN. The first version compared the observed median against the spread of
        # INDIVIDUAL null draws, which is a category error: a median over ~1,300 pairs has a
        # standard error roughly sqrt(n) times tighter than a single draw, so that p was
        # inflated by more than an order of magnitude and made a reproducible ceiling look
        # like it had failed its own null.
        "p_vs_null": round(p_boot, 4) if np.isfinite(p_boot) else float("nan"),
        "null_median_ci_lo": round(boot_lo, 3) if np.isfinite(boot_lo) else float("nan"),
        "null_median_ci_hi": round(boot_hi, 3) if np.isfinite(boot_hi) else float("nan"),
        # Reported separately because it is a real quantity and answers a DIFFERENT question:
        # how much the two distributions overlap, i.e. what share of mismatched pairs reach the
        # typical matched pair. It is an effect size, never a significance test.
        "frac_null_draws_above_observed_median": (
            round(float(np.mean(nl >= med)), 3) if nl.size else float("nan")
        ),
    }
    out = Path(args.out) if Path(args.out).is_absolute() else repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out, index=False)
    _write_params_sidecar(out, args, extra={"n_pairs": int(r_fin.size)})
    print("\n=== delta reproducibility ceiling (real Tahoe delta, plate split-half) ===")
    for k, v in summary.items():
        print(f"  {k:22s} {v}")
    print(
        f"\nCheck-1 achieved r ~ 0.2; the ceiling is the split-half median ({med:.3f}) / "
        f"Spearman-Brown full-data ({sb:.3f}).\nwrote {out}"
    )


if __name__ == "__main__":
    main()
