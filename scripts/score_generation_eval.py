"""Score the Tahoe generation-mode eval: generation quality + the cell-line end-to-end check.

The Tahoe single-cell context (built on Alpine) gives, per (cell line, drug), a REAL
treated-minus-DMSO pseudobulk delta and a per-line baseline. Two checks run over the same
delta sources, so each source is judged on equal footing:

  Check 1 (generation quality, label-free): how faithfully a source reproduces the real
  Tahoe delta, per (cell line, drug) -- the per-pair delta-Pearson (Stack's own generation
  metric) plus the off-diagonal correlation and a specificity rank, which catch a source that
  is merely smooth (correlates with every condition) rather than specific.

  Gate (readout validity): the REAL per-(line, drug) delta scored through Hallmark vs AUC,
  with a random-gene-set negative control -- if it does not clear that control, the readout is
  underpowered on this data and a check-2 null is inconclusive.

  Check 2 (cell-line end-to-end, leave-cell-line-out): each source -> a readout -> a predicted
  sensitivity, scored against external MEASURED viability (GDSC2 AUC) on the shared (DepMap line,
  drug) pairs, by interaction rho with a within-drug label-permutation null and regret@k. Readouts:
  ``hallmark`` (fixed, all four sets), ``proliferation`` (fixed, only the E2F + G2M sets that clear
  the gate -- the death sets add noise), and the SUPERVISED ``szalai`` (L2 linear) and ``xgboost``,
  the latter two fit leakage-free by grouped-by-cell-line k-fold on the real delta vs AUC.

Delta sources form a ladder: ``additive`` (each drug's mean real delta, line-independent -- the
floor); ``knn`` (mean real delta of the lines whose baseline is nearest the held line); and
``pca``/``nmf`` (a baseline -> delta-residual map, learned on an HVG panel). All are rebuilt
leaving the scored line out, so a baseline never sees the held line's own treated cells.
Stack's generated delta joins the same ``sources`` map as a third rung: pass ``--generated-dir``
(the stack-generation output), ``--query-baseline`` (the AnnData fed to it as ``--test-adata``),
and ``--pert-map`` (pert_id -> CID, written by the context split). Its delta is
``logcpm(generated) - logcpm(query baseline)``, keyed (query line, drug CID), so it runs through
both checks identically to the baselines.

  # baselines only:
  PYTHONPATH=src python scripts/score_generation_eval.py --context tahoe_context.h5ad --k 10
  # + Stack:
  PYTHONPATH=src python scripts/score_generation_eval.py --deltas-bundle tahoe_deltas \\
      --generated-dir generated --query-baseline tahoe_query.h5ad \\
      --pert-map context_by_drug/pert_to_cid.tsv --k 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from fmharness.adapters import ALL_METHODS, build_adapters
from fmharness.data.loaders import load_tranche
from fmharness.deltas import (
    build_additive_deltas,
    build_generated_deltas,
    build_knn_deltas,
    build_learned_deltas,
    build_tahoe_deltas,
)
from fmharness.evaluation import build_sample_design, delta_fidelity, score_predictions
from fmharness.signatures import load_hallmark, score_signatures

# The Hallmark proliferation sets -- the two that cleared the gate's random-gene-set control
# (G2M clearly, E2F marginally); the death sets (P53, apoptosis) add only noise on Tahoe. A
# ``proliferation`` readout scores just these, so a real but weak signal is not diluted away.
PROLIFERATION = ("HALLMARK_E2F_TARGETS", "HALLMARK_G2M_CHECKPOINT")
DEFAULT_READOUTS = ("hallmark", "proliferation", *(m for m in ALL_METHODS if m != "hallmark"))


def _rel(repo: Path, p: str) -> Path:
    """Resolve ``p`` against the repo root unless it is already absolute."""
    q = Path(p)
    return q if q.is_absolute() else repo / q


def _load_pert_map(path: Path) -> dict[str, str]:
    """Read a ``pert_id<TAB>cid`` TSV into ``{pert_id: cid}`` for build_generated_deltas.

    The generated files are named by Tahoe pert_id (drug name); this maps each back to
    the PubChem CID the real deltas / designs are keyed by. Written by 03's context split.
    """
    m: dict[str, str] = {}
    for line in path.read_text().splitlines():
        pert, _, cid = line.partition("\t")
        if pert.strip() and cid.strip():
            m[pert.strip()] = cid.strip()
    return m


def _loo_baseline_source(
    kind: str,
    real_delta: pd.DataFrame,
    real_key: pd.DataFrame,
    base: pd.DataFrame,
    *,
    k: int,
    genes: pd.Index | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-cell-line-out baseline deltas: for each line, rebuild the source from the
    OTHER lines and predict the held-out line, so it never sees its own treated cells.

    ``additive``/``knn`` use all genes; ``pca``/``nmf`` (build_learned_deltas) reduce on the
    ``genes`` HVG panel, which keeps the per-line PCA/NMF fast and well-conditioned (49 lines
    vs ~50k genes is hopelessly p>>n; on ~2k informative genes it is sane)."""
    pats = real_key["patient"].astype(str).to_numpy()
    # for the learned sources, restrict the delta/baseline to the shared HVG panel up front.
    rdl = real_delta if genes is None else real_delta[[g for g in genes if g in real_delta.columns]]
    bl = base if genes is None else base[[g for g in rdl.columns if g in base.columns]]
    d_blocks: list[pd.DataFrame] = []
    k_blocks: list[pd.DataFrame] = []
    for line in [str(i) for i in base.index]:
        tr = pats != line
        if not tr.any():
            continue
        rd = real_delta[tr].reset_index(drop=True)
        rk = real_key[tr].reset_index(drop=True)
        if kind == "additive":
            d, kk = build_additive_deltas(rd, rk, [line])
        elif kind == "knn":
            d, kk = build_knn_deltas(base.drop(index=line), rd, rk, base.loc[[line]], [line], k=k)
        elif kind in ("pca", "nmf"):
            d, kk = build_learned_deltas(
                bl.drop(index=line),
                rdl[tr].reset_index(drop=True),
                rk,
                bl.loc[[line]],
                [line],
                reducer=kind,
            )
        else:
            raise ValueError(f"unknown baseline source {kind!r}")
        d_blocks.append(d)
        k_blocks.append(kk)
    if not d_blocks:
        raise ValueError(f"no held-out lines produced a {kind} delta")
    return pd.concat(d_blocks, ignore_index=True), pd.concat(k_blocks, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context", default=None, help="Tahoe context AnnData (build_tahoe_context)")
    ap.add_argument(
        "--deltas-bundle",
        default=None,
        help="dir with real_delta/real_key/base parquet (build_tahoe_pseudobulk_deltas shortcut)",
    )
    ap.add_argument("--auc-tranche", default="gdscv2", help="measured-AUC cohort for check 2")
    ap.add_argument("--k", type=int, default=10, help="neighbors for the k-NN source")
    ap.add_argument("--n-hvg", type=int, default=2000, help="top HVGs for the generation metric")
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument(
        "--n-random",
        type=int,
        default=200,
        help="random gene-set draws for the readout gate's negative control",
    )
    ap.add_argument(
        "--methods",
        default=",".join(DEFAULT_READOUTS),
        help="comma-separated readouts for check 2 (hallmark, proliferation, szalai, xgboost)",
    )
    ap.add_argument(
        "--readout-folds",
        type=int,
        default=5,
        help="grouped-by-cell-line folds for the LEAKAGE-FREE supervised readout fit",
    )
    ap.add_argument(
        "--generated-dir",
        default=None,
        help="dir of Stack-generated <pert_id>.h5ad treated files; adds the 'stack' source",
    )
    ap.add_argument(
        "--query-baseline",
        default=None,
        help="query AnnData given to stack-generation as --test-adata; the generated delta is "
        "generated - this baseline (required with --generated-dir)",
    )
    ap.add_argument(
        "--pert-map",
        default=None,
        help="TSV 'pert_id<TAB>cid' mapping generated pert_ids to PubChem CID "
        "(context split writes context_by_drug/pert_to_cid.tsv; required with --generated-dir)",
    )
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent

    if args.deltas_bundle:
        bdir = Path(args.deltas_bundle)
        bdir = bdir if bdir.is_absolute() else repo / bdir
        real_delta = pd.read_parquet(bdir / "real_delta.parquet")
        real_key = pd.read_parquet(bdir / "real_key.parquet")
        base = pd.read_parquet(bdir / "base.parquet")
    elif args.context:
        ctx = Path(args.context) if Path(args.context).is_absolute() else repo / args.context
        real_delta, real_key, base = build_tahoe_deltas(ad.read_h5ad(ctx))
    else:
        ap.error("provide --context (single-cell) or --deltas-bundle (pseudobulk shortcut)")
    print(
        f"Tahoe: {len(real_key)} (line, drug) pairs over {base.shape[0]} lines, "
        f"{real_delta.shape[1]} genes"
    )

    # top-HVG panel: shared by the learned (pca/nmf) sources and the supervised readouts, to keep
    # the per-line PCA/NMF and the readout fits fast and out of the hopeless p>>n regime.
    hvg = pd.Index(real_delta.var(axis=0).sort_values(ascending=False).index[: args.n_hvg])

    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "additive": _loo_baseline_source("additive", real_delta, real_key, base, k=args.k),
        "knn": _loo_baseline_source("knn", real_delta, real_key, base, k=args.k),
        "pca": _loo_baseline_source("pca", real_delta, real_key, base, k=args.k, genes=hvg),
        "nmf": _loo_baseline_source("nmf", real_delta, real_key, base, k=args.k, genes=hvg),
    }
    # Stack's generated delta joins the same ladder when a generation run is supplied:
    # delta = logcpm(generated) - logcpm(query baseline), keyed (query line, drug CID). It
    # then flows through both checks exactly like the baselines, on equal footing.
    if args.generated_dir:
        if not (args.query_baseline and args.pert_map):
            ap.error("--generated-dir requires --query-baseline and --pert-map")
        sources["stack"] = build_generated_deltas(
            _rel(repo, args.generated_dir),
            _rel(repo, args.query_baseline),
            _load_pert_map(_rel(repo, args.pert_map)),
        )

    # Check 1 -- generation quality vs the real Tahoe delta.
    fid_rows: list[dict[str, object]] = []
    for name, (d, kk) in sources.items():
        f = delta_fidelity(d, kk, real_delta, real_key, n_hvg=args.n_hvg)
        fid_rows.append(
            {
                "source": name,
                "r": round(float(f["r"].mean()), 3),
                "r_offdiag": round(float(f["r_offdiag"].mean()), 3),
                "rank": round(float(f["rank"].mean()), 3),
                "n_pairs": len(f),
                "n_genes": int(f["n_genes"].iloc[0]),
            }
        )
    print("\n=== check 1: generation quality (delta-Pearson vs real Tahoe) ===")
    print(pd.DataFrame(fid_rows).to_string(index=False))

    # Readout + labels shared by the gate and check 2.
    hallmark = load_hallmark(repo / "data/static/hallmark_signatures.gmt")
    _, design = build_sample_design(
        load_tranche(args.auc_tranche, repo), "all", "auc", drug_key="pubchem_cid"
    )

    # Gate -- is the Hallmark readout even powered on this data? Score the REAL per-(line, drug)
    # Tahoe delta (the best-case input, not a reconstruction) through Hallmark vs measured AUC,
    # with a same-size random-gene-set negative control. If a signature's interaction does not
    # clear rnd_p95 (p_vs_random not small), the readout is a generic magnitude detector on Tahoe,
    # not death biology -- so a null on the generated deltas in check 2 is inconclusive, and check
    # 1 (label-free fidelity) is the more trustworthy axis.
    gate = score_signatures(
        real_delta,
        real_key,
        design,
        signatures=hallmark,
        n_perm=args.n_permutations,
        n_random=args.n_random,
    )
    print("\n=== gate: Hallmark readout on the REAL Tahoe delta (vs random gene sets) ===")
    print(gate.to_string(index=False) if not gate.empty else "(no (line, drug) overlap with AUC)")

    # Check 2 -- each source -> readout -> predicted sensitivity vs measured AUC. hallmark and
    # proliferation are FIXED signatures (predict directly). szalai/xgboost are SUPERVISED and fit
    # LEAKAGE-FREE by grouped-by-cell-line k-fold on the REAL Tahoe delta vs AUC (HVG panel): a
    # fold's lines are held out of that fold's fit, so the readout never sees a line's own (delta,
    # viability) before scoring any source's reconstruction of it. Predicting a whole fold at once
    # keeps the readout's per-cohort z-score stable (a single line's ~32 rows would be too few).
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    fixed_sigs = {
        "hallmark": hallmark,
        "proliferation": {n: hallmark[n] for n in PROLIFERATION if n in hallmark},
    }
    fixed_readouts = {
        m: build_adapters(["hallmark"], signatures=fixed_sigs[m])[0]
        for m in methods
        if m in fixed_sigs
    }
    supervised = [m for m in methods if m not in fixed_sigs]

    real_lines = real_key["patient"].astype(str).to_numpy()
    real_via = real_key.merge(
        design.rename(columns={"y": "_y"}), on=["patient", "drug"], how="left"
    )["_y"].to_numpy()
    rd_panel = real_delta.reindex(columns=hvg).fillna(0.0)
    uniq_lines = sorted(set(real_lines))
    n_folds = max(1, min(args.readout_folds, len(uniq_lines)))
    fold_of = {ln: i % n_folds for i, ln in enumerate(uniq_lines)}  # deterministic line -> fold

    # one supervised readout per (method, fold), fit on the lines NOT in that fold.
    fold_readouts: dict[tuple[str, int], object] = {}
    for f in range(n_folds):
        held = {ln for ln in uniq_lines if fold_of[ln] == f}
        tr = (~pd.Series(real_lines).isin(held).to_numpy()) & ~np.isnan(real_via)
        if int(tr.sum()) < 5:
            continue
        for adapter in build_adapters(supervised, signatures=None):
            adapter.fit(rd_panel[tr], real_via[tr])
            fold_readouts[(adapter.name, f)] = adapter

    def _sensitivity(method: str, sdelta: pd.DataFrame, skey: pd.DataFrame) -> np.ndarray:
        """Per-source-row sensitivity through one readout. Fixed: predict directly. Supervised:
        each held row gets its fold's leakage-free model, but z-scored over the WHOLE source (one
        ``predict(sp)`` per fold), so folds share one normalization and cross-fold ranks compare."""
        if method in fixed_readouts:
            return np.asarray(fixed_readouts[method].predict(sdelta), dtype=float)
        sp = sdelta.reindex(columns=hvg).fillna(0.0)
        lines = skey["patient"].astype(str).to_numpy()
        sens = np.full(len(sdelta), np.nan)
        for f in range(n_folds):
            model = fold_readouts.get((method, f))
            if model is None:
                continue
            held = {ln for ln in uniq_lines if fold_of[ln] == f}
            mask = pd.Series(lines).isin(held).to_numpy()
            if mask.any():
                sens[mask] = model.predict(sp)[mask]  # whole-source z-score; fold's held-out model
        return sens

    out: list[dict[str, object]] = []
    for name, (d, kk) in sources.items():
        for method in methods:
            sens = _sensitivity(method, d, kk)
            valid = ~np.isnan(sens)
            merged = pd.DataFrame(
                {
                    "patient": kk["patient"].to_numpy()[valid],
                    "drug": kk["drug"].to_numpy()[valid],
                    "_s": sens[valid],
                }
            ).merge(design.rename(columns={"y": "y_true"}), on=["patient", "drug"], how="inner")
            if merged.empty:
                print(f"  [{name}/{method}] no (line, drug) overlap with {args.auc_tranche}")
                continue
            preds = pd.DataFrame(
                {
                    "patient": merged["patient"],
                    "drug": merged["drug"],
                    "y_true": merged["y_true"].to_numpy(),
                    "y_pred": -merged["_s"].to_numpy(),
                }
            )
            s = score_predictions(preds, n_perm=args.n_permutations)
            out.append(
                {
                    "source": name,
                    "method": method,
                    "global": s["global"],
                    "interaction": s["interaction"],
                    "p_label": s["p_label"],
                    "regret@1": s["regret@1"],
                    "regret@3": s["regret@3"],
                    "n": int(s["n"]),
                }
            )
    print(f"\n=== check 2: end-to-end vs {args.auc_tranche} AUC (leave-cell-line-out) ===")
    print(pd.DataFrame(out).to_string(index=False) if out else "(no scored pairs)")


if __name__ == "__main__":
    main()
