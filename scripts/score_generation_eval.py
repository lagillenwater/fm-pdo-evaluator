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

  Check 2 (cell-line end-to-end, leave-cell-line-out): predict measured viability (GDSC2 AUC) on
  the shared (DepMap line, drug) pairs, scored by global / per-drug / interaction Spearman (per-drug
  keeps the line main effect -- the literature-standard metric), a within-drug label-permutation
  null, and regret@k. Two designs on equal footing: (a) FIXED signature readouts -- ``hallmark``
  (all four Hallmark sets) and ``proliferation`` (only E2F + G2M, the sets that clear the gate) --
  applied to the delta sources; (b) a REPRESENTATION-CONTROLLED grid -- the untreated ``expr``
  baseline AND every delta source fed to the SAME penalized regressions (``l2`` Ridge, ``l1``
  Lasso, ``en`` elastic-net), fit per-drug on that representation, so a difference is the
  representation not the model (Kurilov 2020). A delta source earns its keep only if it beats
  ``expr``; ``--folds`` >= #lines makes the fit true leave-one-cell-line-out.

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
from collections.abc import Callable
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler

from fmharness.adapters import build_adapters
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
FIXED_READOUTS = ("hallmark", "proliferation")  # fixed-signature readouts, applied to delta sources
PENALTY_NAMES = ("l2", "l1", "en")  # penalized regressions for the representation-controlled grid


def _make_penalty(name: str) -> object:
    """A fresh ALPHA-CV-TUNED penalized model: l2=RidgeCV (efficient GCV), l1=LassoCV, en=
    ElasticNetCV (both inner 3-fold on the training lines). Tuning the penalty per representation
    makes the grid model-fair -- a fixed alpha over-/under-regularizes some representations and
    flips the ranking (Kurilov 2020)."""
    if name == "l2":
        return RidgeCV(alphas=np.logspace(-2, 3, 12))
    if name == "l1":
        return LassoCV(n_alphas=30, cv=3, max_iter=20000, random_state=0)
    if name == "en":
        return ElasticNetCV(l1_ratio=0.5, n_alphas=30, cv=3, max_iter=20000, random_state=0)
    raise ValueError(f"unknown penalty {name!r}")


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


def _load_line_matrix(path: Path) -> pd.DataFrame:
    """Load a per-cell-line feature matrix (index = line id) for the check-2 grid, from a
    ``.h5ad`` (X + obs_names), ``.parquet``, or ``.csv``. Used to fold a precomputed FM
    embedding (one vector per line) in head-to-head with expr/pca."""
    if path.suffix == ".h5ad":
        a = ad.read_h5ad(path)
        x = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        return pd.DataFrame(x, index=pd.Index([str(o) for o in a.obs_names])).astype(float)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, index_col=0)
    df.index = pd.Index([str(i) for i in df.index])
    return df.astype(float)


def _repr_by_drug(
    delta: pd.DataFrame, key: pd.DataFrame, genes: pd.Index
) -> dict[str, pd.DataFrame]:
    """Split a delta source into ``{drug: DataFrame[line x genes]}`` for per-drug regression."""
    d = delta.reindex(columns=genes).fillna(0.0)
    pat = key["patient"].astype(str).to_numpy()
    drg = key["drug"].astype(str).to_numpy()
    out: dict[str, pd.DataFrame] = {}
    for drug in pd.unique(drg):
        m = d[drg == drug]
        m.index = pd.Index(pat[drg == drug])
        out[str(drug)] = m
    return out


def _penalized_preds(
    feat: dict[str, pd.DataFrame] | Callable[[str], pd.DataFrame],
    design: pd.DataFrame,
    fold_of: dict[str, int],
    n_folds: int,
    uniq_lines: list[str],
    penalty: str,
    *,
    min_lines: int = 8,
    min_train: int = 5,
) -> pd.DataFrame:
    """Per-drug penalized regression (representation -> AUC), leave-cell-line-out by fold.

    ``feat`` maps a drug to a (line x gene) frame -- a dict for a delta source, or a callable for a
    drug-independent representation (baseline expression). For each drug the model is fit on the
    training-fold lines' features vs AUC and predicts the held-fold lines; the StandardScaler is fit
    on the training lines only, so a single held line (true LOO) is scored leakage-free. All
    representations share one model class, so a difference is the representation, not the model.
    Returns preds (patient, drug, y_true, y_pred); y_pred is an AUC estimate (same sign as y_true).
    """
    auc_by_drug = {
        str(d): dict(zip(g["patient"].astype(str), g["y"], strict=False))
        for d, g in design.groupby("drug")
    }
    rows: list[tuple[str, str, float, float, float]] = []
    for drug, auc in auc_by_drug.items():
        fdf = feat(drug) if callable(feat) else feat.get(drug)
        if fdf is None or fdf.empty:
            continue
        fdf = fdf.copy()
        fdf.index = pd.Index([str(i) for i in fdf.index])
        lines_d = [ln for ln in fdf.index if ln in auc]
        if len(lines_d) < min_lines:
            continue
        for f in range(n_folds):
            held = {ln for ln in uniq_lines if fold_of[ln] == f}
            tr = [ln for ln in lines_d if ln not in held]
            te = [ln for ln in lines_d if ln in held]
            if len(tr) < min_train or not te:
                continue
            sc = StandardScaler().fit(fdf.loc[tr].to_numpy(dtype=np.float64))
            model = _make_penalty(penalty).fit(
                sc.transform(fdf.loc[tr].to_numpy(dtype=np.float64)), [auc[ln] for ln in tr]
            )
            pred = model.predict(sc.transform(fdf.loc[te].to_numpy(dtype=np.float64)))
            # The potency prior: this drug's mean AUC over the training-fold lines, i.e. the
            # same fitted model with its coefficients zeroed. Ranking by it ignores the cell
            # line entirely, so it is the floor any line-specific claim has to clear.
            prior = float(np.mean([auc[ln] for ln in tr]))
            rows.extend(
                (ln, drug, float(auc[ln]), float(p), prior)
                for ln, p in zip(te, pred, strict=False)
            )
    cols = ["patient", "drug", "y_true", "y_pred", "y_prior"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


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
        default=",".join(FIXED_READOUTS),
        help="fixed-signature readouts on the delta sources (subset of hallmark, proliferation)",
    )
    ap.add_argument(
        "--penalties",
        default=",".join(PENALTY_NAMES),
        help="penalized regressions for the representation grid (subset of l2, l1, en)",
    )
    ap.add_argument(
        "--folds",
        type=int,
        default=5,
        help="grouped-by-cell-line folds for the leakage-free penalized fit; set >= #lines "
        "(e.g. 999) for true leave-one-cell-line-out",
    )
    ap.add_argument(
        "--preds-out",
        default="results/check2_preds.parquet",
        help="per-(line, drug) check-2 predictions dump; enables the selection audit",
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
    ap.add_argument(
        "--stack-emb",
        nargs="*",
        default=None,
        help="precomputed per-line FM embeddings to add as check-2 representations, each "
        "'label=path' (path .h5ad/.parquet/.csv, index/obs = cell line id). Repeatable, e.g. "
        "--stack-emb base=emb_base.h5ad aligned=emb_aligned.h5ad",
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

    # top-HVG panel: the supervised readouts fit on it and the learned (pca/nmf) sources reduce on
    # it, keeping the per-line PCA/NMF and readout fits fast and out of the hopeless p>>n regime.
    # The learned sources ALSO emit the fixed-signature genes: pca/nmf output a delta only over the
    # genes they are built on, so without the Hallmark genes a fixed readout would see an empty set
    # on those sources and score them zero (the bug that NaN'd pca/nmf x proliferation).
    hallmark = load_hallmark(repo / "data/static/hallmark_signatures.gmt")
    sig_genes = pd.Index(sorted({g for genes, _ in hallmark.values() for g in genes}))
    hvg = pd.Index(real_delta.var(axis=0).sort_values(ascending=False).index[: args.n_hvg])
    learned_genes = hvg.union(sig_genes)

    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "additive": _loo_baseline_source("additive", real_delta, real_key, base, k=args.k),
        "knn": _loo_baseline_source("knn", real_delta, real_key, base, k=args.k),
        "pca": _loo_baseline_source(
            "pca", real_delta, real_key, base, k=args.k, genes=learned_genes
        ),
        "nmf": _loo_baseline_source(
            "nmf", real_delta, real_key, base, k=args.k, genes=learned_genes
        ),
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

    # Labels for the gate and check 2 (hallmark was loaded above for the learned-source panel).
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

    # Check 2 -- predict AUC (leave-cell-line-out), two designs on equal footing:
    #  (a) FIXED signature readouts (hallmark, proliferation) on the delta sources -- the
    #      generation-through-death/proliferation-biology path (predict directly, no fitting).
    #  (b) REPRESENTATION-CONTROLLED penalized regression: the untreated baseline expression AND
    #      every delta source, each fed to the SAME L1/L2/elastic-net models (per-drug, fit on that
    #      representation), so a difference is the representation, not the model (Kurilov 2020). A
    #      delta source earns its keep only if it beats `expr`. --folds >= #lines gives true LOO.
    fixed_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    penalties = [p.strip() for p in args.penalties.split(",") if p.strip()]
    fixed_sigs = {
        "hallmark": hallmark,
        "proliferation": {n: hallmark[n] for n in PROLIFERATION if n in hallmark},
    }
    fixed_readouts = {
        m: build_adapters(["hallmark"], signatures=fixed_sigs[m])[0]
        for m in fixed_methods
        if m in fixed_sigs
    }
    uniq_lines = sorted(set(real_key["patient"].astype(str)))
    n_folds = max(1, min(args.folds, len(uniq_lines)))
    fold_of = {ln: i % n_folds for i, ln in enumerate(uniq_lines)}  # deterministic line -> fold
    target_drugs = set(real_key["drug"].astype(str))
    design_target = design[design["drug"].astype(str).isin(target_drugs)]

    def _row(s: dict[str, float]) -> dict[str, object]:
        return {
            "global": s["global"],
            "interaction": s["interaction"],
            "perdrug": s["perdrug"],
            "p_label": s["p_label"],
            "regret@1": s["regret@1"],
            "regret@3": s["regret@3"],
            "n": int(s["n"]),
        }

    out: list[dict[str, object]] = []

    # (a) fixed-signature readouts on the delta sources (sensitivity -> -y_pred vs AUC).
    for name, (d, kk) in sources.items():
        for method, adapter in fixed_readouts.items():
            sens = np.asarray(adapter.predict(d), dtype=float)
            merged = pd.DataFrame(
                {"patient": kk["patient"].to_numpy(), "drug": kk["drug"].to_numpy(), "_s": sens}
            ).merge(design.rename(columns={"y": "y_true"}), on=["patient", "drug"], how="inner")
            if merged.empty:
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
            out.append({"source": name, "method": method, **_row(s)})

    # (b) representation-controlled penalized regression: baseline expression + every delta source.
    base_hvg = base.reindex(columns=hvg).fillna(0.0)
    representations: dict[str, dict[str, pd.DataFrame] | Callable[[str], pd.DataFrame]] = {
        "expr": lambda _drug: base_hvg
    }
    for name, (d, kk) in sources.items():
        representations[name] = _repr_by_drug(d, kk, hvg)
    # precomputed FM embeddings (base / aligned Stack) as drug-independent representations --
    # one vector per line, scored in the SAME penalized grid as expr/pca (head-to-head). The base
    # checkpoint has no generation head, so this is how it enters the comparison at all.
    for spec in args.stack_emb or []:
        label, _, p = spec.partition("=")
        if not (label.strip() and p.strip()):
            ap.error(f"--stack-emb expects 'label=path', got {spec!r}")
        emb = _load_line_matrix(_rel(repo, p.strip()))
        representations[label.strip()] = (lambda e: lambda _drug: e)(emb)
    pred_frames: list[pd.DataFrame] = []
    for repr_name, feat in representations.items():
        for pen in penalties:
            preds = _penalized_preds(feat, design_target, fold_of, n_folds, uniq_lines, pen)
            if preds.empty:
                continue
            pred_frames.append(preds.assign(source=repr_name, method=pen))
            s = score_predictions(preds, n_perm=args.n_permutations)
            out.append({"source": repr_name, "method": pen, **_row(s)})

    if pred_frames:
        dest = _rel(repo, args.preds_out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        cols = ["source", "method", "patient", "drug", "y_true", "y_pred", "y_prior"]
        pd.concat(pred_frames, ignore_index=True)[cols].to_parquet(dest, index=False)
        print(f"wrote {dest} ({sum(len(f) for f in pred_frames)} rows)")

    print(f"\n=== check 2: end-to-end vs {args.auc_tranche} AUC (leave-cell-line-out) ===")
    print(pd.DataFrame(out).to_string(index=False) if out else "(no scored pairs)")


if __name__ == "__main__":
    main()
