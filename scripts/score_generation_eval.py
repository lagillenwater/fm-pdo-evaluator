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

from fmharness.adapters import build_adapters
from fmharness.check2 import (
    FIXED_READOUTS,
    PENALTY_NAMES,
    PROLIFERATION,
    load_line_matrix,
    penalized_preds,
    repr_by_drug,
)
from fmharness.data.loaders import load_tranche
from fmharness.deltas import (
    build_generated_deltas,
    build_tahoe_deltas,
    learned_gene_panel,
    load_pert_map,
    loo_baseline_source,
)
from fmharness.evaluation import build_sample_design, score_delta_sources, score_predictions
from fmharness.signatures import load_hallmark, score_signatures


def _rel(repo: Path, p: str) -> Path:
    """Resolve ``p`` against the repo root unless it is already absolute."""
    q = Path(p)
    return q if q.is_absolute() else repo / q


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
    hvg = pd.Index(real_delta.var(axis=0).sort_values(ascending=False).index[: args.n_hvg])
    learned_genes = learned_gene_panel(
        real_delta, repo / "data/static/hallmark_signatures.gmt", n_hvg=args.n_hvg
    )

    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "additive": loo_baseline_source("additive", real_delta, real_key, base, k=args.k),
        "knn": loo_baseline_source("knn", real_delta, real_key, base, k=args.k),
        "pca": loo_baseline_source(
            "pca", real_delta, real_key, base, k=args.k, genes=learned_genes
        ),
        "nmf": loo_baseline_source(
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
            load_pert_map(_rel(repo, args.pert_map)),
        )

    # Check 1 -- generation quality vs the real Tahoe delta.
    fid_table = score_delta_sources(sources, real_delta, real_key, n_hvg=args.n_hvg)
    print("\n=== check 1: generation quality (delta-Pearson vs real Tahoe) ===")
    print(fid_table.to_string(index=False))

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
        representations[name] = repr_by_drug(d, kk, hvg)
    # precomputed FM embeddings (base / aligned Stack) as drug-independent representations --
    # one vector per line, scored in the SAME penalized grid as expr/pca (head-to-head). The base
    # checkpoint has no generation head, so this is how it enters the comparison at all.
    for spec in args.stack_emb or []:
        label, _, p = spec.partition("=")
        if not (label.strip() and p.strip()):
            ap.error(f"--stack-emb expects 'label=path', got {spec!r}")
        emb = load_line_matrix(_rel(repo, p.strip()))
        representations[label.strip()] = (lambda e: lambda _drug: e)(emb)
    for repr_name, feat in representations.items():
        for pen in penalties:
            preds = penalized_preds(feat, design_target, fold_of, n_folds, uniq_lines, pen)
            if preds.empty:
                continue
            s = score_predictions(preds, n_perm=args.n_permutations)
            out.append({"source": repr_name, "method": pen, **_row(s)})

    print(f"\n=== check 2: end-to-end vs {args.auc_tranche} AUC (leave-cell-line-out) ===")
    print(pd.DataFrame(out).to_string(index=False) if out else "(no scored pairs)")


if __name__ == "__main__":
    main()
