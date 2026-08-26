"""Rung 2, stage 2: score one (source, arm) cell of the transfer grid.

Both arms predict the SAME target -- Tahoe's measured delta for a Tahoe (line, drug) -- and are
scored the same way. Only where the map was fit differs:

  in_platform     5-fold on Tahoe itself, shared partition, no line's own delta in its own fold
  cross_platform  fit on L1000, applied to the Tahoe baseline
  bulk_target     5-fold on Tahoe (the SAME shared partition), predicting from the GDSC2 BULK
                  profile of each held-out line instead of its Tahoe pseudobulk -- the
                  baselines' analogue of Stack's 1G/2G granularity controls

so `cross_platform - in_platform` for a source is that source's transfer penalty, and the two
arms sharing one pinned panel is what makes the subtraction legitimate. `bulk_target` shares
in_platform's fold discipline for the same reason: a line's own Tahoe delta must never be in
the training fold that predicts it, regardless of which baseline (Tahoe pseudobulk or GDSC2
bulk) is queried at prediction time -- a version of this that fit on the FULL Tahoe set and
predicted the same lines' bulk profiles was leaking every target's own answer into its fit.

Stack is deliberately absent from this grid. It is not fitted here: it generates from frozen
weights given a query baseline and a drug, so it computes the identical thing in both arms and
its penalty is exactly zero by construction. Including it would put a constant beside numbers
that measure something, and invite reading its constancy as a win.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from fmharness.deltas import (
    build_additive_deltas,
    build_knn_deltas,
    build_learned_deltas,
    fold_assignment,
    shuffled_target_base,
)
from fmharness.statistics import bootstrap_aggregate_pvalue


def score_pairs(pred: pd.DataFrame, pred_key: pd.DataFrame, truth: pd.DataFrame,
                truth_key: pd.DataFrame, genes: pd.Index) -> tuple[list[float], list[str]]:
    """Per-(line, drug) Spearman between a predicted delta and the measured one."""
    tix = {(str(a), str(b)): i for i, (a, b) in enumerate(zip(truth_key["line"], truth_key["dname"], strict=True))}
    rhos, labels = [], []
    for i, (a, b) in enumerate(zip(pred_key["patient"], pred_key["drug"], strict=True)):
        j = tix.get((str(a), str(b)))
        if j is None:
            continue
        p = pred.iloc[i][genes].to_numpy(dtype=float)
        t = truth.iloc[j][genes].to_numpy(dtype=float)
        ok = np.isfinite(p) & np.isfinite(t)
        if ok.sum() < 50:
            continue
        rhos.append(float(stats.spearmanr(p[ok], t[ok]).statistic))
        labels.append(f"{a}|{b}")
    return rhos, labels


def main() -> None:
    """Fit one source under one arm and score it against the Tahoe truth."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan-dir", required=True, type=Path)
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--cell", default=None, help="source|arm, overrides --task-id")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--folds", type=int, default=5, help="shared with every rung; 5 is the invariant")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    plan = json.loads((args.plan_dir / "plan.json").read_text())
    grid = plan["grid"]
    if args.cell is None:
        if args.task_id is None or args.task_id >= len(grid):
            print(f"task {args.task_id} is past the grid ({len(grid)} cells); nothing to do")
            return
        args.cell = grid[args.task_id]
    source, _, arm = args.cell.partition("|")
    print(f"cell: source={source} arm={arm}")

    d = args.plan_dir
    t_delta = pd.read_parquet(d / "tahoe_delta.parquet")
    t_key = pd.read_parquet(d / "tahoe_key.parquet")
    t_base = pd.read_parquet(d / "tahoe_base.parquet")
    genes = pd.Index(t_delta.columns)
    targets = sorted({str(x) for x in t_key["line"]})
    tk_full = t_key.assign(patient=t_key["line"].astype(str), drug=t_key["dname"].astype(str))

    # ---- controls ---------------------------------------------------------------------
    # prior: predict each drug's mean over the TRAINING platform and nothing line-specific.
    # This is the floor every source must beat to have learned anything about lines, and it is
    # the row that reveals a source "transferring well" by only carrying drug identity.
    # shuffled: the real fitted map applied to a line-relabelled target baseline (a real OTHER
    # line's baseline under this line's label, via fmharness.deltas.shuffled_target_base), so
    # the model keeps all its capacity but loses the line correspondence. It must collapse to
    # the null.
    if source in ("prior", "shuffled"):
        src_for_fit = "observed_delta" if source == "prior" else "pca"
    elif source == "planted":
        src_for_fit = "pca"
    else:
        src_for_fit = source

    # POSITIVE CONTROL. Replace the truth with a delta that is a known function of the target
    # baseline, so a working pipeline must recover it under EVERY arm. Rung 2 is where this
    # matters most: every arm is expected to score low, so without something that must succeed
    # a grid of small numbers cannot separate "transfer is hard" from "this pipeline cannot fit
    # anything".
    #
    # Each DRUG gets its own random gene direction and its own random "drug mean" vector,
    # independent across drugs -- not one global direction shared by every row. The first
    # version used a single global direction, which made every row of planted_truth parallel to
    # it: ANY two rows (matched or mismatched, same drug or not) correlated at exactly +-1, so
    # the mismatched-pair null was itself ~1 in magnitude and the control could not "clear its
    # null" even in principle. Per-drug directions give mismatched-DIFFERENT-drug pairs a
    # near-zero expected correlation (a real floor) while same-drug pairs across lines stay
    # correlated through the shared per-drug direction -- the same drug-dominates-the-delta
    # structure real data has, which is exactly why the null below is diff_drug-stratified.
    #
    # The fit target must also be the planted delta, not the real one -- computed here and
    # substituted per arm below (only the SCORING truth was substituted before; every fit still
    # trained on the real t_delta/l1000_delta, so a working pipeline had no way to recover a
    # signal it was never shown).
    planted_truth = None
    _plant_signal = None
    _plant_delta = None
    if source == "planted":
        rngp = np.random.default_rng(args.seed + 11)
        n_g = len(genes)
        plant_w = rngp.normal(size=(n_g, 1))
        drugs_u = sorted({str(x) for x in t_key["dname"] if str(x)})
        plant_gene_load = {dr: rngp.normal(size=n_g) for dr in drugs_u}
        plant_drug_mean = {dr: rngp.normal(size=n_g) * 3.0 for dr in drugs_u}

        def _plant_signal(base: pd.DataFrame) -> dict[str, float]:
            raw = (base[genes].to_numpy(dtype=float) @ plant_w).ravel()
            z = (raw - raw.mean()) / (raw.std() or 1.0)
            return dict(zip(base.index.astype(str), z, strict=True))

        def _plant_delta(key: pd.DataFrame, sig: dict[str, float], line_col: str, drug_col: str) -> pd.DataFrame:
            zeros = np.zeros(n_g)
            rows = [
                plant_drug_mean.get(str(dr), zeros)
                + sig.get(str(ln), 0.0) * plant_gene_load.get(str(dr), zeros)
                for ln, dr in zip(key[line_col], key[drug_col], strict=True)
            ]
            return pd.DataFrame(np.asarray(rows, dtype=np.float64), columns=genes)

        tahoe_signal = _plant_signal(t_base)
        planted_truth = _plant_delta(
            t_key.assign(dname=t_key["dname"].astype(str)), tahoe_signal, "line", "dname"
        )

    def _folded_predictions(query_base: pd.DataFrame, restrict_targets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """5-fold in-platform fit on Tahoe; predicts each held-out fold's lines from `query_base`.

        Shared by in_platform (`query_base=t_base`) and bulk_target (`query_base=b_base`), so
        both honour the SAME held-out-line discipline: no line's own Tahoe delta is ever in the
        fold that fits its prediction, regardless of which baseline is queried at prediction
        time. This was leave-one-line-out while rung 3 ran 5-fold, so the transfer penalty --
        cross_platform minus in_platform -- mixed a bias/variance difference into what is meant
        to be a pure platform effect. Holding the fold map identical across rungs is what makes
        the subtraction mean one thing.
        """
        frames, keys = [], []
        fmap = fold_assignment(restrict_targets, args.folds)
        by_fold: dict[int, list[str]] = {}
        for ln in restrict_targets:
            by_fold.setdefault(fmap[ln], []).append(ln)
        print(f"  {len(by_fold)} folds over {len(restrict_targets)} lines (shared partition)")
        for _f in sorted(by_fold):
            group = by_fold[_f]
            hold = tk_full["patient"].isin(group)
            tr_delta_f, tr_key_f = t_delta[~hold.to_numpy()], tk_full[~hold]
            tr_base_f = t_base.drop(index=group, errors="ignore")
            tgt_base_f = query_base.loc[[ln for ln in group if ln in query_base.index]]
            if tgt_base_f.empty or tr_key_f.empty:
                continue
            gnames = list(tgt_base_f.index)
            if source in ("prior", "observed_delta"):
                p, kk = build_additive_deltas(tr_delta_f, tr_key_f, gnames)
            elif source == "shuffled":
                rngf = np.random.default_rng(args.seed + 7 + _f)
                sb = shuffled_target_base(query_base, gnames, restrict_targets, rngf)
                p, kk = build_learned_deltas(tr_base_f, tr_delta_f, tr_key_f, sb, gnames, reducer="pca", k=args.k)
            elif source == "knn":
                p, kk = build_knn_deltas(tr_base_f, tr_delta_f, tr_key_f, tgt_base_f, gnames, k=args.k)
            elif source == "planted":
                pt_train = planted_truth[~hold.to_numpy()]
                p, kk = build_learned_deltas(tr_base_f, pt_train, tr_key_f, tgt_base_f, gnames, reducer="pca", k=args.k)
            else:
                p, kk = build_learned_deltas(
                    tr_base_f, tr_delta_f, tr_key_f, tgt_base_f, gnames, reducer=src_for_fit, k=args.k
                )
            frames.append(p)
            keys.append(kk)
        if not frames:
            raise SystemExit(f"no scorable fold for {args.cell}")
        return pd.concat(frames, ignore_index=True), pd.concat(keys, ignore_index=True)

    if arm == "bulk_target":
        # Fit on Tahoe under the SAME 5-fold partition as in_platform, then predict each
        # held-out fold's lines from their GDSC2 BULK profile instead of their Tahoe
        # pseudobulk. Platform, drug and line are held fixed; only the profile's construction
        # varies, so the gap from in_platform is the cost of a bulk representation.
        bulk_path = d / "bulk_base.parquet"
        if not bulk_path.exists():
            print("bulk_base.parquet absent -- the plan skipped this arm; nothing to do")
            return
        b_base = pd.read_parquet(bulk_path)
        targets = [ln for ln in targets if ln in b_base.index]
        if not targets:
            raise SystemExit("no line has both a Tahoe pseudobulk and a GDSC2 bulk profile")
        print(f"bulk_target: {len(targets)} lines with a GDSC2 bulk profile")
        pred, pkey = _folded_predictions(b_base, targets)
    elif arm == "cross_platform":
        tr_delta = pd.read_parquet(d / "l1000_delta.parquet")
        tr_key = pd.read_parquet(d / "l1000_key.parquet")
        tr_base = pd.read_parquet(d / "l1000_base.parquet")
        if source == "prior":
            pred, pkey = build_additive_deltas(tr_delta, tr_key, targets)
        elif source == "shuffled":
            rng0 = np.random.default_rng(args.seed + 7)
            sb = shuffled_target_base(t_base, targets, targets, rng0)
            pred, pkey = build_learned_deltas(tr_base, tr_delta, tr_key, sb, targets, reducer="pca", k=args.k)
        elif source == "observed_delta":
            pred, pkey = build_additive_deltas(tr_delta, tr_key, targets)
        elif source == "knn":
            pred, pkey = build_knn_deltas(tr_base, tr_delta, tr_key, t_base, targets, k=args.k)
        elif source == "planted":
            l1000_signal = _plant_signal(tr_base)
            pt_l1000 = _plant_delta(
                tr_key.assign(drug=tr_key["drug"].astype(str)), l1000_signal, "patient", "drug"
            )
            pred, pkey = build_learned_deltas(tr_base, pt_l1000, tr_key, t_base, targets, reducer="pca", k=args.k)
        else:
            pred, pkey = build_learned_deltas(
                tr_base, tr_delta, tr_key, t_base, targets, reducer=src_for_fit, k=args.k
            )
    else:
        pred, pkey = _folded_predictions(t_base, targets)

    tk = t_key.assign(line=t_key["line"].astype(str), dname=t_key["dname"].astype(str))
    truth = planted_truth if planted_truth is not None else t_delta
    rhos, labels = score_pairs(pred, pkey, truth, tk, genes)
    if not rhos:
        raise SystemExit(f"no scorable pairs for {args.cell}")

    # Null: pair each prediction with a DIFFERENT-DRUG truth row (diff_drug stratum), not any
    # mismatched pair. An unstratified draw sometimes shares a drug by chance, and drug effects
    # dominate the delta, so that inflates the null toward the observed score and understates
    # any real signal -- the same defect rung 0's ceiling had (commit b7b1d72) before it was
    # stratified. p then compares the observed MEAN over n_pairs against the BOOTSTRAPPED
    # sampling distribution of that null's mean at the same pair count, not against the spread
    # of individual draws (fmharness.statistics.bootstrap_aggregate_pvalue) -- the aggregate-vs-
    # per-item bug fixed elsewhere this round.
    rng = np.random.default_rng(args.seed)
    tix = {(str(a), str(b)): i for i, (a, b) in enumerate(zip(tk["line"], tk["dname"], strict=True))}
    idx = list(tix.values())
    drug_of_pred = pkey["drug"].astype(str).to_numpy()
    drug_of_truth = tk["dname"].astype(str).to_numpy()
    null: list[float] = []
    tries, max_tries = 0, args.n_perm * 60
    while len(null) < args.n_perm and tries < max_tries and len(pred) > 0:
        tries += 1
        i = int(rng.integers(len(pred)))
        j = int(rng.choice(idx))
        if drug_of_pred[i] == drug_of_truth[j]:
            continue
        p = pred.iloc[i][genes].to_numpy(dtype=float)
        t = truth.iloc[j][genes].to_numpy(dtype=float)
        ok = np.isfinite(p) & np.isfinite(t)
        if ok.sum() >= 50:
            null.append(float(stats.spearmanr(p[ok], t[ok]).statistic))
    nl = np.asarray(null, dtype=float)
    r = np.asarray(rhos, dtype=float)
    p_vs_null, _, _ = bootstrap_aggregate_pvalue(float(r.mean()), nl, len(r), seed=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "source": source,
        "arm": arm,
        "n_pairs": int(len(r)),
        "mean_rho": round(float(r.mean()), 4),
        "sd_rho": round(float(r.std(ddof=1)), 4) if len(r) > 1 else float("nan"),
        "null_mean": round(float(nl.mean()), 4) if nl.size else float("nan"),
        "lift_over_null": round(float(r.mean() - nl.mean()), 4) if nl.size else float("nan"),
        "p_vs_null": round(p_vs_null, 4) if np.isfinite(p_vs_null) else float("nan"),
        "n_null_draws": int(nl.size),
        "panel_size": int(plan["panel_size"]),
        "n_genes_scored": int(len(genes)),
    }
    pd.DataFrame([row]).to_csv(args.out_dir / f"part_{source}__{arm}.csv", index=False)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
