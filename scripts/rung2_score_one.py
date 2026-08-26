"""Rung 2, stage 2: score one (source, arm) cell of the transfer grid.

Both arms predict the SAME target -- Tahoe's measured delta for a Tahoe (line, drug) -- and are
scored the same way. Only where the map was fit differs:

  in_platform     fit leave-one-line-out on Tahoe itself
  cross_platform  fit on L1000, applied to the Tahoe baseline

so `cross_platform - in_platform` for a source is that source's transfer penalty, and the two
arms sharing one pinned panel is what makes the subtraction legitimate.

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
)


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

    # ---- controls ---------------------------------------------------------------------
    # prior: predict each drug's mean over the TRAINING platform and nothing line-specific.
    # This is the floor every source must beat to have learned anything about lines, and it is
    # the row that reveals a source "transferring well" by only carrying drug identity.
    # shuffled: the real fitted map applied to a line-permuted target baseline, so the model
    # keeps all its capacity but loses the line correspondence. It must collapse to the null.
    if source in ("prior", "shuffled"):
        src_for_fit = "observed_delta" if source == "prior" else "pca"
    elif source == "planted":
        src_for_fit = "pca"
    else:
        src_for_fit = source

    # POSITIVE CONTROL. Replace the truth with a delta that is a known linear function of the
    # target baseline, so a working pipeline must recover it under EVERY arm. Rung 2 is where
    # this matters most: every arm is expected to score low, so without something that must
    # succeed a grid of small numbers cannot separate "transfer is hard" from "this pipeline
    # cannot fit anything".
    #
    # Initialised unconditionally. The previous edit assigned it only inside the planted branch
    # and every other cell died on NameError at the point of use -- the whole array failed in 18
    # seconds because a control was bolted on rather than threaded through.
    planted_truth = None
    if source == "planted":
        rngp = np.random.default_rng(args.seed + 11)
        W = rngp.normal(size=(t_base.shape[1], 1))
        signal = (t_base.to_numpy(dtype=float) @ W).ravel()
        signal = (signal - signal.mean()) / (signal.std() or 1.0)
        by_line = dict(zip(t_base.index.astype(str), signal, strict=True))
        gene_load = rngp.normal(size=t_delta.shape[1])
        planted_truth = pd.DataFrame(
            np.outer([by_line.get(str(x), 0.0) for x in t_key["line"]], gene_load),
            columns=t_delta.columns,
        )

    if arm == "bulk_target":
        # Fit in-platform on Tahoe, then predict from the GDSC2 BULK profile of the same line.
        # Platform, drug and line are held fixed; only the profile's construction changes, so
        # the gap from in_platform is the cost of a bulk representation rather than a pseudobulk
        # one -- the baselines' analogue of Stack's 1G/2G.
        bulk_path = d / "bulk_base.parquet"
        if not bulk_path.exists():
            print("bulk_base.parquet absent -- the plan skipped this arm; nothing to do")
            return
        b_base = pd.read_parquet(bulk_path)
        targets = [ln for ln in targets if ln in b_base.index]
        if not targets:
            raise SystemExit("no line has both a Tahoe pseudobulk and a GDSC2 bulk profile")
        tk0 = t_key.assign(patient=t_key["line"].astype(str), drug=t_key["dname"].astype(str))
        if source in ("prior", "observed_delta"):
            pred, pkey = build_additive_deltas(t_delta, tk0, targets)
        elif source == "knn":
            pred, pkey = build_knn_deltas(t_base, t_delta, tk0, b_base.loc[targets], targets, k=args.k)
        else:
            pred, pkey = build_learned_deltas(
                t_base, t_delta, tk0, b_base.loc[targets], targets, reducer=src_for_fit, k=args.k
            )
    elif arm == "cross_platform":
        tr_delta = pd.read_parquet(d / "l1000_delta.parquet")
        tr_key = pd.read_parquet(d / "l1000_key.parquet")
        tr_base = pd.read_parquet(d / "l1000_base.parquet")
        if source == "prior":
            pred, pkey = build_additive_deltas(tr_delta, tr_key, targets)
        elif source == "shuffled":
            rng0 = np.random.default_rng(args.seed + 7)
            shuffled_base = t_base.copy()
            shuffled_base.index = pd.Index(rng0.permutation(list(t_base.index)))
            pred, pkey = build_learned_deltas(
                tr_base, tr_delta, tr_key, shuffled_base.loc[targets], targets, reducer="pca", k=args.k
            )
        elif source == "observed_delta":
            pred, pkey = build_additive_deltas(tr_delta, tr_key, targets)
        elif source == "knn":
            pred, pkey = build_knn_deltas(tr_base, tr_delta, tr_key, t_base, targets, k=args.k)
        else:
            pred, pkey = build_learned_deltas(
                tr_base, tr_delta, tr_key, t_base, targets, reducer=src_for_fit, k=args.k
            )
    else:
        # 5-fold on Tahoe, using the SHARED partition. This was leave-one-line-out while rung 3
        # ran 5-fold, so the transfer penalty -- cross_platform minus in_platform -- mixed a
        # bias/variance difference into what is meant to be a pure platform effect. Holding the
        # fold map identical across rungs is what makes the subtraction mean one thing.
        frames, keys = [], []
        tk = t_key.assign(patient=t_key["line"].astype(str), drug=t_key["dname"].astype(str))
        fmap = fold_assignment(targets, args.folds)
        by_fold: dict[int, list[str]] = {}
        for ln in targets:
            by_fold.setdefault(fmap[ln], []).append(ln)
        print(f"in_platform: {len(by_fold)} folds over {len(targets)} lines (shared partition)")
        for _f in sorted(by_fold):
            group = by_fold[_f]
            line = group[0]
            hold = tk["patient"].isin(group)
            tr_delta, tr_key = t_delta[~hold.to_numpy()], tk[~hold]
            tr_base = t_base.drop(index=group, errors="ignore")
            tgt_base = t_base.loc[[ln for ln in group if ln in t_base.index]]
            if tgt_base.empty or tr_key.empty:
                continue
            if source in ("prior", "observed_delta"):
                p, kk = build_additive_deltas(tr_delta, tr_key, list(tgt_base.index))
            elif source == "shuffled":
                rng0 = np.random.default_rng(args.seed + 7)
                # One label per held-out line. This assigned a single label to a frame that
                # holds a whole FOLD once rung 2 moved from leave-one-out to 5-fold, so the
                # index length stopped matching the row count and the cell died on a shape
                # error -- a control written for one CV scheme and left behind by the other.
                sb = tgt_base.copy()
                sb.index = pd.Index(
                    [str(x) for x in rng0.choice(list(t_base.index), size=len(sb), replace=False)]
                )
                p, kk = build_learned_deltas(
                    tr_base, tr_delta, tr_key, sb, [line], reducer="pca", k=args.k
                )
                kk = kk.assign(patient=line)
            elif source == "knn":
                p, kk = build_knn_deltas(
                    tr_base, tr_delta, tr_key, tgt_base, list(tgt_base.index), k=args.k
                )
            else:
                p, kk = build_learned_deltas(
                    tr_base, tr_delta, tr_key, tgt_base, list(tgt_base.index),
                    reducer=src_for_fit, k=args.k
                )
            frames.append(p)
            keys.append(kk)
        pred = pd.concat(frames, ignore_index=True)
        pkey = pd.concat(keys, ignore_index=True)

    tk = t_key.assign(line=t_key["line"].astype(str), dname=t_key["dname"].astype(str))
    truth = planted_truth if planted_truth is not None else t_delta
    rhos, labels = score_pairs(pred, pkey, truth, tk, genes)
    if not rhos:
        raise SystemExit(f"no scorable pairs for {args.cell}")

    # Null: pair each prediction with a DIFFERENT pair's truth, so the floor from shared
    # gene-level structure is measured under this arm rather than assumed to be zero.
    rng = np.random.default_rng(args.seed)
    tix = {(str(a), str(b)): i for i, (a, b) in enumerate(zip(tk["line"], tk["dname"], strict=True))}
    null = []
    idx = list(tix.values())
    for _ in range(args.n_perm):
        i = rng.integers(len(pred))
        j = int(rng.choice(idx))
        p = pred.iloc[int(i)][genes].to_numpy(dtype=float)
        t = truth.iloc[j][genes].to_numpy(dtype=float)
        ok = np.isfinite(p) & np.isfinite(t)
        if ok.sum() >= 50:
            null.append(float(stats.spearmanr(p[ok], t[ok]).statistic))
    nl = np.asarray(null, dtype=float)
    r = np.asarray(rhos, dtype=float)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "source": source,
        "arm": arm,
        "n_pairs": int(len(r)),
        "mean_rho": round(float(r.mean()), 4),
        "sd_rho": round(float(r.std(ddof=1)), 4) if len(r) > 1 else float("nan"),
        "null_mean": round(float(nl.mean()), 4) if nl.size else float("nan"),
        "lift_over_null": round(float(r.mean() - nl.mean()), 4) if nl.size else float("nan"),
        "p_vs_null": round(float((1 + np.sum(nl >= r.mean())) / (1 + nl.size)), 4) if nl.size else float("nan"),
        "panel_size": int(plan["panel_size"]),
        "n_genes_scored": int(len(genes)),
    }
    pd.DataFrame([row]).to_csv(args.out_dir / f"part_{source}__{arm}.csv", index=False)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
