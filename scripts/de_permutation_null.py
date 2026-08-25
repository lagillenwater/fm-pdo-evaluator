"""The Check-1b permutation null -- committed, and with the null that actually tests the claim.

`docs/tahoe_generation_results.md` publishes an eight-row null table and rests Finding 5 on it:
"real, statistically robust (patient, drug)-specific signal, p < 0.005, 7-45 null-SDs out". It
is the only evidence that reverses Check 1's null verdict. The script that produced it was never
committed, so nobody can check it or rerun it for a new checkpoint. This is that script.

It differs from the published version in two ways that matter.

**Two nulls, not one.** The published null shuffles ``pred_key``'s row order across the whole
frame. With ~32 drugs, a random reassignment lands on a DIFFERENT DRUG about 97% of the time,
so that null is dominated by drug mismatch and clearing it demonstrates drug-specificity --
the axis every representation already solves. It does NOT establish per-line specificity, and
the clean proof is that `additive`, which is line-independent by construction, would clear it
decisively: its prediction is keyed to the drug and nothing else. The null that tests the
published claim permutes patient labels WITHIN each drug, which holds drug identity fixed and
asks only whether the prediction is matched to the right line. Both are computed here; report
both, and read the within-drug one as the test of line-specificity.

**All sources, not just Stack.** The published table nulls only the two Stack checkpoints, then
compares Stack's lift-over-null against the baselines' RAW point estimates. Those are different
quantities. Every source is nulled here so the comparison is like-for-like.

Emits one row per (source, null_kind, metric). Promote with scripts/promote_result.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fmharness.evaluation import de_fidelity

METRICS = ("de_spearman_lfc", "pr_auc", "de_overlap_accuracy", "jaccard")


def shuffle_all(key: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Permute the whole key: breaks BOTH drug and line identity.

    This is the published null. Clearing it shows the prediction is matched to its (line, drug)
    pair better than to a random pair -- but since a random pair usually has the wrong drug, it
    is mostly a test of drug-specificity.
    """
    return key.iloc[rng.permutation(len(key))].reset_index(drop=True)


def shuffle_within_drug(key: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Permute patient labels among rows sharing a drug: breaks ONLY line identity.

    Drug identity is held fixed, so clearing this null is evidence of per-line specificity --
    the claim Check 1b is actually used to support.
    """
    out = key.copy().reset_index(drop=True)
    pat = out["patient"].to_numpy().copy()
    drg = out["drug"].to_numpy()
    for d in np.unique(drg):
        idx = np.flatnonzero(drg == d)
        pat[idx] = pat[rng.permutation(idx)]
    out["patient"] = pat
    return out


NULLS = {"shuffle_all": shuffle_all, "within_drug": shuffle_within_drug}


def metric_means(delta: pd.DataFrame, key: pd.DataFrame, de_calls: pd.DataFrame) -> dict[str, float]:
    """Mean of each DE metric over the matched pairs, or NaN when nothing matches."""
    try:
        f = de_fidelity(delta, key, de_calls)
    except ValueError:
        return dict.fromkeys(METRICS, float("nan"))
    return {m: float(f[m].mean()) for m in METRICS}


def main() -> None:
    """Score every delta source against both nulls and write a tidy table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources-dir", required=True, type=Path,
                    help="dir of <name>_delta.parquet / <name>_key.parquet pairs")
    ap.add_argument("--de-calls", required=True, type=Path, help="build_tahoe_de_calls output")
    ap.add_argument("--n-shuffles", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", required=True, type=Path)
    args = ap.parse_args()

    de_calls = pd.read_parquet(args.de_calls)
    names = sorted(p.name[: -len("_delta.parquet")] for p in args.sources_dir.glob("*_delta.parquet"))
    if not names:
        raise SystemExit(f"no *_delta.parquet under {args.sources_dir}")
    print(f"sources: {names}")

    rows: list[dict[str, object]] = []
    for name in names:
        delta = pd.read_parquet(args.sources_dir / f"{name}_delta.parquet")
        key = pd.read_parquet(args.sources_dir / f"{name}_key.parquet")
        observed = metric_means(delta, key, de_calls)
        for null_kind, fn in NULLS.items():
            draws = {m: [] for m in METRICS}
            for b in range(args.n_shuffles):
                rng = np.random.default_rng(args.seed + 1 + b)
                got = metric_means(delta, fn(key, rng), de_calls)
                for m in METRICS:
                    draws[m].append(got[m])
            for m in METRICS:
                a = np.asarray(draws[m], dtype=float)
                a = a[np.isfinite(a)]
                obs = observed[m]
                if a.size == 0 or not np.isfinite(obs):
                    continue
                rows.append({
                    "source": name,
                    "null_kind": null_kind,
                    "metric": m,
                    "observed": round(obs, 5),
                    "null_mean": round(float(a.mean()), 5),
                    "null_sd": round(float(a.std(ddof=1)), 5),
                    "null_p95": round(float(np.quantile(a, 0.95)), 5),
                    "specific_lift": round(float(obs - a.mean()), 5),
                    # (1 + count) / (1 + n): can never print 0, unlike the published table.
                    "p": round(float((1 + np.sum(a >= obs)) / (1 + a.size)), 5),
                    "n_draws": int(a.size),
                })
            print(f"  {name} / {null_kind}: done")

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv} ({len(out)} rows)")
    if not out.empty:
        piv = out.pivot_table(index=["source", "metric"], columns="null_kind", values="p")
        print("\np by null kind (within_drug is the test of LINE specificity):")
        print(piv.to_string())


if __name__ == "__main__":
    main()
