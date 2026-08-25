"""Is the ``gen_logit < 0`` replicate filter defensible, or just the value someone picked?

`docs/tahoe_generation_results.md` says the threshold was "calibrated empirically against this
Check-1 Pearson-Delta itself". The pointer it gives leads to an unchecked to-do box in an
implementation plan describing a sweep to run -- no thresholds, no r values, no chosen value,
no output. So the filter that decides which generated replicates enter EVERY Stack row, in both
the Tahoe and Soragni arms, currently rests on nothing a reader can check.

This runs the sweep the doc claims was already run. For each candidate threshold it aggregates
the replicates, builds the generated delta, and scores Check 1's Pearson-Delta against the real
delta -- the same quantity the calibration claim names.

Two outcomes, and BOTH are useful:

- Results barely move across the range. Then the threshold is not doing real work, the exact
  value never mattered, and the defence is the sweep table itself rather than an appeal to a
  calibration nobody can find.
- Results move a lot. Then the reported Stack numbers are a function of an unjustified choice,
  and that is a finding about the published results, not a tuning detail.

Either way the answer is a committed table instead of a sentence. Promote it with
scripts/promote_result.py and cite it wherever the threshold is mentioned.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fmharness.deltas import build_generated_deltas, load_pert_map
from fmharness.evaluation import delta_fidelity
from fmharness.stack_aggregate import aggregate_generated_replicates

# WHY these candidates: 0.0 is the shipped value; the rest bracket it by an order of magnitude
# either side plus "no filtering at all", which is the honest reference point -- if unfiltered
# scores the same, the filter earns nothing.
THRESHOLDS = (float("inf"), 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0)


def main() -> None:
    """Sweep the confidence threshold and report Check-1 fidelity at each value."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generated-dir", required=True, type=Path)
    ap.add_argument("--query-baseline", required=True, type=Path)
    ap.add_argument("--pert-map", required=True, type=Path)
    ap.add_argument("--deltas-bundle", required=True, type=Path)
    ap.add_argument("--work-dir", type=Path, default=Path("/tmp/gen_logit_sweep"))
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--out-csv", required=True, type=Path)
    args = ap.parse_args()

    real_delta = pd.read_parquet(args.deltas_bundle / "real_delta.parquet")
    real_key = pd.read_parquet(args.deltas_bundle / "real_key.parquet")
    # build_generated_deltas takes the MAPPING, not the path -- mirror the driver.
    pert_map = load_pert_map(args.pert_map)

    rows: list[dict[str, object]] = []
    for thr in THRESHOLDS:
        label = "none" if np.isinf(thr) else f"{thr:g}"
        agg_dir = args.work_dir / f"thr_{label}"
        agg_dir.mkdir(parents=True, exist_ok=True)
        # Returns a per-file summary frame; threshold is keyword-only.
        summary = aggregate_generated_replicates(args.generated_dir, agg_dir, threshold=thr)
        gen_delta, gen_key = build_generated_deltas(agg_dir, args.query_baseline, pert_map)
        fid = delta_fidelity(gen_delta, gen_key, real_delta, real_key, n_hvg=args.n_hvg)
        rows.append({
            "threshold": label,
            "pairs": int(len(fid)),
            "files_aggregated": int(len(summary)),
            "r": round(float(fid["r"].mean()), 5),
            "r_offdiag": round(float(fid["r_offdiag"].mean()), 5),
            "rank": round(float(fid["rank"].mean()), 5),
        })
        print(f"  threshold {label:>5}: r={rows[-1]['r']:+.4f} over {rows[-1]['pairs']} pairs")

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv}")
    print(out.to_string(index=False))
    span = float(out["r"].max() - out["r"].min())
    print(f"\nr spans {span:.4f} across the swept range.")
    print("  If that is small relative to the gap between Stack (~0.01) and the additive floor")
    print("  (0.225), the threshold is not what makes Stack null and the sweep IS the defence.")


if __name__ == "__main__":
    main()
