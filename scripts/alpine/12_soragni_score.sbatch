#!/bin/bash
#SBATCH --job-name=soragni-score
#SBATCH --partition=amem
#SBATCH --qos=mem-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12           # amem's confirmed 240G/12cpu floor -- see 10's comment.
#SBATCH --mem=240G
#SBATCH --time=03:00:00              # reading the 82G L1000 Level-3 matrix dominates; the
                                      # supervised adapter fits (szalai/xgboost) are cheap by
                                      # comparison.
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Path B final step: reduce 11's per-replicate generated output to one row per (patient,
# drug) (Task-3-style confidence filtering, same fmharness.stack_aggregate module Tahoe
# uses), then score the fair source x readout-adapter grid against real Soragni viability.
#
# --baseline is stack_input_sarcoma.h5ad -- the SAME tumor-RNA file 10_soragni_context's
# build_soragni_query_pool.py used to seed generation. Fixed 2026-08-20: this script
# previously passed score_viability_adapters.py's own stale default,
# stack_input_soragni.h5ad -- a pre-"use tumor RNA as the Soragni model input" (commit
# 16f42df) organoid-RNA artifact never updated to match. Confirmed materially different
# data (correlation as low as 0.57 on shared patient IDs, not just a naming/subset gap):
# using it silently subtracted a mismatched-substrate baseline from every non-additive
# delta source (pca/nmf/stack all take --baseline; only additive doesn't).
# CONFIRM BEFORE RUNNING:
#   * GCTX matches the file already staged in $REPO (see build_l1000_context.py's docstring
#     for the download recipe if it's ever missing).
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

module load anaconda
conda activate stack
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1

GCTX="${GCTX:-GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx}"
GENDIR="${GENDIR:-generated_soragni_faithful}"
AGGDIR="${AGGDIR:-generated_soragni_faithful_agg}"
THRESHOLD="${THRESHOLD:-0.0}"        # gen_logit < 0 -- same confidence-filter boundary the
                                      # Tahoe faithful-generation rerun calibrated (see
                                      # docs/tahoe_generation_results.md, Reproducibility);
                                      # not independently re-tuned for Soragni -- there is no
                                      # delta-fidelity ground truth here to calibrate against.

# ---- Task-3-style confidence-filtered aggregation: 408 replicates/drug -> <=17 rows/drug --
python - <<PY
from pathlib import Path

from fmharness.stack_aggregate import aggregate_generated_replicates

summary = aggregate_generated_replicates(
    Path("$GENDIR"), Path("$AGGDIR"), threshold=$THRESHOLD
)
n_drugs = summary["pert_id"].nunique()
dropped = int(summary["dropped"].sum())
total = len(summary)
print(
    f"aggregated {n_drugs} drugs, {total} (patient, drug) cells, "
    f"{dropped} dropped (zero replicates survived threshold=$THRESHOLD) -> $AGGDIR/"
)
summary.to_csv("soragni_aggregation_summary.csv", index=False)
PY

# ---- fair source (additive/pca/nmf/stack) x readout-adapter (hallmark/szalai/xgboost) grid -
python scripts/score_viability_adapters.py \
    --l1000-dir . \
    --gctx "$GCTX" \
    --generated-dir "$AGGDIR" \
    --baseline data/reference/stack_input_sarcoma.h5ad \
    --out-csv soragni_pathb_results.csv
