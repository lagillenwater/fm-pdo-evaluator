#!/bin/bash
#SBATCH --job-name=soragni-score
#SBATCH --partition=amem
#SBATCH --qos=mem-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12           # amem's confirmed 240G/12cpu floor -- see 10's comment.
#SBATCH --mem=240G
#SBATCH --time=03:00:00              # reading the 82G L1000 Level-3 matrix dominates; the
                                      # supervised adapter fits (l1/l2) are cheap by comparison.
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
#
# Multi-checkpoint (2026-08-20): to score a second Stack checkpoint (matching 11's own
# CKPT/OUTDIR override pattern), pass GENDIR/AGGDIR/LABEL together so outputs don't
# clobber the cytokine-aligned (default-checkpoint) run's soragni_pathb_results.csv, e.g.
# after running 11 with
# CKPT=/scratch/alpine/$USER/sciplex_finetune/finetuned-epoch=5-val_loss=6.1078.ckpt
# OUTDIR=generated_soragni_drug_aligned:
#   sbatch --export=GENDIR=generated_soragni_drug_aligned,AGGDIR=generated_soragni_drug_aligned_agg,LABEL=drug_aligned \
#       scripts/alpine/12_soragni_score.sbatch
# LABEL unset (the default) reproduces the original unlabeled filenames.
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
LABEL="${LABEL:-}"                   # unset -> original unlabeled filenames (the cytokine-
                                      # aligned, default-checkpoint run). Set alongside
                                      # GENDIR/AGGDIR for any run whose generation used a
                                      # non-default CKPT, or it silently overwrites those files.
HALLMARK_SETS="${HALLMARK_SETS:-}"   # e.g. HALLMARK_E2F_TARGETS,HALLMARK_G2M_CHECKPOINT to
                                      # score proliferation-only instead of all four Hallmark
                                      # sets (see score_viability_adapters.py's --hallmark-sets).
THRESHOLD="${THRESHOLD:-0.0}"        # gen_logit < 0 -- same confidence-filter boundary the
                                      # Tahoe faithful-generation rerun calibrated (see
                                      # docs/tahoe_generation_results.md, Reproducibility);
                                      # not independently re-tuned for Soragni -- there is no
                                      # delta-fidelity ground truth here to calibrate against.

# ---- filename suffix: "" when LABEL is unset, "_<label>" otherwise -------------------------
SUFFIX=""
[[ -n "$LABEL" ]] && SUFFIX="_$LABEL"

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
summary.to_csv("soragni_aggregation_summary$SUFFIX.csv", index=False)
PY

# ---- fair source (additive/pca/nmf/stack) x readout-adapter (hallmark/l1/l2) grid -----------
HALLMARK_SETS_FLAG=()
if [[ -n "$HALLMARK_SETS" ]]; then
  HALLMARK_SETS_FLAG=(--hallmark-sets "$HALLMARK_SETS")
fi
python scripts/score_viability_adapters.py \
    --l1000-dir . \
    --gctx "$GCTX" \
    --generated-dir "$AGGDIR" \
    --baseline data/reference/stack_input_sarcoma.h5ad \
    "${HALLMARK_SETS_FLAG[@]}" \
    --out-csv "soragni_pathb_results$SUFFIX.csv"
