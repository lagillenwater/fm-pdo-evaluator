#!/bin/bash
#SBATCH --job-name=sarcoma_organoids_2024-score
#SBATCH --partition=amem
#SBATCH --qos=mem-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12           # amem 240G/12cpu floor [checked: sacctmgr show qos mem-normal]
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
# --baseline is stack_input_sarcoma.h5ad -- the SAME tumor-RNA file 10_sarcoma_organoids_2024_context's
# build_sarcoma_organoids_2024_query_pool.py used to seed generation. Fixed 2026-08-20: this script
# previously passed score_viability_adapters.py's own stale default,
# stack_input_sarcoma_organoids_2024.h5ad -- a pre-"use tumor RNA as the Soragni model input" (commit
# 16f42df) organoid-RNA artifact never updated to match. Confirmed materially different
# data (correlation as low as 0.57 on shared patient IDs, not just a naming/subset gap):
# using it silently subtracted a mismatched-substrate baseline from every non-additive
# delta source (pca/nmf/stack all take --baseline; only additive doesn't).
#
# Multi-checkpoint (2026-08-20): to score a second Stack checkpoint (matching 11's own
# CKPT/OUTDIR override pattern), pass GENDIR/AGGDIR/LABEL together so outputs don't
# clobber the cytokine-aligned (default-checkpoint) run's sarcoma_organoids_2024_pathb_results.csv, e.g.
# after running 11 with
# CKPT=/scratch/alpine/$USER/sciplex_finetune/finetuned-epoch=5-val_loss=6.1078.ckpt
# OUTDIR=generated_sarcoma_organoids_2024_drug_aligned:
#   sbatch --export=GENDIR=generated_sarcoma_organoids_2024_drug_aligned,AGGDIR=generated_sarcoma_organoids_2024_drug_aligned_agg,LABEL=drug_aligned \
#       scripts/alpine/12_sarcoma_organoids_2024_score.sbatch
# LABEL unset (the default) reproduces the original unlabeled filenames.
# Stack embedding representations (2026-08-21): base/aligned checkpoints are a SEPARATE axis
# from GENDIR/AGGDIR/LABEL above (embedding-mode, not generation) -- see 13_sarcoma_organoids_2024_embed.sbatch
# and score_viability_adapters.py's --stack-emb. They don't depend on which generation
# checkpoint this run is scoring, so leave STACK_EMB unset on the drug-aligned (LABEL=drug_
# aligned) pass to avoid duplicating base/aligned rows into both output CSVs; set it on the
# default (unlabeled) pass once 13 has produced emb_sarcoma_organoids_2024_base.h5ad / emb_sarcoma_organoids_2024_aligned.h5ad:
#   sbatch --export=STACK_EMB="base=emb_sarcoma_organoids_2024_base.h5ad aligned=emb_sarcoma_organoids_2024_aligned.h5ad" \
#       scripts/alpine/12_sarcoma_organoids_2024_score.sbatch
#
# CONFIRM BEFORE RUNNING:
#   * GCTX matches the file already staged in $REPO (see build_l1000_context.py's docstring
#     for the download recipe if it's ever missing).
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

# Activate the `stack` env by PATH instead of the module system. `module load anaconda`
# fails inside batch jobs (Lmod: module unknown); the cause is NOT established, so do
# not assert one here. Earlier text claiming it was "gone from amem's module tree" was
# never verified -- corrected 2026-08-24. This pattern is what is PROVEN to work.
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1

GCTX="${GCTX:-GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx}"
GENDIR="${GENDIR:-generated_sarcoma_organoids_2024_faithful}"
AGGDIR="${AGGDIR:-generated_sarcoma_organoids_2024_faithful_agg}"
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
STACK_EMB="${STACK_EMB:-}"           # e.g. "base=emb_sarcoma_organoids_2024_base.h5ad aligned=emb_sarcoma_organoids_2024_
                                      # aligned.h5ad" (unset -> no embedding rows scored).

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
summary.to_csv("sarcoma_organoids_2024_aggregation_summary$SUFFIX.csv", index=False)
PY

# ---- fair source (additive/pca/nmf/stack[/base/aligned]) x readout-adapter grid -------------
HALLMARK_SETS_FLAG=()
if [[ -n "$HALLMARK_SETS" ]]; then
  HALLMARK_SETS_FLAG=(--hallmark-sets "$HALLMARK_SETS")
fi
STACK_EMB_FLAG=()
if [[ -n "$STACK_EMB" ]]; then
  STACK_EMB_FLAG=(--stack-emb $STACK_EMB)  # intentionally unquoted: splits "label=path label=path"
fi
python scripts/score_viability_adapters.py \
    --l1000-dir . \
    --gctx "$GCTX" \
    --generated-dir "$AGGDIR" \
    --baseline data/reference/stack_input_sarcoma.h5ad \
    "${HALLMARK_SETS_FLAG[@]}" \
    "${STACK_EMB_FLAG[@]}" \
    --out-csv "sarcoma_organoids_2024_pathb_results$SUFFIX.csv"
