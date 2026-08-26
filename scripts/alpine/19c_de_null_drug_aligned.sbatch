#!/bin/bash
#SBATCH --job-name=de-null-drugaln
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# The drug-aligned checkpoint's cells of the Check-1b null, which 19 missed.
#
# 19 ran with --generated-dir generated_agg, the CYTOKINE-aligned output, so it nulled only one
# of the two Stack checkpoints the published table reports -- and it nulled the weaker one.
# Finding 5's stronger claim is drug-aligned (de_spearman_lfc 0.466 against cytokine's 0.357).
#
# Two tasks: --array=0-1 maps to the two null kinds for this one source.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1

SOURCES="${SOURCES:-tahoe_sources}"
DE_CALLS="${DE_CALLS:-tahoe_de_calls/de_calls.parquet}"
SHUFFLES="${SHUFFLES:-200}"
PARTS="${PARTS:-promote/de_null_parts}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
NULLS=(shuffle_all within_drug)
NULL="${NULLS[$TASK]}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) job=${SLURM_JOB_NAME:-local} SOURCES=$SOURCES NULL=$NULL SHUFFLES=$SHUFFLES"

# Dump the drug-aligned generated delta alongside the cytokine one, under its own label.
if [ ! -f "$SOURCES/stack_drug_aligned_delta.parquet" ]; then
  if mkdir "$SOURCES/.drugaln.lock" 2>/dev/null; then
    echo "=== dumping stack_drug_aligned ==="
    python scripts/score_generation_eval.py \
        --deltas-bundle tahoe_deltas \
        --generated-dir stack_drug_aligned=generated_drug_aligned_agg \
        --query-baseline tahoe_query_baseline.h5ad \
        --pert-map context_by_drug/pert_to_cid.tsv \
        --dump-sources "$SOURCES" --dump-only
    rmdir "$SOURCES/.drugaln.lock"
  else
    while [ ! -f "$SOURCES/stack_drug_aligned_delta.parquet" ]; do sleep 20; done
  fi
fi

python scripts/de_permutation_null.py \
    --sources-dir "$SOURCES" \
    --de-calls "$DE_CALLS" \
    --n-shuffles "$SHUFFLES" \
    --only-source stack_drug_aligned \
    --only-null "$NULL" \
    --out-csv "$PARTS/part_drugaligned_${NULL}.csv"
