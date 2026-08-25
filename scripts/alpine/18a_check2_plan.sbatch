#!/bin/bash
#SBATCH --job-name=check2-plan
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Stage 1: build every representation once, fix the shared (patient, drug) support across all
# of them, and pin it to disk. See check2_plan.py for why this cannot be done per array task.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1

PLAN_DIR="${PLAN_DIR:-results/plan_${SLURM_JOB_ID:-local}}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) script=$(basename "${BASH_SOURCE[0]}") PLAN_DIR=$PLAN_DIR"

python scripts/check2_plan.py \
    --deltas-bundle tahoe_deltas \
    --generated-dir generated_agg \
    --query-baseline tahoe_query_baseline.h5ad \
    --pert-map context_by_drug/pert_to_cid.tsv \
    --stack-emb base=emb_base.h5ad aligned=emb_aligned.h5ad \
    --out-dir "$PLAN_DIR"
echo
echo "NEXT: sbatch --array=0-<N-1>%12 --export=PLAN_DIR=$PLAN_DIR scripts/alpine/18_check2_array.sbatch"
