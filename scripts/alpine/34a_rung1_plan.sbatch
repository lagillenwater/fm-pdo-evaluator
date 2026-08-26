#!/bin/bash
#SBATCH --job-name=rung1-plan
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
SCRATCH="/scratch/alpine/$USER"
PLAN_DIR="${PLAN_DIR:-results/rung1_${SLURM_JOB_ID:-local}}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) PLAN_DIR=$PLAN_DIR"
python scripts/rung1_plan.py \
    --deltas-bundle tahoe_deltas \
    --generated-dir stack_cytokine=generated_agg \
    --generated-dir stack_drug_aligned=generated_drug_aligned_agg \
    --query-baseline tahoe_query_baseline.h5ad \
    --pert-map context_by_drug/pert_to_cid.tsv \
    --panel-source "sciplex=$SCRATCH/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad" \
    --folds 5 --k 10 \
    --out-dir "$PLAN_DIR"
