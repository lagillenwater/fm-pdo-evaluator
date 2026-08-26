#!/bin/bash
#SBATCH --job-name=rung2-plan
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Rung 2 stage 1. The .gctx read is the expensive shared part and both arms need the same
# pinned panel, so it happens once here rather than per array task.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
SCRATCH="/scratch/alpine/$USER"
PLAN_DIR="${PLAN_DIR:-results/rung2_${SLURM_JOB_ID:-local}}"
GCTX="${GCTX:-GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) job=${SLURM_JOB_ID:-local} PLAN_DIR=$PLAN_DIR"
python scripts/rung2_plan.py \
    --deltas-bundle tahoe_deltas \
    --l1000-dir . --gctx "$GCTX" \
    --pert-map context_by_drug/pert_to_cid.tsv \
    --panel-source "stack_cytokine=generated_agg/5-Azacytidine.h5ad" \
    --panel-source "stack_drug_aligned=generated_drug_aligned_agg/5-Azacytidine.h5ad" \
    --panel-source "sciplex=$SCRATCH/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad" \
    --out-dir "$PLAN_DIR"
