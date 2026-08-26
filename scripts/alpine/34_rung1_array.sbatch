#!/bin/bash
#SBATCH --job-name=rung1-array
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# One baseline per task. pca and nmf each fit a ridge per fold across every panel gene, which is
# what pushed the serial version past its walltime; they share nothing but the pinned inputs.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
PLAN_DIR="${PLAN_DIR:?set PLAN_DIR to the directory rung1_plan.py wrote}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) PLAN_DIR=$PLAN_DIR TASK=$TASK"
python scripts/rung1_build_one.py --plan-dir "$PLAN_DIR" --task-id "$TASK"
