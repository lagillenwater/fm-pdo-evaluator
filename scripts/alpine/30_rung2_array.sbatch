#!/bin/bash
#SBATCH --job-name=rung2-array
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Rung 2 stage 2: one (source, arm) cell per task. Tasks past the grid exit cleanly.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
PLAN_DIR="${PLAN_DIR:?set PLAN_DIR to the directory rung2_plan.py wrote}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
PARTS="${PARTS:-$PLAN_DIR/parts}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) PLAN_DIR=$PLAN_DIR TASK=$TASK"
python scripts/rung2_score_one.py --plan-dir "$PLAN_DIR" --task-id "$TASK" --out-dir "$PARTS"
