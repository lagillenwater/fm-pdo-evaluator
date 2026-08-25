#!/bin/bash
#SBATCH --job-name=check2-array
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Scatter stage of the Check-2 grid. One array task per representation.
#
# Serial, this took ~3h: 540 penalized fits for the noise draws plus ~57 scored rows, and
# LassoCV/ElasticNetCV cost 12.9s and 10.9s against RidgeCV's 0.3s (measured 2026-08-25 on a
# 44x30x2000 grid). Every unit is independent, so the wall clock should be one representation,
# not all of them.
#
# The support restriction is CROSS-representation -- restrict_representation_support drops a
# drug if any representation lacks it -- so tasks must NOT rebuild their own features. Stage 1
# (check2_plan.py) pins the support; each task here restricts to what it decided. Getting this
# wrong scores every row on a different (patient, drug) set and still produces a table that
# looks right.
#
# Usage, in order:
#   sbatch scripts/alpine/18a_check2_plan.sbatch                     # writes PLAN_DIR
#   sbatch --array=0-N%12 --export=PLAN_DIR=<dir> scripts/alpine/18_check2_array.sbatch
#   sbatch --export=PLAN_DIR=<dir> scripts/alpine/18b_check2_gather.sbatch
# N comes from the plan stage's own output ("N array tasks needed").
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1

PLAN_DIR="${PLAN_DIR:?set PLAN_DIR to the directory check2_plan.py wrote}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
PARTS="${PARTS:-$PLAN_DIR/parts}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) job=${SLURM_JOB_NAME:-local} PLAN_DIR=$PLAN_DIR TASK=$TASK PARTS=$PARTS"

python scripts/check2_score_one.py \
    --plan-dir "$PLAN_DIR" \
    --task-id "$TASK" \
    --n-permutations "${NPERM:-1000}" \
    --out-dir "$PARTS"
