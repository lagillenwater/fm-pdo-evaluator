#!/bin/bash
#SBATCH --job-name=check2-gather
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Stage 3: assemble the parts. Refuses to write a final table if any array task is missing --
# a short table that looks whole is the failure mode worth guarding against.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1

PLAN_DIR="${PLAN_DIR:?set PLAN_DIR}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) job=${SLURM_JOB_NAME:-local} PLAN_DIR=$PLAN_DIR"

python scripts/check2_gather.py \
    --plan-dir "$PLAN_DIR" \
    --parts-dir "${PARTS:-$PLAN_DIR/parts}" \
    --out-dir "${OUT:-$PLAN_DIR}"
