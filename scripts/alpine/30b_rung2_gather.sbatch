#!/bin/bash
#SBATCH --job-name=rung2-gather
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
PLAN_DIR="${PLAN_DIR:?set PLAN_DIR}"
PARTS="${PARTS:-$PLAN_DIR/parts}"
OUT="${OUT:-promote/rung2_$(basename "$PLAN_DIR")}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) PLAN_DIR=$PLAN_DIR OUT=$OUT"
python scripts/rung2_gather.py --plan-dir "$PLAN_DIR" --parts-dir "$PARTS" --out-dir "$OUT"
