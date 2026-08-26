#!/bin/bash
#SBATCH --job-name=rung1-gather
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
PLAN_DIR="${PLAN_DIR:?set PLAN_DIR}"
OUT="${OUT:-promote/rung1_$(basename "$PLAN_DIR")}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) PLAN_DIR=$PLAN_DIR OUT=$OUT"
python scripts/rung1_gather.py --plan-dir "$PLAN_DIR" --out-dir "$OUT"
# the DE-null array consumes the same pinned sources, so point it here
echo "NEXT: --export=SOURCES=$PLAN_DIR/sources for the Check-1b null array"
