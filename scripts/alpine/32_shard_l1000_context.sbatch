#!/bin/bash
#SBATCH --job-name=shard-l1000-context
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
CONTEXT="${CONTEXT:-l1000_context_tahoe.h5ad}"
OUTDIR="${OUTDIR:-l1000_context_by_drug}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) CONTEXT=$CONTEXT OUTDIR=$OUTDIR"
python scripts/shard_context_by_drug.py --context "$CONTEXT" --out-dir "$OUTDIR"
