#!/bin/bash
#SBATCH --job-name=tahoe-baselines
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Score the additive + k-NN baselines (both checks) off the tahoe_deltas/ bundle 01 wrote.
# Kept separate from 01 so you can re-score with different --k / --n-hvg / --n-permutations
# without re-pulling from HF.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

module load anaconda
conda activate stack
export PYTHONPATH="$REPO/src"

python scripts/score_generation_eval.py \
    --deltas-bundle tahoe_deltas \
    --k 10 \
    --n-permutations 1000
