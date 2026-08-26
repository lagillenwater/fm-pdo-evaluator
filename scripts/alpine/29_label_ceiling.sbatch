#!/bin/bash
#SBATCH --job-name=label-ceiling
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Rung-3 ceiling: agreement between independent viability screens (GDSC2 vs CTRPv2/PRISM).
# The 0.31-0.47 interaction figure is quoted in docs/tahoe_generation_results.md with no
# promoted artifact behind it, so it cannot be verified without this rerun.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
J="${SLURM_JOB_ID:-local}"
OUT="promote/label_ceiling_$J"
mkdir -p "$OUT"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) job=$J OUT=$OUT"
python scripts/label_ceiling.py \
    --ref-dir data/raw/coderdata --ref-prefix gdscv2 \
    --comp-dir data/raw/coderdata --comparators ctrpv2,prism \
    --out "$OUT/label_ceiling.csv"
