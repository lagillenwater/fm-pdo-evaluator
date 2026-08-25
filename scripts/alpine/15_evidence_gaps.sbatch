#!/bin/bash
#SBATCH --job-name=evidence-gaps
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Closes the two load-bearing evidence gaps the 2026-08-24 audit found: the Check-1b
# permutation null (published, script never committed) and the gen_logit threshold
# (described as calibrated, no sweep output anywhere). Both write a CSV to promote/.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

# Activate the `stack` env by PATH instead of the module system -- the pattern proven to work.
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1

mkdir -p promote

echo "=== gen_logit threshold sweep ==="
python scripts/sweep_gen_logit.py \
    --generated-dir generated \
    --query-baseline tahoe_query_baseline.h5ad \
    --pert-map context_by_drug/pert_to_cid.tsv \
    --deltas-bundle tahoe_deltas \
    --out-csv promote/gen_logit_sweep.csv
