#!/bin/bash
#SBATCH --job-name=dataset-overlap
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Gene AND cell-line overlap for every dataset, including the sarcoma cohorts.
#
# Serial deliberately: this reads file metadata (parquet schemas, HDF5 axis arrays) plus two
# small LINCS tables, so the work is seconds and array-task setup would exceed it.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
OUT_DIR="${OUT_DIR:-promote/dataset_overlap_${SLURM_JOB_ID:-local}}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) job=${SLURM_JOB_ID:-local} OUT_DIR=$OUT_DIR"

python scripts/dataset_overlap_matrix.py \
    --scratch "/scratch/alpine/$USER" \
    --out-dir "$OUT_DIR"
