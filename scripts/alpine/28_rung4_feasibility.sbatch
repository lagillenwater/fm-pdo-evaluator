#!/bin/bash
#SBATCH --job-name=rung4-feasibility
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Rung-4 feasibility: usable cohort, ceiling availability, drug axis.
# Serial: metadata-only reads of small tables, seconds of work. It runs CONCURRENTLY with the
# two ceiling jobs rather than after them -- they share no inputs.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
J="${SLURM_JOB_ID:-local}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) job=$J"
python scripts/organoid_rung4_feasibility.py \
    --screen data/raw/soragni/tables/drug_screen.parquet \
    --expression data/reference/stack_input_sarcoma.h5ad \
    --tables-dir data/raw/soragni/tables \
    --l1000-pert-info GSE92742_Broad_LINCS_pert_info.txt.gz \
    --gdsc2-compounds data/raw/gdsc2_sarcoma/gdsc2/screened_compounds_rel_8.5.csv \
    --out-dir "promote/rung4_feasibility_$J"
