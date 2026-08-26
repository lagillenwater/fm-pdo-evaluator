#!/bin/bash
#SBATCH --job-name=path-panels
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
# The exact panel for each path and check. Pairwise overlaps bound these but do not give them:
# a four-way intersection can be far smaller than its smallest pair, so it has to be computed.
#
# Serial deliberately: metadata-only reads (parquet schemas, HDF5 axis arrays), seconds of work.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
S="/scratch/alpine/$USER"
J="${SLURM_JOB_ID:-local}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) job=$J"

SRC=(
  --source "tahoe=parquet=tahoe_deltas/real_delta.parquet"
  --source "stack_cytokine=h5ad=generated_agg/5-Azacytidine.h5ad"
  --source "stack_drug_aligned=h5ad=generated_drug_aligned_agg/5-Azacytidine.h5ad"
  --source "sciplex=h5ad=$S/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad"
  --source "gdsc2=h5ad=data/reference/stack_input_gdscv2.h5ad"
  --source "organoids=h5ad=data/reference/stack_input_sarcoma.h5ad"
)

echo "=============== Path A / Check 1: tahoe n stack n sciplex ==============="
python scripts/gene_panel_audit.py "${SRC[@]}" \
    --require "tahoe,stack_cytokine,stack_drug_aligned,sciplex" \
    --out-dir "promote/panel_pathA_check1_$J"

echo
echo "=========== Path A / Check 2: + gdsc2 (the AUC readout cohort) ==========="
python scripts/gene_panel_audit.py "${SRC[@]}" \
    --require "tahoe,stack_cytokine,stack_drug_aligned,sciplex,gdsc2" \
    --out-dir "promote/panel_pathA_check2_$J"

echo
echo "=========== Path B: tahoe as the delta-transfer corpus ==========="
python scripts/gene_panel_audit.py "${SRC[@]}" \
    --require "tahoe,stack_cytokine,stack_drug_aligned,organoids,gdsc2" \
    --out-dir "promote/panel_pathB_$J"
