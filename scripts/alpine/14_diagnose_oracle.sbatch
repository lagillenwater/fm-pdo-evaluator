#!/bin/bash
#SBATCH --job-name=diagnose-oracle
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
# Diagnostic for the oracle/additive identity seen in job 31631728 (2026-08-24): both rows
# returned byte-identical Check-2 numbers despite being built from different data. Also tests
# whether make_penalty's alpha ceiling (1e3) is binding at p ~ 2000 -- see
# scripts/diagnose_oracle_additive.py's module docstring for the full hypothesis.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

# Activate the `stack` env by PATH instead of the module system. `module load anaconda`
# fails inside batch jobs (Lmod: module unknown); the cause is NOT established, so do not
# assert one here. This pattern is what is PROVEN to work.
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1

python scripts/diagnose_oracle_additive.py \
    --deltas-bundle tahoe_deltas \
    --folds 5
