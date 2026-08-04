#!/bin/bash
#SBATCH --job-name=sciplex-prep
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Prepare sci-Plex 3 (Srivatsan 2020, GEO GSE139944) as the single-cell DRUG-perturbation
# training set for stack-finetune -- the drug-domain analogue of the cytokine alignment, and
# disjoint from Tahoe so testing on Tahoe stays non-circular. ~650k cells, A549/K562/MCF7 x
# ~188 compounds x doses.
#
# TODO: scripts/build_sciplex_finetune.py is NOT written yet. It must download GSE139944 and
# emit an h5ad in stack-finetune's expected layout: raw counts on the 15,012-gene panel, an obs
# split column (compound name incl. a vehicle control), cell_line, dose. The exact --dataset_configs
# string format 09 consumes is still unknown -- inspect how the cytokine alignment was configured
# (a --config YAML example in the arc-stack repo/cli) before finalizing both this and 09.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

module load anaconda
conda activate stack
export PYTHONPATH="$REPO/src"
export HF_HUB_CACHE="/scratch/alpine/$USER/hf"

python scripts/build_sciplex_finetune.py \
    --geo GSE139944 \
    --out /scratch/alpine/$USER/sciplex/sciplex_finetune.h5ad \
    --genelist stack-large/basecount_1000per_15000max.pkl
