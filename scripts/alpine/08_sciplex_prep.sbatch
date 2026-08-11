#!/bin/bash
#SBATCH --job-name=sciplex-prep
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Prepare sci-Plex 3 (Srivatsan 2020, GEO GSE139944) as the single-cell DRUG-perturbation
# training set for stack-finetune -- the drug-domain analogue of the cytokine alignment, and
# disjoint from Tahoe so testing on Tahoe stays non-circular. ~650k cells, A549/K562/MCF7 x
# ~188 compounds x doses.
#
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

module load anaconda
conda activate stack
export PYTHONPATH="$REPO/src"

SCIPLEX_DIR="/scratch/alpine/$USER/sciplex"
mkdir -p "$SCIPLEX_DIR"
RAW="$SCIPLEX_DIR/sciplex3_raw.h5ad"

# 1. Download the sci-Plex 3 AnnData (scPerturb's full-gene h5ad, Zenodo record 13350497 --
#    see the chemCPA "middle_subset" alternative below for the pre-subset-to-2000-genes
#    version this replaces; the full-gene version is what Stack's generation panel needs).
#    Override SCIPLEX_URL to swap it.
SCIPLEX_URL="${SCIPLEX_URL:-https://zenodo.org/records/13350497/files/SrivatsanTrapnell2020_sciplex3.h5ad?download=1}"
[[ -f "$RAW" ]] || wget -O "$RAW" "$SCIPLEX_URL"

# 2. Reformat to stack-finetune's layout: raw counts, gene symbols, pert_id/is_control/cell_line/dose.
#    build_sciplex_finetune.py auto-detects the obs/layer names and prints what it picked -- check
#    that line in the log; override --pert-col/--cell-line-col/--dose-col/--counts-layer if wrong.
python scripts/build_sciplex_finetune.py \
    --input "$RAW" \
    --out "$SCIPLEX_DIR/sciplex_finetune.h5ad" \
    --min-cells-per-cond 50
