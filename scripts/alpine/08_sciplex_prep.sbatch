#!/bin/bash
#SBATCH --job-name=sciplex-prep
# amem, not acpu (Alpine's amilan partition was retired) -- acpu's MaxMemPerCPU=3840MB caps
# an 8-cpu job at ~30GB, below what the full-gene sci-Plex source needs. mem-normal's own
# QOS floor (MinTRES mem=240G) [checked: sacctmgr show qos mem-normal] sets --mem here, not a usage estimate;
# 12 cpus keeps the per-cpu average (20000MB) under amem's MaxMemPerCPU=21333MB ceiling at
# that --mem.
#SBATCH --partition=amem
#SBATCH --qos=mem-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
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

# Activate the `stack` env by PATH instead of the module system. `module load anaconda`
# fails inside batch jobs (Lmod: module unknown) for reasons NOT established -- see
# ralpine's submit comment; do not assert a cause here. This pattern is the one that is
# PROVEN to work: 10_sarcoma_organoids_2024_context and 12_sarcoma_organoids_2024_score ran with it on 2026-08-22.
# `conda activate` is deliberately not used: this install ships no etc/profile.d/conda.sh.
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"

SCIPLEX_DIR="/scratch/alpine/$USER/sciplex"
mkdir -p "$SCIPLEX_DIR"

# 1. Download the sci-Plex 3 AnnData (scPerturb's full-gene h5ad, Zenodo record 13350497 --
#    the full-gene version is what Stack's generation panel needs). Override SCIPLEX_URL to
#    swap it. RAW's filename is derived from the URL (not hardcoded) so a source swap can't
#    silently reuse a stale cache from a different source -- caught in production 2026-08-11,
#    when a same-named cache left over from the retired 2000-gene chemCPA URL was silently
#    reused after this URL was switched to the full-gene release.
SCIPLEX_URL="${SCIPLEX_URL:-https://zenodo.org/records/13350497/files/SrivatsanTrapnell2020_sciplex3.h5ad?download=1}"
RAW="$SCIPLEX_DIR/$(basename "${SCIPLEX_URL%%\?*}")"
[[ -f "$RAW" ]] || wget -O "$RAW" "$SCIPLEX_URL"

# 2. Reformat to stack-finetune's layout: raw counts, gene symbols, pert_id/is_control/cell_line/dose.
#    build_sciplex_finetune.py auto-detects the obs/layer names and prints what it picked -- check
#    that line in the log; override --pert-col/--cell-line-col/--dose-col/--counts-layer if wrong.
#    Written to its own subdirectory, not loose in $SCIPLEX_DIR: stack-finetune's --dataset_configs
#    path arg is glob'd (Path(path).glob("*.h5ad")), not read as a direct file path -- a bare
#    directory here would also glob-match RAW and any other .h5ad left in $SCIPLEX_DIR (both the
#    raw download and a stale prior-URL cache sit there today), silently feeding the wrong obs
#    schema into fine-tuning. Caught in production 2026-08-12 (job 31159439 loaded 0 cells because
#    09 originally pointed --dataset_configs at the *file*, which glob() also treats as empty).
FINETUNE_DIR="$SCIPLEX_DIR/finetune_input"
mkdir -p "$FINETUNE_DIR"
python scripts/build_sciplex_finetune.py \
    --input "$RAW" \
    --out "$FINETUNE_DIR/sciplex_finetune.h5ad" \
    --min-cells-per-cond 50
