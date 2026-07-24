#!/bin/bash
#SBATCH --job-name=tahoe-pull
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G                    # per-drug streaming keeps peak low; no big materialize
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# The pseudobulk shortcut, as a SINGLE streaming process -- deliberately NOT an array.
# Tahoe's pseudobulk table is a flat 1026-file shard set with no drug partition, so every
# query scans all footers; fanning it across array tasks trips HF's 429 rate limit and each
# task still buffers all its drugs at once -> OOM. The script instead pulls one drug at a
# time and reshapes immediately, so peak memory is a single drug and the requests serialize.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

module load anaconda
conda activate stack                 # data env: datasets + pyarrow + huggingface_hub (no torch)
export PYTHONPATH="$REPO/src"
export HF_HOME="/scratch/alpine/$USER/hf"
export HF_TOKEN="CHANGE_ME"          # an HF token lifts the anonymous 429 ceiling (huggingface-cli login)

python scripts/build_tahoe_pseudobulk_deltas.py \
    --drugs-cid-file data/static/tahoe_target_cids.txt \
    --out-dir tahoe_deltas
