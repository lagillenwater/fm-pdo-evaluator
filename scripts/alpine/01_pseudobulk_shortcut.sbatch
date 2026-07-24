#!/bin/bash
#SBATCH --job-name=tahoe-pull
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                    # bounded scanner + per-drug reshape; headroom over ~3G/drug
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# The pseudobulk shortcut, as a SINGLE streaming process -- deliberately NOT an array.
# Tahoe's pseudobulk table is a flat 1026-file shard set with no drug partition, so every
# query touches all files; fanning it across array tasks trips HF's 429 rate limit. The
# script reads one drug at a time through a bounded scanner (low readahead, no pre_buffer),
# so peak memory is a single drug's result, not the scan. --local-dir downloads the config
# to scratch once (authenticated) and reads locally -- no per-query network at all.
#
# Authenticate first so downloads aren't rate-limited:  hf auth login
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

module load anaconda
conda activate stack                 # data env: datasets + pyarrow + huggingface_hub (no torch)
export PYTHONPATH="$REPO/src"
export HF_HUB_CACHE="/scratch/alpine/$USER/hf"   # cache large downloads on scratch, not /projects
# do NOT set HF_HOME to scratch -- that hides the token `hf auth login` saved under the HF cache:
export HF_TOKEN="${HF_TOKEN:-$(cat "${XDG_CACHE_HOME:-$HOME/.cache}/huggingface/token" 2>/dev/null || true)}"

python scripts/build_tahoe_pseudobulk_deltas.py \
    --drugs-cid-file data/static/tahoe_target_cids.txt \
    --local-dir "/scratch/alpine/$USER/tahoe_pseudobulk_de" \
    --out-dir tahoe_deltas
