#!/bin/bash
#SBATCH --job-name=tahoe-pull
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                    # DuckDB memory_limit is 28G + the aggregated result frame
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# The pseudobulk shortcut, as a SINGLE process -- deliberately NOT an array. Tahoe's pseudobulk
# table is a flat 1026-file shard set with no drug partition. --local-dir downloads it to scratch
# once (authenticated), then DuckDB aggregates the treated-vs-DMSO means IN-ENGINE (bounded memory,
# spills to disk), so no drug's raw rows -- some are tens of millions -- are ever materialized.
#
# Authenticate first so the download isn't rate-limited:  hf auth login
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

module load anaconda
conda activate stack                 # data env: datasets + pyarrow + huggingface_hub + duckdb
export PYTHONPATH="$REPO/src"
export HF_HUB_CACHE="/scratch/alpine/$USER/hf"   # cache large downloads on scratch, not /projects
# do NOT set HF_HOME to scratch -- that hides the token `hf auth login` saved under the HF cache:
export HF_TOKEN="${HF_TOKEN:-$(cat "${XDG_CACHE_HOME:-$HOME/.cache}/huggingface/token" 2>/dev/null || true)}"

python -c "import duckdb" 2>/dev/null || pip install -q duckdb   # aggregate engine (compute node has internet)

python scripts/build_tahoe_pseudobulk_deltas.py \
    --drugs-cid-file data/static/tahoe_target_cids.txt \
    --local-dir "/scratch/alpine/$USER/tahoe_pseudobulk_de" \
    --out-dir tahoe_deltas
