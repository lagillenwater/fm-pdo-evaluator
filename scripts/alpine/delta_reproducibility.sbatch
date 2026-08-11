#!/bin/bash
#SBATCH --job-name=delta-repro
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                    # DuckDB memory_limit 36G + the aggregated result frame
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Delta-reproducibility ceiling for check 1. Splits each (line, drug)'s replicate plates into two
# halves, aggregates each half's delta in DuckDB, and correlates the halves over the HVG genes --
# the delta's own test-retest reliability, i.e. the most any predictor (Stack included) could score
# on check 1. CPU only; reads the Tahoe DE parquet already on scratch, so it runs alongside the
# context build (03) on a separate node -- NOT part of the 00-05 pipeline.
#
# Authenticate first (drug_metadata is fetched from HF):  hf auth login
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

module load anaconda
conda activate stack                 # data env: datasets + pyarrow + huggingface_hub + duckdb
export PYTHONPATH="$REPO/src"
export HF_HUB_CACHE="/scratch/alpine/$USER/hf"   # cache off /projects
# do NOT set HF_HOME to scratch -- that hides the token `hf auth login` saved under the HF cache:
export HF_TOKEN="${HF_TOKEN:-$(cat "${XDG_CACHE_HOME:-$HOME/.cache}/huggingface/token" 2>/dev/null || true)}"

python -c "import duckdb" 2>/dev/null || pip install -q duckdb

python scripts/delta_reproducibility.py \
    --local-dir "/scratch/alpine/$USER/tahoe_pseudobulk_de" \
    --drugs-cid-file data/static/tahoe_target_cids.txt
