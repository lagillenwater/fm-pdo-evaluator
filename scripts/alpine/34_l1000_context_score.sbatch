#!/bin/bash
#SBATCH --job-name=l1000ctx-score
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Stack's actual rung-2 arm (decision D3): score generation from an L1000-built context against
# the Tahoe query baseline, for both checkpoints. Generation already ran (2026-08-25) into
# generated_l1000ctx_cytokine/ and generated_l1000ctx_drugaligned/; this maps their BRD-K
# filenames to Tahoe's own PubChem-CID convention and scores them the same way rung 1 scores
# stack_cytokine/stack_drug_aligned.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
J="${SLURM_JOB_ID:-local}"
OUT_DIR="${OUT_DIR:-promote/l1000ctx_score_$J}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) job=$J OUT_DIR=$OUT_DIR"

python scripts/score_l1000_context_generation.py \
    --deltas-bundle tahoe_deltas \
    --l1000-dir . \
    --query-baseline tahoe_query_baseline.h5ad \
    --generated-dir "stack_cytokine=generated_l1000ctx_cytokine" \
    --generated-dir "stack_drug_aligned=generated_l1000ctx_drugaligned" \
    --agg-out-dir "$OUT_DIR/agg" \
    --out-dir "$OUT_DIR"
