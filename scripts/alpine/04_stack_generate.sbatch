#!/bin/bash
#SBATCH --job-name=stack-gen
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-119%8              # one drug per task; %8 caps concurrent A100s. Over-
#SBATCH --output=logs/%x-%A_%a.out   #   sized on purpose -- tasks past the drug count exit 0.
#SBATCH --error=logs/%x-%A_%a.out
#
# Stack-Large-Aligned generation, one drug per GPU task, reading the per-drug context
# shards 03 produced. Each task is the documented single-context invocation on a
# 1-drug context, so this needs no special stack-generation flag -- it just parallelizes
# the independent per-drug generations across A100s.
#
# QUERY is tahoe_query.h5ad (the per-line control baselines 03 writes) -- the same file
# 05 passes as --query-baseline, so the scored delta is generated - this baseline.
# CONFIRM BEFORE RUNNING (your Stack setup, not derivable here):
#   * --checkpoint / --genelist   your Stack-Aligned checkpoint + genelist paths.
#   * output naming: build_generated_deltas matches generated/<file> to a pert via the
#                    filename stem or uns; verify Stack names outputs by pert_id.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

# ---- environment: the arc-stack GPU env (torch cu121 + stack-generation), NOT .venv --
module purge
module load anaconda
conda activate stack        # <-- CHANGE to your Stack env
export HF_HOME="/scratch/alpine/$USER/hf"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
SHARD=$(printf "context_by_drug/%04d.h5ad" "$TASK")
if [[ ! -f "$SHARD" ]]; then
  echo "no shard for task $TASK ($SHARD) -- past the drug count, nothing to do"; exit 0
fi
PERT=$(awk -F'\t' -v i="$TASK" '$1==i{print $2}' context_by_drug/manifest.tsv)
echo "task $TASK -> pert '$PERT' ($SHARD)"

QUERY="tahoe_query.h5ad"              # cell-line control baselines (written by 03)
CKPT="CHANGE_ME/bc_large_aligned.ckpt"
GENELIST="CHANGE_ME/basecount_1000per_15000max.pkl"
mkdir -p generated

stack-generation \
    --checkpoint "$CKPT" \
    --base-adata "$SHARD" \
    --test-adata "$QUERY" \
    --genelist "$GENELIST" \
    --gene-name-col feature_name \
    --output-dir generated/
