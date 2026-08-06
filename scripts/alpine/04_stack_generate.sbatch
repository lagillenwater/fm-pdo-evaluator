#!/bin/bash
#SBATCH --job-name=stack-gen
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-32%16              # 33 drugs = tasks 0-32; %16 caps concurrent A100s.
#SBATCH --output=logs/%x-%A_%a.out   #   override with --array=<N> to run a single drug.
#SBATCH --error=logs/%x-%A_%a.out
#
# Stack-Large-Aligned in-context generation, one drug per GPU task, reading the per-drug
# context shards 03 produced. --split-column pert_id groups the base cells by drug and
# --split-values "$PERT" restricts each task to its own drug (the shard also holds DMSO),
# generating the query (test-adata) treated state under that drug. Parallelized across A100s.
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

QUERY="tahoe_query.h5ad"              # cell-line control baselines (written by 03); 50 cells
# Env-overridable so the base (unaligned) checkpoint can be A/B'd against the aligned one, e.g.:
#   CKPT=stack-large/bc_large.ckpt GENELIST=stack-large/basecount_1000per_15000max.pkl \
#   OUTDIR=generated_base sbatch --array=3 scripts/alpine/04_stack_generate.sbatch
CKPT="${CKPT:-stack-aligned/bc_large_aligned.ckpt}"
GENELIST="${GENELIST:-stack-aligned/basecount_1000per_15000max.pkl}"
OUTDIR="${OUTDIR:-generated}"
mkdir -p "$OUTDIR"

# Stack splits its 512-cell set as n_test = int(512*(1 - prompt_ratio - context_ratio)), rest base.
# --mode vanilla is REQUIRED: the default (generate) mode ignores context_ratio and instead SCHEDULES
# it (linspace(context_ratio_min, context_ratio)) per step, so n_test varies and most steps pad the
# 50-cell query -> IndexError. vanilla passes the fixed context_ratio, so ratio = prompt+context is
# constant. We want n_test = 50 = the query size (no padding): prompt+context ~= 1 - 50/512 = 0.90;
# 0.501 + 0.4 = 0.901 -> n_test = 50, n_base = 462 (unique for drugs with >=462 cells; thinner drugs
# loop the base -> duplicates -> skipped). COUPLED to the 50-line query -- recompute if 03 changes.

stack-generation \
    --checkpoint "$CKPT" \
    --base-adata "$SHARD" \
    --test-adata "$QUERY" \
    --genelist "$GENELIST" \
    --gene-name-col feature_name \
    --split-column pert_id \
    --split-values "$PERT" \
    --device cuda \
    --mode vanilla \
    --prompt-ratio 0.501 \
    --context-ratio 0.4 \
    --output-dir "$OUTDIR"/
