#!/bin/bash
#SBATCH --job-name=soragni-gen
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --array=0-18%4               # 19 L1000-matched Soragni drugs (confirmed via
                                      # context_by_drug_soragni/manifest.tsv's line count,
                                      # 2026-08-20) = tasks 0-18. Reconfirm if 10's context
                                      # build ever changes the L1000<->Soragni drug match.
                                      # Concurrency capped at 4 (2026-08-21, down from 16):
                                      # ah200 nodes (c3gpu-g5-*) carry 4 H200s each; the
                                      # first 19-task/%16 run put 6-7 tasks concurrently on
                                      # one node and 17/19 failed with "CUDA unknown error"
                                      # (torch _lazy_init) [checked: sacct + per-task logs on the compute node]
                                      # on the actual compute node) -- GPU device isolation
                                      # under rapid concurrent launch on one shared node, not
                                      # a code bug. %4 keeps launches at or under one node's
                                      # physical GPU count.
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Path B faithful generation: Stack-Large-Aligned in-context generation against the
# synthetic query pool (10_soragni_context.sbatch), one drug per GPU task -- mirrors
# 04_stack_generate.sbatch's Tahoe procedure (same --mode mdm schedule) with a Soragni-
# specific context/query.
#
# QUERY is soragni_query_pool.h5ad (24 synthetic replicates/patient, NOT real cells --
# see build_soragni_query_pool.py). Output carries multiple rows per patient
# (obs["gen_logit"]); run fmharness.stack_aggregate.aggregate_generated_replicates on
# OUTDIR before scoring, same as the Tahoe pipeline.
#
# Multi-checkpoint (2026-08-20): to match Tahoe's Check 1/2 checkpoint comparison, run
# this a second time against the drug-aligned (sci-Plex fine-tuned) checkpoint -- the
# exact file that produced this week's published drug-aligned Tahoe numbers:
#   sbatch --export=CKPT=/scratch/alpine/$USER/sciplex_finetune/finetuned-epoch=5-val_loss=6.1078.ckpt,GENELIST=stack-aligned/basecount_1000per_15000max.pkl,OUTDIR=generated_soragni_drug_aligned \
#       scripts/alpine/11_soragni_generate.sbatch
# Then score it with 12_soragni_score.sbatch's matching GENDIR/AGGDIR/LABEL=drug_aligned
# override (see that script's own comment).
# CONFIRM BEFORE RUNNING:
#   * --checkpoint / --genelist   your Stack-Aligned checkpoint + genelist paths.
#   * --array bound against context_by_drug_soragni/manifest.tsv (see 10).
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

module purge
# Activate the `stack` env by PATH instead of the module system. `module load anaconda`
# fails inside batch jobs (Lmod: module unknown) for reasons NOT established -- see
# ralpine's submit comment; do not assert a cause here. This pattern is the one that is
# PROVEN to work: 10_soragni_context and 12_soragni_score ran with it on 2026-08-22.
# `conda activate` is deliberately not used: this install ships no etc/profile.d/conda.sh.
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="/scratch/alpine/$USER/hf"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
SHARD=$(printf "context_by_drug_soragni/%04d.h5ad" "$TASK")
if [[ ! -f "$SHARD" ]]; then
  echo "no shard for task $TASK ($SHARD) -- past the drug count, nothing to do"; exit 0
fi
PERT=$(awk -F'\t' -v i="$TASK" '$1==i{print $2}' context_by_drug_soragni/manifest.tsv)
echo "task $TASK -> pert '$PERT' ($SHARD)"

QUERY="soragni_query_pool.h5ad"          # 17 patients x 24 synthetic replicates (written by 10)
# Same --export=VAR=val,... requirement as 04_stack_generate.sbatch (no --export=ALL --
# it breaks `module load anaconda`'s Lmod state; see that script's comment for the full story).
CKPT="${CKPT:-stack-aligned/bc_large_aligned.ckpt}"
GENELIST="${GENELIST:-stack-aligned/basecount_1000per_15000max.pkl}"
OUTDIR="${OUTDIR:-generated_soragni_faithful}"
echo "Resolved: CKPT=$CKPT GENELIST=$GENELIST OUTDIR=$OUTDIR"
mkdir -p "$OUTDIR"

stack-generation \
    --checkpoint "$CKPT" \
    --base-adata "$SHARD" \
    --test-adata "$QUERY" \
    --genelist "$GENELIST" \
    --gene-name-col feature_name \
    --split-column pert_id \
    --split-values "$PERT" \
    --device cuda \
    --mode mdm \
    --prompt-ratio 0.25 \
    --context-ratio 0.4 \
    --context-ratio-min 0.2 \
    --output-dir "$OUTDIR"/
