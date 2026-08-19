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
#SBATCH --array=0-19%16              # 20 L1000-matched Soragni drugs (generated_rich/'s
                                      # known count) = tasks 0-19; CONFIRM against
                                      # context_by_drug_soragni/manifest.tsv's line count
                                      # after 10 runs and adjust --array if it differs.
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
# CONFIRM BEFORE RUNNING:
#   * --checkpoint / --genelist   your Stack-Aligned checkpoint + genelist paths.
#   * --array bound against context_by_drug_soragni/manifest.tsv (see 10).
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

module purge
module load anaconda
conda activate stack
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
