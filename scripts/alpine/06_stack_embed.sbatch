#!/bin/bash
#SBATCH --job-name=stack-embed
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-1                   # 0 = base (bc_large), 1 = cytokine-aligned -- run both in parallel
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Embed the 50-line query baseline with the base AND cytokine-aligned checkpoints so 07 can score
# both as check-2 representations, head-to-head with expr/PCA. The base has no generation head but
# DOES have the encoder, so its embedding is its only way into the eval. Independent per checkpoint
# -> a 2-task array.
#
# stack-embedding: --checkpoint/--adata/--genelist/--output(.h5ad|.npy)/--gene-name-col/--device,
# plus --obs-source to attach the query's obs (the line ids) to the embeddings so 07 keys on them.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

module purge
module load anaconda
conda activate stack

TASK="${SLURM_ARRAY_TASK_ID:-0}"
case "$TASK" in
  0) CKPT="stack-large/bc_large.ckpt"           ; GENELIST="stack-large/basecount_1000per_15000max.pkl"   ; LABEL="base" ;;
  1) CKPT="stack-aligned/bc_large_aligned.ckpt" ; GENELIST="stack-aligned/basecount_1000per_15000max.pkl" ; LABEL="aligned" ;;
  *) echo "no checkpoint for task $TASK -- nothing to do"; exit 0 ;;
esac
echo "task $TASK -> embed query with '$LABEL' ($CKPT)"

# tahoe_query.h5ad is the per-line control baseline 03 wrote (50 lines) -> one vector per line.
stack-embedding \
    --checkpoint "$CKPT" \
    --adata tahoe_query.h5ad \
    --genelist "$GENELIST" \
    --gene-name-col feature_name \
    --obs-source tahoe_query.h5ad \
    --device cuda \
    --output "emb_${LABEL}.h5ad"
