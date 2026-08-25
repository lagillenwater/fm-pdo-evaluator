#!/bin/bash
#SBATCH --job-name=stack-embed
#SBATCH --partition=ah200             # aa100 has a long queue backlog; ah200 (H200, same
                                       # gpu-normal QoS) is the confirmed-working, shorter-queue
                                       # default for all GPU jobs in this repo (2026-08-18).
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1             # ah200 mixes full and MIG-sliced H200 types; explicit
                                       # type required or the job may land on a partial slice.
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
# Activate the `stack` env by PATH instead of the module system. `module load anaconda`
# fails inside batch jobs (Lmod: module unknown) for reasons NOT established -- see
# ralpine's submit comment; do not assert a cause here. This pattern is the one that is
# PROVEN to work: 10_sarcoma_organoids_2024_context and 12_sarcoma_organoids_2024_score ran with it on 2026-08-22.
# `conda activate` is deliberately not used: this install ships no etc/profile.d/conda.sh.
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONUNBUFFERED=1   # stack-generation is a python console script, so its stdout
                           # block-buffers to the log too; without this a multi-hour GPU

# Record what this run consumed. Most inputs here are literals in the invocation below rather
# than --export variables, so the commit is what pins them: a log line naming the sha lets a
# reader open the exact script that ran. Without it a job log says which stage ran and nothing
# about which version of it, which is why only 11 distinct "Resolved:" lines exist across 453
# logs and most published numbers cannot be traced to their inputs.
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) job=${SLURM_JOB_NAME:-local} TASK=${SLURM_ARRAY_TASK_ID:-0}"

                           # job shows 0 bytes until it exits.

TASK="${SLURM_ARRAY_TASK_ID:-0}"
case "$TASK" in
  0) CKPT="stack-large/bc_large.ckpt"           ; GENELIST="stack-large/basecount_1000per_15000max.pkl"   ; LABEL="base" ;;
  1) # stack-embedding loads into StateICLModel (no gen head), so the aligned checkpoint's head
     # keys fail the strict load -- strip them once to an encoder-only checkpoint and embed that.
     SRC="stack-aligned/bc_large_aligned.ckpt"
     CKPT="stack-aligned/bc_large_aligned_encoder.ckpt"
     GENELIST="stack-aligned/basecount_1000per_15000max.pkl" ; LABEL="aligned"
     [[ -f "$CKPT" ]] || python scripts/strip_ckpt_head.py --in "$SRC" --out "$CKPT" ;;
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
