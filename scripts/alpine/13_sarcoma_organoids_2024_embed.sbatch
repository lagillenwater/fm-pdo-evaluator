#!/bin/bash
#SBATCH --job-name=sarcoma_organoids_2024-embed
#SBATCH --partition=ah200             # same confirmed-working GPU partition as 06/11; module
                                       # load anaconda works here (unlike amem -- see 10/12's
                                       # comment), no PATH bypass needed.
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1             # ah200 mixes full and MIG-sliced H200 types; explicit
                                       # type required or the job may land on a partial slice.
#SBATCH --time=01:00:00
#SBATCH --array=0-1                   # 0 = base (bc_large), 1 = cytokine-aligned -- same
                                       # 2-checkpoint split as 06_stack_embed.sbatch.
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Soragni analogue of 06_stack_embed.sbatch: embed the 17-patient tumor-RNA baseline with the
# base AND cytokine-aligned checkpoints, so a Path-B analogue of score_generation_eval.py's
# --stack-emb can score both as leave-patient-out penalized-regression representations,
# head-to-head with the generation-based stack/additive/pca/nmf sources (2026-08-21, matching
# Check 2's base-vs-aligned embedding comparison). The base checkpoint has no generation head
# (see 04's comment) -- embedding is its ONLY way into a Path-B comparison at all.
#
# stack-embedding: --checkpoint/--adata/--genelist/--output(.h5ad|.npy)/--gene-name-col/--device,
# plus --obs-source to attach the query's obs (the patient ids) so downstream scoring keys on
# them the same way 07_stack_emb_score.sbatch's load_line_matrix does for Tahoe lines.
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
                           # job shows 0 bytes until it exits.

QUERY="data/reference/stack_input_sarcoma.h5ad"   # 17 patients x genes -- the SAME tumor-RNA
                                                    # baseline 10_sarcoma_organoids_2024_context.sbatch seeds
                                                    # generation from (one row per patient, no
                                                    # synthetic replicate pool needed here --
                                                    # embedding is a single forward pass, not
                                                    # --mode mdm's scheduled draw).

TASK="${SLURM_ARRAY_TASK_ID:-0}"
case "$TASK" in
  0) CKPT="stack-large/bc_large.ckpt"           ; GENELIST="stack-large/basecount_1000per_15000max.pkl"   ; LABEL="base" ;;
  1) # stack-embedding loads into StateICLModel (no gen head), so the aligned checkpoint's head
     # keys fail the strict load -- reuse 06's already-stripped encoder-only checkpoint if
     # present (same file both scripts need), else strip it here.
     SRC="stack-aligned/bc_large_aligned.ckpt"
     CKPT="stack-aligned/bc_large_aligned_encoder.ckpt"
     GENELIST="stack-aligned/basecount_1000per_15000max.pkl" ; LABEL="aligned"
     [[ -f "$CKPT" ]] || python scripts/strip_ckpt_head.py --in "$SRC" --out "$CKPT" ;;
  *) echo "no checkpoint for task $TASK -- nothing to do"; exit 0 ;;
esac
echo "task $TASK -> embed Soragni query with '$LABEL' ($CKPT)"

stack-embedding \
    --checkpoint "$CKPT" \
    --adata "$QUERY" \
    --genelist "$GENELIST" \
    --gene-name-col feature_name \
    --obs-source "$QUERY" \
    --device cuda \
    --output "emb_sarcoma_organoids_2024_${LABEL}.h5ad"
