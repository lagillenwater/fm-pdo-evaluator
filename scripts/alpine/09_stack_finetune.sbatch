#!/bin/bash
#SBATCH --job-name=stack-finetune
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00              # training run; raise/lower once epoch time is known
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# DRUG-align Stack: fine-tune the base bc_large.ckpt on sci-Plex (08 output) to add + train the
# ICL generation head (query_pos_embedding + cls) on DRUG perturbations instead of cytokines.
# Not arrayable (single training run); one A100. Bump --gpus + --strategy ddp for multi-GPU.
#
# VERIFY before running (from `stack-finetune --help`, but the data-spec format is not yet known):
#   * --dataset_configs : "Dataset configuration strings" -- the sci-Plex file + its split/control
#       columns, format TBD. Confirm against an arc-stack config example, or pass --config <yaml>.
#   * --sample_size 512 must match generation's set size; --replacement_ratio = 1 - n_kept/512.
#   * checkpoint out lands under --save_dir; feed it to 04 as CKPT=<that>.ckpt for generation.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

module purge
module load anaconda
conda activate stack

SCIPLEX="/scratch/alpine/$USER/sciplex/sciplex_finetune.h5ad"   # from 08
SAVE_DIR="/scratch/alpine/$USER/sciplex_finetune"
mkdir -p "$SAVE_DIR"

stack-finetune \
    --checkpoint_path stack-large/bc_large.ckpt \
    --genelist_path stack-large/basecount_1000per_15000max.pkl \
    --dataset_configs "$SCIPLEX" \
    --sample_size 512 \
    --replacement_ratio 0.1 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --max_epochs 20 \
    --precision bf16-mixed \
    --gpus 1 \
    --num_workers 8 \
    --logger tensorboard \
    --save_dir "$SAVE_DIR" \
    --run_name sciplex-drug-align

# Then generate on Tahoe with the drug-aligned checkpoint (reuses 04's per-drug array):
#   CKPT="$SAVE_DIR"/<best>.ckpt GENELIST=stack-large/basecount_1000per_15000max.pkl \
#   OUTDIR=generated_sciplex sbatch scripts/alpine/04_stack_generate.sbatch
# and score it: --generated-dir generated_sciplex through score_generation_eval.py.
