#!/bin/bash
#SBATCH --job-name=stack-finetune
#SBATCH --partition=aa100
#SBATCH --qos=gpu-testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00              # training run; raise/lower once epoch time is known
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# DRUG-align Stack: fine-tune the base bc_large.ckpt on sci-Plex (08 output) to add + train the
# ICL generation head (query_pos_embedding + cls) on DRUG perturbations instead of cytokines.
# Not arrayable (single training run); one A100. Bump --gpus + --strategy ddp for multi-GPU.
#
# --dataset_configs for a DRUG dataset (parse_dataset_configs, finetune/utils.py) is colon-separated:
#   drug:PATH:condition_col:cell_line_col:control_condition[:filter_organism[:gene_name_col]]
# controls are identified by a VALUE in the condition column -- 08's prep normalizes them to "control"
# and writes gene symbols to var["feature_name"], so the config below is exact. Other knobs:
#   * --sample_size 512 must match generation's set size; --replacement_ratio = 1 - n_kept/512.
#   * checkpoint out lands under --save_dir; feed it to 04 as CKPT=<that>.ckpt for generation.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

module load anaconda
conda activate stack

SCIPLEX="/scratch/alpine/$USER/sciplex/sciplex_finetune.h5ad"   # from 08
SAVE_DIR="/scratch/alpine/$USER/sciplex_finetune"
mkdir -p "$SAVE_DIR"

stack-finetune \
    --checkpoint_path stack-large/bc_large.ckpt \
    --genelist_path stack-large/basecount_1000per_15000max.pkl \
    --dataset_configs "drug:$SCIPLEX:pert_id:cell_line:control:true:feature_name" \
    --sample_size 512 \
    --replacement_ratio 0.1 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --max_epochs 1 \
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
