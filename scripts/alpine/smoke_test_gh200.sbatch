#!/bin/bash
#SBATCH --job-name=smoke-gh200
#SBATCH --partition=gh200
#SBATCH --qos=gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Cheap, non-generative smoke test for docs/superpowers/specs/2026-08-18-stack-faithful-generation-
# and-de-metrics-design.md's three open items, batched into one Alpine round-trip before any real
# GPU time is spent on the actual Change-1/Change-2 generation runs:
#
#   1. gh200 vs aarch64 compatibility -- gh200's 2 nodes are confirmed Arch=aarch64 (ARM), NOT the
#      x86_64 aa100 nodes the `stack` conda env has only ever been run on in this project. Conda
#      envs are architecture-specific; this may simply fail to import. If it does, the design's own
#      fallback is aa100 for the real runs -- this job's only purpose is finding that out cheaply
#      (CPU-cost import checks, no model forward pass) rather than discovering it mid-generation.
#   2. self.n_cells for both checkpoints Change 1's schedule math (context_ratio linspace, the
#      n_test_cells/n_base_cells table in the design doc) assumes =512 for. Confirmed in source/
#      config already; this is the direct, file-level check the design doc calls "required before
#      trusting" that table.
#   3. Real per-cell-line sci-Plex control-pool sizes -- the one still-open item in the design doc's
#      sci-Plex methods paragraph (needed to state precisely how often the fine-tuning control-
#      replacement step's with-replacement fallback path triggers).
#
# CONFIRM BEFORE RUNNING (this worktree's own paths, not derivable here):
#   * CKPT_ALIGNED / CKPT_SCIPLEX below -- verify both actually exist at these paths on Alpine
#     before submitting (ls via `ralpine ls`); the sci-Plex one especially, since its filename
#     bakes in a specific val_loss (finetuned-epoch=4-val_loss=5.0847.ckpt per the 2026-08-12 run's
#     own log) that would change if that job is ever re-run.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

module purge
module load anaconda
conda activate stack
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"  # GLIBCXX_3.4.29 fix, same as 04/09

CKPT_ALIGNED="${CKPT_ALIGNED:-stack-aligned/bc_large_aligned.ckpt}"
CKPT_SCIPLEX="${CKPT_SCIPLEX:-/scratch/alpine/$USER/sciplex_finetune/finetuned-epoch=4-val_loss=5.0847.ckpt}"
SCIPLEX_H5AD="${SCIPLEX_H5AD:-/scratch/alpine/$USER/sciplex/finetune_input/sciplex_finetune.h5ad}"
echo "Resolved: CKPT_ALIGNED=$CKPT_ALIGNED CKPT_SCIPLEX=$CKPT_SCIPLEX SCIPLEX_H5AD=$SCIPLEX_H5AD"

echo "=== 1. arch + import check ==="
python -c "
import platform
print('machine:', platform.machine())
import torch
print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
import stack
print('stack import: OK')
from stack.model_loading import load_model_from_checkpoint
print('load_model_from_checkpoint import: OK')
"

echo "=== 2. n_cells for both checkpoints (direct file read, no model construction) ==="
python -c "
import torch
for label, path in [('cytokine-aligned', '$CKPT_ALIGNED'), ('sci-Plex-drug-aligned', '$CKPT_SCIPLEX')]:
    ckpt = torch.load(path, map_location='cpu')
    n_cells = ckpt['hyper_parameters']['model_config'].get('n_cells', '<missing>')
    print(f'{label}: n_cells={n_cells}  ({path})')
"

echo "=== 3. real per-cell-line sci-Plex control-pool sizes ==="
python -c "
import anndata as ad
a = ad.read_h5ad('$SCIPLEX_H5AD', backed='r')
obs = a.obs
print('total cells:', a.n_obs)
ctl = obs[obs['pert_id'].astype(str) == 'control']
print('control cells total:', len(ctl))
print(ctl.groupby('cell_line').size().to_string())
"

echo "smoke test complete"
