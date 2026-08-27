#!/bin/bash
#SBATCH --job-name=smoke-env
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Cheap, non-generative smoke test for
# docs/tasks/stack-faithful-generation-and-de-metrics/design.md's three open items,
# batched into one Alpine round-trip before any real
# GPU time is spent on the actual Change-1/Change-2 generation runs:
#
#   1. GPU partition/env sanity -- confirms the `stack` conda env actually imports and sees a GPU
#      on whichever partition this submits to, before spending real time on generation there.
#   2. self.n_cells for both checkpoints Change 1's schedule math (context_ratio linspace, the
#      n_test_cells/n_base_cells table in the design doc) assumes =512 for. Confirmed in source/
#      config already; this is the direct, file-level check the design doc calls "required before
#      trusting" that table.
#   3. Real per-cell-line sci-Plex control-pool sizes -- the one still-open item in the design doc's
#      sci-Plex methods paragraph (needed to state precisely how often the fine-tuning control-
#      replacement step's with-replacement fallback path triggers).
#
# PARTITION HISTORY (2026-08-18): first tried gh200 (2 idle ARM/aarch64 nodes) -- rejected at
# submission with "Invalid qos specification". Confirmed via CURC's own docs
# (curc.readthedocs.io/en/latest/clusters/alpine/alpine-hardware.html) that gh200 is
# request-only access (a support-request form grants the QoS at the account level; the partition
# listing it as an allowed QoS is not sufficient on its own) -- not a syntax issue, don't retry
# gh200 without first filing that request. `ah200` (H200 GPUs, x86_64 [checked: scontrol show partition ah200], confirmed via `scontrol
# show node`) is a DIFFERENT partition that uses this project's already-proven `gpu-normal` QoS
# (same as aa100) -- no special access needed. Its nodes carry a MIX of full and MIG-sliced H200
# GPU types (`gpu:h200` vs `gpu:h200_2g.35gb` / `gpu:h200_3g.71gb`), so an ah200 submission needs
# an explicit, non-MIG GRES type or it may land on a partial-memory slice (the #SBATCH directives
# above already specify it).
#
# 2026-08-18, later: aa100 was carrying a long queue backlog while ah200 ran near-immediately
# (confirmed job 31418001) [checked: sacct -j 31418001], so ah200 is now this repo's default
# smoke test -- see 04_stack_generate.sbatch, 06_stack_embed.sbatch, 09_stack_finetune.sbatch.
# To fall back to aa100 (e.g. if ah200 itself backs up later), override on the command line --
# sbatch CLI flags take precedence over in-script #SBATCH lines:
#
#   sbatch --job-name=smoke-aa100 --partition=aa100 --qos=gpu-normal --gres=gpu:1 \
#       scripts/alpine/smoke_test_env.sbatch
#
# CONFIRM BEFORE RUNNING (this worktree's own paths, not derivable here):
#   * CKPT_ALIGNED / CKPT_SCIPLEX below -- verify both actually exist at these paths on Alpine
#     before submitting (ls via `ralpine ls`); the sci-Plex one especially, since its filename
#     bakes in a specific val_loss (finetuned-epoch=4-val_loss=5.0847.ckpt per the 2026-08-12 run's
#     own log) that would change if that job is ever re-run.
#
# 2026-08-18: first ah200 attempt hit the (then) 20-minute time limit having printed nothing past
# this script's own bash `echo` lines -- not even step 1's first `print('machine: ...')`, which
# should be near-instant. That is Python's default full stdout buffering when writing to a file
# (unlike a terminal): any output from a python -c block sits in an in-process buffer until it
# flushes or the process exits normally, so a SIGTERM from the time limit discards it whole,
# whether the script was hung or just slow. This does NOT distinguish "ah200 import hangs" from
# "ah200 import is merely slow" -- fixed below (PYTHONUNBUFFERED + explicit timestamps per step),
# time limit raised to 30 min as a safety margin, not because 20 was necessarily insufficient.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PYTHONUNBUFFERED=1   # see the 2026-08-18 note above -- without this, output from any
                             # python -c block below is invisible in the log until that block
                             # exits normally; a time-limit kill would discard it silently.

module purge
# Activate the `stack` env by PATH instead of the module system. `module load anaconda`
# fails inside batch jobs (Lmod: module unknown) for reasons NOT established -- see
# ralpine's submit comment; do not assert a cause here. This pattern is the one that is
# PROVEN to work: 10_sarcoma_organoids_2024_context and 12_sarcoma_organoids_2024_score ran with it on 2026-08-22.
# `conda activate` is deliberately not used: this install ships no etc/profile.d/conda.sh.
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"  # GLIBCXX_3.4.29 fix, same as 04/09

echo "$(date -Is) partition=$SLURM_JOB_PARTITION qos=$SLURM_JOB_QOS host=$(hostname) job=$SLURM_JOB_ID"

CKPT_ALIGNED="${CKPT_ALIGNED:-stack-aligned/bc_large_aligned.ckpt}"
CKPT_SCIPLEX="${CKPT_SCIPLEX:-/scratch/alpine/$USER/sciplex_finetune/finetuned-epoch=4-val_loss=5.0847.ckpt}"
SCIPLEX_H5AD="${SCIPLEX_H5AD:-/scratch/alpine/$USER/sciplex/finetune_input/sciplex_finetune.h5ad}"
echo "$(date -Is) Resolved: CKPT_ALIGNED=$CKPT_ALIGNED CKPT_SCIPLEX=$CKPT_SCIPLEX SCIPLEX_H5AD=$SCIPLEX_H5AD"

echo "$(date -Is) === 1. arch + import check ==="
python -u -c "
import time
def log(msg):
    print(f'{time.strftime(\"%Y-%m-%dT%H:%M:%S\")} {msg}', flush=True)

import platform
log(f'machine: {platform.machine()}')
log('importing torch...')
import torch
log(f'torch: {torch.__version__}  cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    log(f'device: {torch.cuda.get_device_name(0)}')
log('importing stack...')
import stack
log('stack import: OK')
from stack.model_loading import load_model_from_checkpoint
log('load_model_from_checkpoint import: OK')
"

echo "$(date -Is) === 2. n_cells for both checkpoints (direct file read, no model construction) ==="
python -u -c "
import torch
for label, path in [('cytokine-aligned', '$CKPT_ALIGNED'), ('sci-Plex-drug-aligned', '$CKPT_SCIPLEX')]:
    ckpt = torch.load(path, map_location='cpu')
    n_cells = ckpt['hyper_parameters']['model_config'].get('n_cells', '<missing>')
    print(f'{label}: n_cells={n_cells}  ({path})', flush=True)
"

echo "$(date -Is) === 3. real per-cell-line sci-Plex control-pool sizes ==="
python -u -c "
import anndata as ad
a = ad.read_h5ad('$SCIPLEX_H5AD', backed='r')
obs = a.obs
print('total cells:', a.n_obs, flush=True)
ctl = obs[obs['pert_id'].astype(str) == 'control']
print('control cells total:', len(ctl), flush=True)
print(ctl.groupby('cell_line').size().to_string(), flush=True)
"

echo "$(date -Is) smoke test complete"
