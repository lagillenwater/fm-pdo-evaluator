#!/bin/bash
#SBATCH --job-name=stack-gen
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00             # smoke test measured ~11-14min cold-cache import/checkpoint-
                                     # load cost per task on ah200 (job 31418001); --mode mdm's
                                     # fuller per-step schedule may also cost more than vanilla's
                                     # fixed ratio -- confirm on Task 7's first real run and raise
                                     # this if any array task approaches the limit.
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
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"  # GLIBCXX_3.4.29 fix
export HF_HOME="/scratch/alpine/$USER/hf"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
SHARD=$(printf "context_by_drug/%04d.h5ad" "$TASK")
if [[ ! -f "$SHARD" ]]; then
  echo "no shard for task $TASK ($SHARD) -- past the drug count, nothing to do"; exit 0
fi
PERT=$(awk -F'\t' -v i="$TASK" '$1==i{print $2}' context_by_drug/manifest.tsv)
echo "task $TASK -> pert '$PERT' ($SHARD)"

QUERY="tahoe_query.h5ad"              # 400 real control cells, 8/line (written by 03)
# Env-overridable so the base (unaligned) checkpoint can be A/B'd against the aligned one, e.g.:
#   sbatch --export=CKPT=stack-large/bc_large.ckpt,GENELIST=stack-large/basecount_1000per_15000max.pkl,OUTDIR=generated_base \
#       --array=3 scripts/alpine/04_stack_generate.sbatch
# `VAR=val sbatch ...` prefix assignment does NOT reach this script on Alpine's Slurm config --
# it must be `sbatch --export=VAR=val,... script`, WITHOUT "ALL": --export=ALL propagates the
# submitting shell's entire environment (including $MODULEPATH/Lmod state), which broke `module
# load anaconda` outright ("unknown module") in production 2026-08-12. Plain --export=VAR=val,...
# (no ALL) gives the job a clean environment plus just the named vars, matching this script's own
# `module purge; module load anaconda` expectations. Also caught 2026-08-12: a prefix-assigned
# CKPT/OUTDIR (no --export at all) silently no-op'd, regenerating the cytokine-aligned
# checkpoint's output into the default generated/ dir instead of the intended drug-aligned run --
# logged loudly here so a repeat is the first thing in every array task's log, not something to
# notice 30 lines in.
CKPT="${CKPT:-stack-aligned/bc_large_aligned.ckpt}"
GENELIST="${GENELIST:-stack-aligned/basecount_1000per_15000max.pkl}"
OUTDIR="${OUTDIR:-generated}"
echo "Resolved: CKPT=$CKPT GENELIST=$GENELIST OUTDIR=$OUTDIR"
mkdir -p "$OUTDIR"

# Stack's true in-context generative procedure (the CLI's own default, --mode mdm, NOT --mode
# vanilla): T=5 steps, context_ratio SCHEDULED via linspace(context_ratio_min, context_ratio, 5)
# with confidence-guided selective unmasking carried between steps (Methods 4.2.5 of the Stack
# preprint) -- this is what the paper actually describes; --mode vanilla is a materially simpler
# fixed-ratio baseline the CLI happens to also expose, not "one-shot generation." n_test_cells =
# max(1, int(512*(1-prompt_ratio-context_ratio))) is recomputed per step under the SCHEDULED
# context_ratio, ranging 179-333 cells across these 5 steps; the 400-real-cell tahoe_query.h5ad
# (03) covers every step's n_test_cells with margin (see 03's own comment for why 400 suffices).
# This pipeline previously used --mode vanilla specifically to dodge an IndexError at the old
# 50-row pseudobulk query; that workaround is no longer needed now the query pool itself is large
# enough for the model's own default mode. Explicit rather than left to CLI defaults, matching
# this repo's "log resolved CKPT/GENELIST/OUTDIR" reproducibility discipline and guarding against
# a future package upgrade silently changing a default.
#
# Output is now MULTIPLE rows per line (one per surviving real query cell, each carrying
# obs["gen_logit"]) instead of one -- run
# fmharness.stack_aggregate.aggregate_generated_replicates (Task 3) on $OUTDIR before
# build_generated_deltas/delta_fidelity, which still expect exactly one row per (line, drug).

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
