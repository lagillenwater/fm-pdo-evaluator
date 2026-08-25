#!/bin/bash
#SBATCH --job-name=stack-score
#SBATCH --partition=acpu      # was amilan; CURC retired that partition (2026-08-24)
#SBATCH --qos=cpu-normal      # acpu accepts only cpu-normal / cpu-long, not normal
#SBATCH --account=CHANGE_ME          # <-- your Alpine allocation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Score the full source ladder -- additive, k-NN, AND Stack -- through both checks.
# --generated-dir adds the Stack rung: its delta is logcpm(generated) - logcpm(query
# baseline), keyed (line, CID) via the pert->CID map, so it runs through both checks on
# equal footing with the baselines. All three inputs come from 03/04.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

# Activate the `stack` env by PATH instead of the module system. `module load anaconda`
# fails inside batch jobs (Lmod: module unknown) for reasons NOT established -- see
# ralpine's submit comment; do not assert a cause here. This pattern is the one that is
# PROVEN to work: 10_sarcoma_organoids_2024_context and 12_sarcoma_organoids_2024_score ran with it on 2026-08-22.
# `conda activate` is deliberately not used: this install ships no etc/profile.d/conda.sh.
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1   # else stdout block-buffers to the log file and a multi-hour
                           # job shows 0 bytes until it exits, making "working" and

# Record what this run consumed. Most inputs here are literals in the invocation below rather
# than --export variables, so the commit is what pins them: a log line naming the sha lets a
# reader open the exact script that ran. Without it a job log says which stage ran and nothing
# about which version of it, which is why only 11 distinct "Resolved:" lines exist across 453
# logs and most published numbers cannot be traced to their inputs.
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) job=${SLURM_JOB_NAME:-local} "

                           # "wedged" indistinguishable (job 31655278, 2026-08-25).

# Inputs are the AGGREGATED / COLLAPSED artifacts, not Stack's raw output. Since faithful
# generation (--mode mdm, 2026-08-18) Stack writes one row PER QUERY CELL, so `generated/`
# holds ~8 rows per (line, drug) and `tahoe_query.h5ad` is cell-indexed. build_generated_deltas
# needs exactly one row per (line, drug) and a line-indexed baseline. Passing the raw dirs got
# as far as Check 1 and then died in restrict_common_support with "no (patient, drug) pairs
# shared across every source and design" (job 31631592, 2026-08-24). Run
# fmharness.stack_aggregate.aggregate_generated_replicates + collapse_query_baseline first.
python scripts/score_generation_eval.py \
    --deltas-bundle tahoe_deltas \
    --generated-dir generated_agg \
    --query-baseline tahoe_query_baseline.h5ad \
    --pert-map context_by_drug/pert_to_cid.tsv \
    --k 10 \
    --n-permutations 1000
