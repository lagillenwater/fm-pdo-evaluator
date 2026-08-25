#!/bin/bash
#SBATCH --job-name=stack-emb-score
#SBATCH --partition=acpu      # was amilan; CURC retired that partition (2026-08-24)
#SBATCH --qos=cpu-normal      # acpu accepts only cpu-normal / cpu-long, not normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Score the base + cytokine-aligned Stack embeddings as check-2 representations, in the same
# penalized grid as expr/pca/nmf and the generated delta -- the base's only way into the eval
# (no generation head). Run AFTER 06 (emb_base.h5ad, emb_aligned.h5ad) and 04 (generated/).
# CPU only; single job. --stack-emb takes label=path pairs (score_generation_eval loads .h5ad).
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

# Activate the `stack` env by PATH instead of the module system. `module load anaconda`
# fails inside batch jobs (Lmod: module unknown) for reasons NOT established -- see
# ralpine's submit comment; do not assert a cause here. This pattern is the one that is
# PROVEN to work: 10_soragni_context and 12_soragni_score ran with it on 2026-08-22.
# `conda activate` is deliberately not used: this install ships no etc/profile.d/conda.sh.
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"

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
    --stack-emb base=emb_base.h5ad aligned=emb_aligned.h5ad \
    --k 10 \
    --n-permutations 1000
