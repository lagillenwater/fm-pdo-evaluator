#!/bin/bash
#SBATCH --job-name=tahoe-de-calls
#SBATCH --partition=amem
#SBATCH --qos=mem-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12          # amem 240G/12cpu floor [checked: sacctmgr show qos mem-normal]
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Build the ground-truth DE calls Check 1b scores against. Wilcoxon rank-sum + BH per
# (line, drug, gene) over the raw per-cell context -- ~1,650 pairs, one-time compute, cached
# like the tahoe_deltas/ bundle.
#
# This exists because the bundle it produces was missing entirely: tahoe_de_calls/ is in
# neither git (.gitignore) nor on the cluster, so Check 1b's published numbers cannot be
# regenerated and scripts/de_permutation_null.py cannot run at all. The builder's own docstring
# says it needs no Alpine submission because the context "is already local" -- it is not, it is
# a 2.6 GB gitignored file that lives here.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

# Activate the `stack` env by PATH instead of the module system -- the pattern proven to work.
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1

CONTEXT="${CONTEXT:-tahoe_context.h5ad}"
OUT="${OUT:-tahoe_de_calls/de_calls.parquet}"
echo "Resolved: CONTEXT=$CONTEXT OUT=$OUT LFC=${LFC:-0.25} FDR=${FDR:-0.05}"

python scripts/build_tahoe_de_calls.py \
    --context "$CONTEXT" \
    --out "$OUT" \
    --lfc-threshold "${LFC:-0.25}" \
    --fdr-threshold "${FDR:-0.05}"
