#!/bin/bash
#SBATCH --job-name=granularity-prep
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Prepare BOTH granularity controls, then shard 2G's context for the generation array.
#
#   1G  query granularity  -- tahoe_query.h5ad (8 cells/line) collapsed to 1 profile/line,
#                             matching what the organoid rung hands Stack. Isolates the cost of
#                             bulk-ifying the query, measured where a ceiling exists.
#   2G  context granularity -- the Tahoe context collapsed to bulk-like pseudo-cells, so its
#                             granularity matches the L1000 context while the platform stays
#                             Tahoe. Without it the rung-2 Stack arm confounds platform with
#                             granularity, since the L1000 context is bulk wells as pseudo-cells.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) job=${SLURM_JOB_ID:-local}"

echo "=============== 1G: collapse the query to one profile per line ==============="
# tahoe_query.h5ad is indexed by cell POSITION ("0".."399") with the line in obs.cell_line_id,
# so the grouping key is that column. Job 31665454 proved it: --strip-suffix produced 400 groups
# for 400 cells and the aggregator refused rather than writing a no-op.
python scripts/aggregate_query_to_pseudobulk.py \
    --input tahoe_query.h5ad \
    --group-col cell_line_id \
    --out tahoe_query_pseudobulk.h5ad

echo
echo "=============== 2G: collapse the Tahoe context to pseudo-cells ==============="
# The context is already sharded per drug; collapse each shard within cell line so a shard holds
# one profile per (line, condition) instead of many cells, matching the L1000 context's shape.
mkdir -p tahoe_context_pseudobulk_by_drug
for f in context_by_drug/[0-9]*.h5ad; do
    b=$(basename "$f")
    # (line, pert) not line alone: grouping on cell line would average a line's treated cells
    # together with its controls, erasing the treated-vs-control contrast the context exists to
    # carry -- a context that still loads and teaches nothing.
    python scripts/aggregate_query_to_pseudobulk.py \
        --input "$f" --group-col cell_line_id,pert_id \
        --out "tahoe_context_pseudobulk_by_drug/$b"
done
cp context_by_drug/manifest.tsv tahoe_context_pseudobulk_by_drug/manifest.tsv
echo "2G context shards: $(ls tahoe_context_pseudobulk_by_drug/[0-9]*.h5ad | wc -l)"
