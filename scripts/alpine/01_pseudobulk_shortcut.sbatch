#!/bin/bash
#SBATCH --job-name=tahoe-pull
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                    # per shard; 8 shards -> ~1/8 the single-run peak
#SBATCH --time=08:00:00
#SBATCH --array=0-7                  # 8 drug shards; each pulls its slice of the DE table
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# The pseudobulk shortcut, fanned out over drugs. Each array task pulls ONLY its
# round-robin slice of Tahoe's pseudobulk_differential_expression table (pyarrow
# predicate pushdown on drug), so peak memory and the row-materialize are divided
# across tasks instead of one ~1B-row query. 02 merges the shard bundles.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

module load anaconda
conda activate stack           # data/scoring env (no torch)
export PYTHONPATH="$REPO/src"
export HF_HOME="/scratch/alpine/$USER/hf"

NSHARDS="${SLURM_ARRAY_TASK_COUNT:-8}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
CIDFILE="$REPO/data/static/tahoe_target_cids.txt"  # written by 00 (Tahoe-mapped CIDs only)
SHARDDIR="$REPO/tahoe_deltas/shard_${TASK}"
SHARDCIDS="$REPO/tahoe_deltas/cids_${TASK}.txt"
mkdir -p "$REPO/tahoe_deltas"

# round-robin the CID list so mapped drugs are spread evenly across shards.
mapfile -t ALL < <(tr ' \t' '\n\n' < "$CIDFILE" | grep .)
: > "$SHARDCIDS"
for i in "${!ALL[@]}"; do
  if (( i % NSHARDS == TASK )); then echo "${ALL[$i]}" >> "$SHARDCIDS"; fi
done
echo "shard $TASK/$NSHARDS: $(wc -l < "$SHARDCIDS") drugs"

python scripts/build_tahoe_pseudobulk_deltas.py \
    --drugs-cid-file "$SHARDCIDS" \
    --out-dir "$SHARDDIR"
