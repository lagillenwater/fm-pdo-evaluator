#!/bin/bash
#SBATCH --job-name=de-null-array
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Array form of the Check-1b permutation null. Serial this is ~2,400 de_fidelity calls
# (6 sources x 2 null kinds x 200 shuffles), each looping 1,600 pairs -- hours.
#
# A plain per-cell split is safe here, UNLIKE Check 2: de_fidelity scores one source's
# (delta, key) against the DE calls with no cross-source term, so a task computing one cell
# gets exactly the value the serial run would give it. Check 2 needed plan/scatter/gather
# because its support restriction couples every representation.
#
# Grid is (source x null_kind); --task-id indexes it. Tasks past the end exit cleanly, so an
# over-wide --array is harmless.
#
#   sbatch --array=0-11%12 scripts/alpine/19_de_null_array.sbatch
#   sbatch --dependency=afterok:<ARRAYJOBID> scripts/alpine/19b_de_null_gather.sbatch
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1

SOURCES="${SOURCES:-tahoe_sources}"
DE_CALLS="${DE_CALLS:-tahoe_de_calls/de_calls.parquet}"
SHUFFLES="${SHUFFLES:-200}"
PARTS="${PARTS:-promote/de_null_parts}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) script=$(basename "${BASH_SOURCE[0]}") SOURCES=$SOURCES DE_CALLS=$DE_CALLS SHUFFLES=$SHUFFLES TASK=$TASK"

mkdir -p "$PARTS"

# The sources dump is shared, so build it once under a lock rather than in every task.
if [ ! -d "$SOURCES" ]; then
  if mkdir "$SOURCES.lock" 2>/dev/null; then
    echo "=== dumping delta sources (this task won the lock) ==="
    python scripts/score_generation_eval.py \
        --deltas-bundle tahoe_deltas --generated-dir generated_agg \
        --query-baseline tahoe_query_baseline.h5ad \
        --pert-map context_by_drug/pert_to_cid.tsv \
        --dump-sources "$SOURCES" --n-permutations 1
    rmdir "$SOURCES.lock"
  else
    echo "=== another task is dumping sources; waiting ==="
    while [ ! -d "$SOURCES" ]; do sleep 20; done
  fi
fi

python scripts/de_permutation_null.py \
    --sources-dir "$SOURCES" \
    --de-calls "$DE_CALLS" \
    --n-shuffles "$SHUFFLES" \
    --task-id "$TASK" \
    --out-csv "$PARTS/part_${TASK}.csv"
