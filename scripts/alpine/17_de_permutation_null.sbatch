#!/bin/bash
#SBATCH --job-name=de-perm-null
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Rebuild Check 1b's permutation null, which Finding 5 rests on and whose script was never
# committed. Two nulls are computed, not one: the published version shuffled the whole key,
# so with ~32 drugs a random pair has the wrong drug ~97% of the time and clearing it shows
# DRUG-specificity, not line-specificity. A within-drug patient shuffle holds drug identity
# fixed and tests the per-line claim the finding actually makes. Every source is nulled, not
# just the two Stack checkpoints, so lifts compare like with like.
#
# Requires 16_tahoe_de_calls.sbatch first (the DE bundle) and a --dump-sources run of
# score_generation_eval.py (the per-source delta parquets). Neither existed before 2026-08-25.
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
OUT="${OUT:-promote/de_permutation_null.csv}"
echo "Resolved: SOURCES=$SOURCES DE_CALLS=$DE_CALLS SHUFFLES=$SHUFFLES OUT=$OUT"

mkdir -p "$(dirname "$OUT")"

# Step 1: materialise the delta sources, which nothing else emits.
if [ ! -d "$SOURCES" ]; then
  echo "=== dumping delta sources to $SOURCES ==="
  python scripts/score_generation_eval.py \
      --deltas-bundle tahoe_deltas \
      --generated-dir generated_agg \
      --query-baseline tahoe_query_baseline.h5ad \
      --pert-map context_by_drug/pert_to_cid.tsv \
      --dump-sources "$SOURCES" \
      --n-permutations 1
fi

# Step 2: the null itself.
echo "=== permutation null ==="
python scripts/de_permutation_null.py \
    --sources-dir "$SOURCES" \
    --de-calls "$DE_CALLS" \
    --n-shuffles "$SHUFFLES" \
    --out-csv "$OUT"
