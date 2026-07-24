#!/bin/bash
#SBATCH --job-name=stack-score
#SBATCH --partition=amilan
#SBATCH --qos=normal
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

module load anaconda
conda activate stack
export PYTHONPATH="$REPO/src"

python scripts/score_generation_eval.py \
    --deltas-bundle tahoe_deltas \
    --generated-dir generated \
    --query-baseline tahoe_query.h5ad \
    --pert-map context_by_drug/pert_to_cid.tsv \
    --k 10 \
    --n-permutations 1000
