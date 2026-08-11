#!/bin/bash
#SBATCH --job-name=stack-emb-score
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
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

module load anaconda
conda activate stack
export PYTHONPATH="$REPO/src"

python scripts/score_generation_eval.py \
    --deltas-bundle tahoe_deltas \
    --generated-dir generated \
    --query-baseline tahoe_query.h5ad \
    --pert-map context_by_drug/pert_to_cid.tsv \
    --stack-emb base=emb_base.h5ad aligned=emb_aligned.h5ad \
    --k 10 \
    --n-permutations 1000
