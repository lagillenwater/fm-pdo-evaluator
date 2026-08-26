#!/bin/bash
#SBATCH --job-name=l1000-context-tahoe
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Build Stack's PERTURBATION CONTEXT from L1000 instead of Tahoe, over Tahoe's drug panel.
#
# This is what gives Stack a real cross-platform arm at rung 2. The context (--base-adata) is
# the in-context learning corpus -- the examples Stack reads a drug's effect from -- so it is
# the direct analogue of the training data the pca/nmf/knn maps are fit on. Swapping it from
# Tahoe to L1000 poses exactly the question rung 2 asks of the baselines, while the query line
# stays Tahoe in both arms.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
GCTX="${GCTX:-GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx}"
OUT="${OUT:-l1000_context_tahoe.h5ad}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) job=${SLURM_JOB_ID:-local} OUT=$OUT"
python scripts/build_l1000_context.py \
    --l1000-dir . --gctx "$GCTX" \
    --pert-map context_by_drug/pert_to_cid.tsv \
    --cell-lines all \
    --out "$OUT"
