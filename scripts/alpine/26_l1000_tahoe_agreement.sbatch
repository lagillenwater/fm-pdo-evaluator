#!/bin/bash
#SBATCH --job-name=l1000-tahoe-agree
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Why is L1000-vs-Tahoe delta agreement ~zero even for the 978 MEASURED genes?
# Establishes the noise ceiling (L1000 split-half), direction (sign concordance vs 50% chance)
# and magnitude (median |delta| per platform). The noise ceiling bounds everything else and
# should have been measured before job 31661570, not after it.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
J="${SLURM_JOB_ID:-local}"
GCTX="${GCTX:-GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx}"
OUT_DIR="${OUT_DIR:-promote/l1000_tahoe_agreement_$J}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) job=$J GCTX=$GCTX OUT_DIR=$OUT_DIR"

python scripts/l1000_tahoe_agreement_diagnosis.py \
    --l1000-dir . \
    --gctx "$GCTX" \
    --deltas-bundle tahoe_deltas \
    --pert-map context_by_drug/pert_to_cid.tsv \
    --time 24.0 \
    --top-n 100 \
    --out-dir "$OUT_DIR"
