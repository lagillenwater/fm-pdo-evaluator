#!/bin/bash
#SBATCH --job-name=l1000-imp-fidelity
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Do L1000's imputed genes carry real delta signal, or only the landmarks' shadow?
#
# Serial deliberately. The standing rule is to parallelise on Alpine and the exception needs a
# reason: the cost here is ONE pass over the .gctx restricted to the shared lines and drugs,
# which is a few thousand wells out of 1.3M. Splitting by pair would re-open and re-seek the
# same .gctx per task, and the DMSO baseline is shared across all pairs of a line -- a per-pair
# split would recompute it identically N times. The work after the read is seconds.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
J="${SLURM_JOB_ID:-local}"
GCTX="${GCTX:-GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx}"
OUT_DIR="${OUT_DIR:-promote/l1000_imputation_fidelity_$J}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) job=$J GCTX=$GCTX OUT_DIR=$OUT_DIR"

python scripts/l1000_imputation_fidelity.py \
    --l1000-dir . \
    --gctx "$GCTX" \
    --deltas-bundle tahoe_deltas \
    --pert-map context_by_drug/pert_to_cid.tsv \
    --time 24.0 \
    --n-perm 200 \
    --out-dir "$OUT_DIR"
