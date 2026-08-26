#!/bin/bash
#SBATCH --job-name=gene-panel-audit
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Measure every dataset's gene axis and the panel they share.
#
# SERIAL DELIBERATELY. The standing rule is to parallelise on Alpine, and the exception needs a
# stated reason: this reads only file METADATA -- parquet schemas and HDF5 axis arrays, never a
# matrix -- so the whole job is a handful of seeks. The parallelisable unit is five file reads,
# and array-task setup would cost more than the work it distributes.
#
# L1000 is measured but NOT required in the panel (see --require). Its 12,328-gene matrix is
# INF_mlr12k: 978 measured landmarks plus ~11,350 imputed from them by multiple linear
# regression, per the filename and Subramanian et al. 2017 (Cell 171:1437). Requiring it would
# cut the panel to genes that are mostly a linear function of 978 measurements. Measuring it
# anyway prices that decision instead of assuming it.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1

SCRATCH="/scratch/alpine/$USER"
OUT_DIR="${OUT_DIR:-promote/gene_panel_${SLURM_JOB_ID:-local}}"
GCTX="GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) job=${SLURM_JOB_ID:-local} OUT_DIR=$OUT_DIR"

python scripts/gene_panel_audit.py \
    --source "tahoe=parquet=tahoe_deltas/real_delta.parquet" \
    --source "stack_cytokine=h5ad=generated_agg/5-Azacytidine.h5ad" \
    --source "stack_drug_aligned=h5ad=generated_drug_aligned_agg/5-Azacytidine.h5ad" \
    --source "sciplex=h5ad=$SCRATCH/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad" \
    --source "l1000_inferred=gene_info=GSE92742_Broad_LINCS_gene_info.txt.gz" \
    --source "l1000_gctx_rows=gctx=$GCTX" \
    --cohort "sciplex=h5ad=$SCRATCH/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad" \
    --cohort "l1000=l1000=GSE92742_Broad_LINCS_inst_info.txt.gz" \
    --require "tahoe,stack_cytokine,stack_drug_aligned,sciplex" \
    --out-dir "$OUT_DIR"
echo
echo "panel + sidecar in $OUT_DIR"
