#!/bin/bash
#SBATCH --job-name=sciplex-gene-dx
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
# Two questions the first panel audit raised and could not answer.
#
# 1. sci-Plex reported 110,983 genes including mouse RIKEN clones, for a screen of three human
#    cell lines. Either the file uses a combined reference or the audit read the wrong column.
# 2. What the panel would actually be if L1000 were REQUIRED. The first run only bounded it
#    (l1000 n stack = 8,865); the cost of keeping L1000 should be an exact number, not a bound.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
SCRATCH="/scratch/alpine/$USER"
GCTX="GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) job=${SLURM_JOB_ID:-local}"

echo "=============== 1. sci-Plex gene axis ==============="
python scripts/diagnose_sciplex_genes.py \
    --h5ad "$SCRATCH/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad"

echo
echo "=============== 2. panel WITH L1000 required ==============="
python scripts/gene_panel_audit.py \
    --source "tahoe=parquet=tahoe_deltas/real_delta.parquet" \
    --source "stack_cytokine=h5ad=generated_agg/5-Azacytidine.h5ad" \
    --source "stack_drug_aligned=h5ad=generated_drug_aligned_agg/5-Azacytidine.h5ad" \
    --source "sciplex=h5ad=$SCRATCH/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad" \
    --source "l1000_inferred=gene_info=GSE92742_Broad_LINCS_gene_info.txt.gz" \
    --require "tahoe,stack_cytokine,stack_drug_aligned,sciplex,l1000_inferred" \
    --out-dir "promote/gene_panel_with_l1000_${SLURM_JOB_ID:-local}"
