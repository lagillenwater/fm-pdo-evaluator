#!/bin/bash
#SBATCH --job-name=drug-overlap
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
# Can Tahoe actually supply the organoids' drugs? Path B learns a baseline -> delta map on a
# corpus of real perturbations and applies it to organoids, so gene coverage decides what can
# be scored but DRUG overlap decides whether the corpus is usable at all. L1000 has 25,157
# perturbagens and 978 measured genes; Tahoe has a full transcriptome and a far smaller panel.
# That trade cannot be read off the gene tables.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
J="${SLURM_JOB_ID:-local}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) job=$J"

python scripts/drug_overlap.py \
    --source "tahoe=tahoe_deltas/real_key.parquet" \
    --source "organoid_screen=data/raw/soragni/tables/drug_screen.parquet" \
    --source "gdsc2_compounds=data/raw/gdsc2_sarcoma/gdsc2/screened_compounds_rel_8.5.csv" \
    --source "tahoe_pertmap=context_by_drug/pert_to_cid.tsv" \
    --source "l1000_perts=GSE92742_Broad_LINCS_pert_info.txt.gz" \
    --generated-dir "tahoe_generated=generated_agg" \
    --generated-dir "organoid_generated=generated_sarcoma_organoids_2024_drug_aligned_agg" \
    --out-dir "promote/drug_overlap_$J"
