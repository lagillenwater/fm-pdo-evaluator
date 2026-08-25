#!/bin/bash
#SBATCH --job-name=soragni-context
#SBATCH --partition=amem
#SBATCH --qos=mem-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12           # amem's MaxMemPerCPU=21333MB caps cpus at this --mem;
                                      # matches 08_sciplex_prep.sbatch's own solved ratio
                                      # (240G/12=20000MB/core, under the 21333MB ceiling).
#SBATCH --mem=240G                   # QOSMinMemory: mem-normal's floor is confirmed 240G
                                      # (03_stack_context.sbatch, 08_sciplex_prep.sbatch, via
                                      # sacctmgr) -- amem enforces this regardless of actual
                                      # job need.
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Path B faithful-generation prep: split the already-built l1000_context.h5ad into one
# per-drug shard (mirrors 03_stack_context.sbatch's Tahoe splitting), and build the
# synthetic query pool 11 needs for --mode mdm (Soragni's real query is bulk, one row
# per patient -- see build_soragni_query_pool.py for why a real-cell pool isn't possible
# here the way 03 built one for Tahoe).
# CONFIRM BEFORE RUNNING:
#   * l1000_context.h5ad already exists (scripts/build_l1000_context.py) -- this job does
#     NOT rebuild it from the raw .gctx.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

# Activate the `stack` env by PATH instead of the module system. `module load anaconda`
# fails inside batch jobs (Lmod: module unknown); the cause is NOT established, so do
# not assert one here. Earlier text claiming it was "gone from amem's module tree" was
# never verified -- corrected 2026-08-24. This pattern is what is PROVEN to work.
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1

CONTEXT="${CONTEXT:-l1000_context.h5ad}"
OUTDIR="context_by_drug_soragni"

# ---- split l1000_context.h5ad per drug (treated cells for that drug + ALL DMSO controls) --
python - <<PY
from pathlib import Path

import anndata as ad

repo = Path.cwd()
adata = ad.read_h5ad(repo / "$CONTEXT")
obs = adata.obs
is_ctl = obs["is_control"].to_numpy(dtype=bool)
outdir = repo / "$OUTDIR"
outdir.mkdir(exist_ok=True)

ctl = adata[is_ctl]
perts = sorted(set(obs.loc[~is_ctl, "pert_id"].astype(str)))
lines = []
for i, p in enumerate(perts):
    trt = adata[(obs["pert_id"].astype(str) == p).to_numpy()]
    shard = ad.concat([trt, ctl])
    shard.var["feature_name"] = [str(v) for v in shard.var_names]
    shard.obs_names_make_unique()
    shard.write_h5ad(outdir / f"{i:04d}.h5ad")
    lines.append(f"{i}\t{p}\t{i:04d}.h5ad")
(outdir / "manifest.tsv").write_text("\n".join(lines) + "\n")
print(f"{len(perts)} per-drug shards -> {outdir}/")
PY

# ---- synthetic per-patient replicate pool for --mode mdm's scheduled draw -------------
python scripts/build_soragni_query_pool.py \
    --baseline data/reference/stack_input_sarcoma.h5ad \
    --out soragni_query_pool.h5ad \
    --n-replicates 24 \
    --library-size 5000 \
    --seed 0
