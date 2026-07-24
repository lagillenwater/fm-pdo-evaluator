#!/bin/bash
#SBATCH --job-name=tahoe-context
#SBATCH --partition=amilan
#SBATCH --qos=long                   # streaming full-scan of Tahoe expression_data can exceed 24h
#SBATCH --account=CHANGE_ME          # <-- your Alpine allocation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                    # bounded by --batch; the stream is not held in memory
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Build the single-cell Tahoe context Stack generation consumes (the pseudobulk
# shortcut does NOT produce this), then pre-split it per drug so 04 can fan out one
# drug per GPU. Single job, NOT an array: `datasets` streaming full-scans regardless
# of filter, so arraying this would mean N redundant scans of the 100M-cell stream.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

module load anaconda
conda activate stack               # data env: fmharness + datasets
export PYTHONPATH="$REPO/src"
export HF_HOME="/scratch/alpine/$USER/hf"

# ---- build the context (target drugs + their DMSO_TF controls, capped per cond) ----
python scripts/build_tahoe_context.py \
    --drugs-cid-file data/static/tahoe_target_cids.txt \
    --max-cells-per-cond 200 \
    --out tahoe_context.h5ad

# ---- pre-split + emit the query baseline and pert->CID map for 04/05 ----------------
# per-drug shards: each = one drug's treated cells + ALL DMSO controls (the prompt);
#   manifest.tsv maps array index -> pert_id so 04 and the generated/ files line up.
# tahoe_query.h5ad: per-DepMap-line control pseudobulk on the Stack input scale (CPM);
#   this is 04's --test-adata (the QUERY) and 05's --query-baseline (delta = gen - it).
# pert_to_cid.tsv: pert_id (drug name) -> PubChem CID, so 05 re-keys generated deltas.
python - <<'PY'
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from fmharness.deltas import _group_mean

repo = Path.cwd()
adata = ad.read_h5ad(repo / "tahoe_context.h5ad")
obs = adata.obs
is_ctl = obs["is_control"].to_numpy(dtype=bool)
outdir = repo / "context_by_drug"
outdir.mkdir(exist_ok=True)

# per-drug context shards
ctl = adata[is_ctl]
perts = sorted(set(obs.loc[~is_ctl, "pert_id"].astype(str)))
lines = []
for i, p in enumerate(perts):
    trt = adata[(obs["pert_id"].astype(str) == p).to_numpy()]
    ad.concat([trt, ctl]).write_h5ad(outdir / f"{i:04d}.h5ad")
    lines.append(f"{i}\t{p}\t{i:04d}.h5ad")
(outdir / "manifest.tsv").write_text("\n".join(lines) + "\n")

# pert_id (name) -> PubChem CID, from the treated obs
pm = obs.loc[~is_ctl, ["pert_id", "pubchem_cid"]].astype(str).drop_duplicates()
(outdir / "pert_to_cid.tsv").write_text(
    "\n".join(f"{p}\t{c}" for p, c in zip(pm["pert_id"], pm["pubchem_cid"])) + "\n"
)

# query baseline: control pseudobulk per DepMap line, renormalized to 1e6 (Stack scale)
cid = obs["cell_id"].astype(str).to_numpy()
cln = obs["cell_line_id"].astype(str).to_numpy()
patient = np.where((cid != "") & (cid != "nan"), cid, cln)
x = adata.X
xc = x if sparse.issparse(x) else sparse.csr_matrix(np.asarray(x, dtype=np.float64))
codes, uniq = pd.factorize(patient[np.flatnonzero(is_ctl)])
qb = _group_mean(xc[np.flatnonzero(is_ctl)], codes, len(uniq))
row = qb.sum(axis=1, keepdims=True)
row[row == 0] = 1.0
query = ad.AnnData(X=(qb / row * 1e6).astype("float32"))
query.obs_names = [str(u) for u in uniq]
query.var_names = [str(v) for v in adata.var_names]
query.var["feature_name"] = list(query.var_names)
query.write_h5ad(repo / "tahoe_query.h5ad")
print(
    f"{len(perts)} per-drug shards, {len(pm)} pert->CID rows, "
    f"query baseline {query.n_obs} lines x {query.n_vars} genes -> tahoe_query.h5ad"
)
PY
