#!/bin/bash
#SBATCH --job-name=tahoe-context
#SBATCH --partition=amem
#SBATCH --qos=mem-long               # generous ceiling; the pushed-down read is typically ~1-2h
                                      # amem only accepts admin/mem-normal/mem-long -- "long"
                                      # (valid under the old, retired amilan partition) errors
                                      # with "Error 1: The amem partition accepts the following
                                      # QoS values: admin or mem-normal or mem-long"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12            # amem's MaxMemPerCPU=21333MB caps cpus at this --mem;
                                       # matches 08_sciplex_prep.sbatch's own solved ratio
                                       # (240G/12=20000MB/core, under the 21333MB ceiling).
#SBATCH --mem=240G                    # QOSMinMemory floor [checked: sacctmgr show qos mem-normal]
                                       # (08_sciplex_prep.sbatch, via sacctmgr); mem-long's own
                                       # floor is not independently confirmed but a plain 48G
                                       # request failed the same QOSMinMemory check here, so
                                       # applying the same, best-evidenced 240G floor.
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Build the single-cell Tahoe context Stack generation consumes (the pseudobulk
# shortcut does NOT produce this), then pre-split it per drug so 04 can fan out one
# drug per GPU. Single job, NOT an array: one pushed-down read over the parquet already
# covers every target drug (pyarrow prunes non-matching row-groups), so arraying would
# only duplicate work.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

# Activate the `stack` env by PATH instead of the module system. `module load anaconda`
# fails inside batch jobs (Lmod: module unknown) for reasons NOT established -- see
# ralpine's submit comment; do not assert a cause here. This pattern is the one that is
# PROVEN to work: 10_sarcoma_organoids_2024_context and 12_sarcoma_organoids_2024_score ran with it on 2026-08-22.
# `conda activate` is deliberately not used: this install ships no etc/profile.d/conda.sh.
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1                          # flush prints immediately so the heartbeat shows live
export HF_HUB_CACHE="/scratch/alpine/$USER/hf"    # cache large downloads on scratch, not /projects
# do NOT set HF_HOME to scratch -- that hides the token `hf auth login` saved under the HF cache:
export HF_TOKEN="${HF_TOKEN:-$(cat "${XDG_CACHE_HOME:-$HOME/.cache}/huggingface/token" 2>/dev/null || true)}"

# ---- build the context (target drugs + their DMSO_TF controls, capped per cond) ----
# The drug filter is pushed into the parquet read (pyarrow prunes non-matching row-groups), so
# only the target drugs' + DMSO cells are decoded -- complete drug coverage, no full-stream scan,
# so no scan cap is needed. --max-cells-per-cond still caps cells per (line, drug) for memory.
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
    shard = ad.concat([trt, ctl])
    # ad.concat drops var columns by default -> re-attach feature_name so stack-generation's
    # --gene-name-col aligns on symbols instead of falling back to the index; make obs unique.
    shard.var["feature_name"] = [str(v) for v in shard.var_names]
    shard.obs_names_make_unique()
    shard.write_h5ad(outdir / f"{i:04d}.h5ad")
    lines.append(f"{i}\t{p}\t{i:04d}.h5ad")
(outdir / "manifest.tsv").write_text("\n".join(lines) + "\n")

# pert_id (name) -> PubChem CID, from the treated obs
pm = obs.loc[~is_ctl, ["pert_id", "pubchem_cid"]].astype(str).drop_duplicates()
(outdir / "pert_to_cid.tsv").write_text(
    "\n".join(f"{p}\t{c}" for p, c in zip(pm["pert_id"], pm["pubchem_cid"])) + "\n"
)

# query baseline: 8 REAL single control cells per DepMap line (400 total) -- NOT a pseudobulk
# mean. Stack's true generative procedure (--mode mdm, see 04) schedules its query-cell draw per
# step (n_test_cells up to 281 of the fixed self.n_cells=512 pool, at --prompt-ratio 0.25); a
# single pseudobulk row per line can never satisfy that (04's confirmed IndexError root cause:
# the CLI's single-pass pad, `pad = test_indices[:need]`, silently truncates once `need` exceeds
# the pool it draws from). 8/line x 50 lines = 400 clears the sufficient bound (pool >=
# max(n_test_cells) = 281, since the pad always draws from the FULL original pool, not a
# per-batch remainder) with ~40% headroom -- well inside the 200 real controls/line always
# available in tahoe_context.h5ad, so no replacement is needed or used. Fixed seed (0) for
# reproducible cell selection across re-runs.
cid = obs["cell_id"].astype(str).to_numpy()
cln = obs["cell_line_id"].astype(str).to_numpy()
patient = np.where((cid != "") & (cid != "nan"), cid, cln)
ctl_idx = np.flatnonzero(is_ctl)
rng = np.random.default_rng(0)
codes, uniq = pd.factorize(patient[ctl_idx])
sel = np.sort(
    np.concatenate(
        [
            rng.choice(
                ctl_idx[codes == code], size=min(8, int((codes == code).sum())), replace=False
            )
            for code in range(len(uniq))
        ]
    )
)
qx = adata.X[sel]
qx = np.asarray(qx.todense() if sparse.issparse(qx) else qx, dtype=np.float64)
row = qx.sum(axis=1, keepdims=True)
row[row == 0] = 1.0
query = ad.AnnData(X=(qx / row * 1e6).astype("float32"))
query.obs_names = [str(i) for i in range(query.n_obs)]
query.obs["cell_line_id"] = patient[sel]
query.var_names = [str(v) for v in adata.var_names]
query.var["feature_name"] = list(query.var_names)
query.write_h5ad(repo / "tahoe_query.h5ad")
print(
    f"{len(perts)} per-drug shards, {len(pm)} pert->CID rows, "
    f"query baseline {query.n_obs} real cells x {query.n_vars} genes "
    f"({len(uniq)} lines, {query.n_obs / len(uniq):.1f} cells/line avg) -> tahoe_query.h5ad"
)
PY
