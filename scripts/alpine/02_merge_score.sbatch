#!/bin/bash
#SBATCH --job-name=tahoe-baselines
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Merge the shard bundles from 01 into one tahoe_deltas/ bundle, then score the
# additive + k-NN baselines (both checks). Kept separate from 01 so you can re-score
# with different --k / --n-hvg / --n-permutations without re-pulling from HF.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

module load anaconda
conda activate stack
export PYTHONPATH="$REPO/src"

# ---- merge shard_*/ into tahoe_deltas/ --------------------------------------------
# real_delta/real_key concatenate row-for-row (shards hold disjoint (line,drug) pairs);
# columns are unioned and NaN->0 (a gene absent from a drug's DE table == no change).
# base is per-line baseMean averaged across shards (a k-NN neighbor proxy only).
python - <<'PY'
import glob
from pathlib import Path
import pandas as pd

repo = Path.cwd()
out = repo / "tahoe_deltas"
shards = sorted(glob.glob(str(out / "shard_*")))
if not shards:
    raise SystemExit("no shard_*/ dirs found -- did 01 run?")
print(f"merging {len(shards)} shards")

rd = pd.concat([pd.read_parquet(Path(s) / "real_delta.parquet") for s in shards]).fillna(0.0)
rk = pd.concat([pd.read_parquet(Path(s) / "real_key.parquet") for s in shards])
# real_delta and real_key were concatenated in the SAME shard order -> still row-aligned.
rd = rd.reset_index(drop=True)
rk = rk.reset_index(drop=True)
base = pd.concat([pd.read_parquet(Path(s) / "base.parquet") for s in shards]).groupby(level=0).mean()
base.index = base.index.astype(str)

rd.to_parquet(out / "real_delta.parquet", index=False)
rk.to_parquet(out / "real_key.parquet", index=False)
base.fillna(0.0).to_parquet(out / "base.parquet")
print(f"wrote {out}: {len(rk)} (line, drug) pairs, {base.shape[0]} lines, {rd.shape[1]} genes")
PY

# ---- score the baselines (additive + k-NN) ----------------------------------------
python scripts/score_generation_eval.py \
    --deltas-bundle tahoe_deltas \
    --k 10 \
    --n-permutations 1000
