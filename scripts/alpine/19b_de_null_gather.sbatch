#!/bin/bash
#SBATCH --job-name=de-null-gather
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
PARTS="${PARTS:-promote/de_null_parts}"
OUT="${OUT:-promote/de_permutation_null.csv}"
echo "Resolved: commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) job=${SLURM_JOB_NAME:-local} PARTS=$PARTS OUT=$OUT"
python - "$PARTS" "$OUT" <<'PY'
import sys, glob
import pandas as pd
parts = sorted(glob.glob(f"{sys.argv[1]}/part_*.csv"))
if not parts:
    raise SystemExit(f"no parts under {sys.argv[1]}")
df = pd.concat([pd.read_csv(p) for p in parts]).sort_values(["source", "metric", "null_kind"])
df.to_csv(sys.argv[2], index=False)
print(f"{len(parts)} parts -> {len(df)} rows -> {sys.argv[2]}")
print("sources present:", sorted(df["source"].unique()))
print(df.pivot_table(index=["source","metric"], columns="null_kind", values="p").to_string())
PY

# The Check-2 gather validates against the declared model matrix and this one did not, which is
# how the published Check-1b null covered one of the two Stack checkpoints -- the weaker one --
# without anything flagging it. A concat cannot tell a missing variant from a deliberate choice.
echo
echo "=============== model matrix coverage ==============="
python scripts/check_matrix.py --check check1b --result "$OUT"
