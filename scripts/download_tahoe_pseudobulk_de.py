"""One authenticated bulk download of the Tahoe-100M pseudobulk DE config to scratch.

Extracted from the archived lineage's ``build_tahoe_pseudobulk_deltas.py`` (branch
``rung0-replicate-ceiling-old-lineage`` on origin), whose delta-bundle aggregation
imports rung-1 machinery and arrives with rung 1. This half is rung 0's provenance
chain: it reproduces the 2026-07-24 pull recorded in docs/DATA.md.

The table is a flat 1,026-file shard set with no drug partition; run as ONE process and
authenticate first (``hf auth login``) so the pull is not rate-limited.

    python scripts/download_tahoe_pseudobulk_de.py \\
        --local-dir /scratch/alpine/$USER/tahoe_pseudobulk_de
"""

from __future__ import annotations

import argparse
from pathlib import Path

TAHOE = "tahoebio/Tahoe-100M"
DE = "pseudobulk_differential_expression"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local-dir", required=True, type=Path)
    args = ap.parse_args()

    local = args.local_dir
    existing = [p for p in local.rglob("*.parquet") if DE in str(p)] if local.exists() else []
    if existing:
        print(f"{len(existing)} {DE} parquet already under {local}; nothing to do")
        return
    from huggingface_hub import snapshot_download  # type: ignore  # Alpine-only

    print(f"downloading the {DE} config to {local} (one-time, authenticated) ...")
    snapshot_download(TAHOE, repo_type="dataset", allow_patterns=[f"*{DE}*"], local_dir=str(local))
    got = [p for p in local.rglob("*.parquet") if DE in str(p)]
    print(f"downloaded {len(got)} parquet shards")


if __name__ == "__main__":
    main()
