"""Score Stack's rung-2 arm: generation from an L1000-built context instead of Tahoe's.

Decision D3 (`docs/decisions/2026-08-25-ladder-round.md`) chose rebuilding Stack's CONTEXT from
L1000 over swapping its query baseline, because the context is Stack's true analogue of
training data -- the in-context examples it reads a drug's effect from. Generation already ran
for both checkpoints (`generated_l1000ctx_cytokine/`, `generated_l1000ctx_drugaligned/`,
2026-08-25) but the outputs are named by L1000's own Broad `pert_id` (``BRD-K...``) while
`build_generated_deltas` matches filenames against the keys of a ``pert_to_drug`` map. This
builds that map directly from L1000's own `pert_info` (`pert_id`, `pubchem_cid`) -- the
identical join `build_l1000_context.py` already does the other direction (drug name -> pert_id)
for the Tahoe/Soragni corpora, just inverted and simpler here since `pert_info` carries both
columns directly, so no name lookup step is needed. No renaming of the generated files: the
match happens by filename stem against this map's keys, same mechanism Tahoe's own
`stack_cytokine`/`stack_drug_aligned` sources already use with drug-name-keyed files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from fmharness.deltas import build_generated_deltas
from fmharness.evaluation import score_delta_sources
from fmharness.stack_aggregate import aggregate_generated_replicates


def _ncid(x: object) -> str:
    try:
        return str(int(float(x)))
    except (ValueError, TypeError):
        return ""


def brd_to_cid(l1000_dir: Path) -> dict[str, str]:
    """L1000 Broad pert_id -> PubChem CID, in the clean-integer-string format Tahoe's own
    generated sources are keyed by (`context_by_drug/pert_to_cid.tsv`)."""
    pert = pd.read_csv(
        l1000_dir / "GSE92742_Broad_LINCS_pert_info.txt.gz", sep="\t", low_memory=False
    )
    out: dict[str, str] = {}
    for pid, cid in zip(pert["pert_id"], pert["pubchem_cid"], strict=True):
        c = _ncid(cid)
        if c:
            out[str(pid)] = c
    return out


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deltas-bundle", default="tahoe_deltas")
    ap.add_argument("--l1000-dir", default=".")
    ap.add_argument("--query-baseline", default="tahoe_query_baseline.h5ad")
    ap.add_argument(
        "--generated-dir", action="append", required=True,
        help="label=raw_generation_dir (BRD-K-named), repeatable",
    )
    ap.add_argument("--agg-out-dir", default="promote/l1000ctx_agg")
    # Matches the Tahoe-context procedure exactly (docs/tahoe_generation_results.md: "gen_logit
    # < 0 -- swept, not calibrated: Check-1 fidelity is insensitive to this filter").
    ap.add_argument("--gen-logit-threshold", type=float, default=0.0)
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bdir = Path(args.deltas_bundle)
    real_delta = pd.read_parquet(bdir / "real_delta.parquet")
    real_key = pd.read_parquet(bdir / "real_key.parquet")

    pert_to_drug = brd_to_cid(Path(args.l1000_dir))
    print(f"L1000 pert_id -> CID map: {len(pert_to_drug)} entries")

    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    matched_cids: dict[str, list[str]] = {}
    for spec in args.generated_dir:
        label, _, raw_dir = spec.partition("=")
        if not raw_dir:
            raise SystemExit(f"--generated-dir must be label=dir, got {spec!r}")
        agg_dir = Path(args.agg_out_dir) / label
        summary = aggregate_generated_replicates(
            Path(raw_dir), agg_dir, threshold=args.gen_logit_threshold
        )
        n_kept = int(summary["n_kept"].sum()) if len(summary) else 0
        n_dropped_files = int(summary.groupby("pert_id")["dropped"].all().sum()) if len(summary) else 0
        print(
            f"  {label}: aggregated {raw_dir} -> {agg_dir} "
            f"({n_kept} replicates kept, {n_dropped_files} pert_ids dropped entirely)"
        )
        delta, key = build_generated_deltas(agg_dir, Path(args.query_baseline), pert_to_drug)
        sources[label] = (delta, key)
        matched_cids[label] = sorted(set(key["drug"]))
        print(f"  {label}: {delta.shape[0]} (line, drug) rows, {len(matched_cids[label])} distinct drugs")

    table = score_delta_sources(sources, real_delta, real_key, n_hvg=args.n_hvg)
    table.to_csv(args.out_dir / "l1000_context_generation.csv", index=False)
    print("\n=== Stack, generation from an L1000-built context (rung 2's Stack arm, D3) ===")
    print(table.to_string(index=False))

    (args.out_dir / "l1000_context_generation.params.json").write_text(
        json.dumps(
            {
                "git_sha": git_sha(),
                "matched_drug_cids": matched_cids,
                "n_pert_to_drug": len(pert_to_drug),
                "gen_logit_threshold": args.gen_logit_threshold,
                "n_hvg": args.n_hvg,
                "generated_dirs": {
                    s.partition("=")[0]: s.partition("=")[2] for s in args.generated_dir
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.out_dir}/l1000_context_generation.csv")


if __name__ == "__main__":
    main()
