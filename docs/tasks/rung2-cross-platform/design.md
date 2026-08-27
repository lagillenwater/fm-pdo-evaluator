# Rung 2 — cross-platform (map fit on L1000, tested on Tahoe)

**Status** OPEN — promoted with valid controls; one reproducibility defect outstanding.
**Steps** build, restrict, split, fit, score, null, promote.
**Parent** [`docs/PROJECT_SPEC.md`](../../PROJECT_SPEC.md); **state** [`docs/PROJECT_STATE.md`](../../PROJECT_STATE.md).
**Design source** [`docs/transfer_ladder_protocol.md`](../../transfer_ladder_protocol.md), rung 2, plus decision D3 in [`docs/decisions/2026-08-25-ladder-round.md`](../../decisions/2026-08-25-ladder-round.md).

## What this rung establishes

What crossing a measurement platform costs: the same mapping fitted on L1000 instead of Tahoe, tested on Tahoe.
This isolates the platform shift that rung 4's organoid transfer will also contain.

## Current result

`docs/results/rung2_grid.csv`, `rung2_transfer_penalty.csv` [jobs 31677382-4], `rung2_l1000_context_generation.csv` [job 31678008].
Every real baseline loses 0.17-0.53 of correlation moving from Tahoe-fit to L1000-fit; cross-platform scores sit at the shuffled control (pca 0.035 vs 0.033).
Stack's own arm is null in both checkpoints, matching its rung-1 result — its failure is not a context/platform artifact.

## Why still OPEN

`cmapPy` is a hard dependency of four rung-2 files and is declared in no dependency file, so the rung is not reproducible from a fresh `uv sync`.

## Child tasks

None; this rung was built inside the 2026-08-25 ladder round.
