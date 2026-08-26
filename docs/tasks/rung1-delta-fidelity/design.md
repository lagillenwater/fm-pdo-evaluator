# Rung 1 — held-out Tahoe line, delta fidelity

**Status** OPEN — promoted but incomplete against its own protocol row.
**Steps** build, fit, score, null, promote.
**Parent** [`docs/PROJECT_SPEC.md`](../../PROJECT_SPEC.md); **state** [`docs/PROJECT_STATE.md`](../../PROJECT_STATE.md) §1, rung 1.
**Design source** [`docs/transfer_ladder_protocol.md`](../../transfer_ladder_protocol.md), rung 1 (the old Check 1 / Check 1b).

## What this rung establishes

Whether a method can predict a held-out cell line's expression response (delta) within the same platform — the easiest transfer, one distribution shift (an unseen line), nothing else.

## Current result

`docs/results/rung1_check1_fidelity.csv` [job 31675161]: `knn`/`pca`/`nmf` (~0.28-0.32) clearly beat `stack_cytokine`/`stack_drug_aligned` (~0.018/0.040), matched by the DE-fidelity table's `within_drug` p-values.

## Why still OPEN

The table carries no floor, no positive control, and pins baseline capacity at `--k 10` while rung 2 cross-validates it — so it cannot yet separate "the baselines genuinely beat Stack" from "this harness cannot fit anything."
Both defects close in one re-run, owned by the child task below.

## Child tasks

- [`rung1-controls-and-capacity`](../rung1-controls-and-capacity/design.md) — OPEN; the re-run above.
- [`stack-drug-alignment-and-check1`](../stack-drug-alignment-and-check1/design.md) — ORPHANED; its registry driver is correct and bypassed in production.
- [`stack-faithful-generation-and-de-metrics`](../stack-faithful-generation-and-de-metrics/design.md) — OPEN (mostly done); DE metrics not yet merged into the primary delta table.
