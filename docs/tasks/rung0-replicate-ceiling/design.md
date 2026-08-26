# Rung 0 — replicate ceiling

**Status** OPEN — promoted, with one comparability defect outstanding.
**Steps** build, score, null, promote.
**Parent** [`docs/PROJECT_SPEC.md`](../../PROJECT_SPEC.md); **state** [`docs/PROJECT_STATE.md`](../../PROJECT_STATE.md) §1, rung 0.
**Design source** [`docs/transfer_ladder_protocol.md`](../../transfer_ladder_protocol.md), rung 0.

## What this rung establishes

Whether the target itself is reproducible: the correlation between two half-samples of Tahoe replicates, on the same 14,121-gene panel rung 1 is scored on.
Every higher rung's score is read as a fraction of this ceiling; a rung cannot beat the reproducibility of its own target.

## Current result

`docs/results/rung0_delta_reproducibility.csv` [job 31676846]: split-half median 0.109, Spearman-Brown full-data 0.197, both clearing their diff-drug and same-drug nulls (p=0.0005).
Earlier figures (0.299/0.461) were measured on an unpinned top-2000-HVG panel and are superseded.

## Why still OPEN

Rung 0 aggregates by median while rung 1 aggregates by mean, so rung 1's fraction-of-ceiling is blocked rather than computed wrong (project rule 2).
`splithalf_mean_r` is already in the CSV; closing means picking the shared aggregation, declaring it (see `project-rule-enforcement`, change 4), and republishing the headline.

## Child tasks

None.
