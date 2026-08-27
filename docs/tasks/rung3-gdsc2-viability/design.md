# Rung 3 — GDSC2 viability (cell line)

**Status** OPEN — promoted and solid; two reporting defects outstanding.
**Steps** build, restrict, split, fit, score, null, promote.
**Parent** [`docs/PROJECT_SPEC.md`](../../PROJECT_SPEC.md); **state** [`docs/PROJECT_STATE.md`](../../PROJECT_STATE.md).
**Design source** [`docs/transfer_ladder_protocol.md`](../../transfer_ladder_protocol.md), rung 3 (the old Check 2).

## What this rung establishes

Whether a representation predicts drug response (viability), not just expression — still on cell lines, so rung 4's substrate shift is the only shift left after this one.

## Current result

`docs/results/rung3_check2_grid.csv`, `rung3_declared_variants.csv` [job 31665927]: `base` embedding under L2 penalty is the only representation clearing Bonferroni across 24 declared variants (interaction 0.137, p=0.001) — about 30% of the 0.457 screen-agreement ceiling.

## Why still OPEN

`perdrug`/`global` are computed on non-residualized predictions while `interaction` residualizes, so the `perdrug` column reads fold-intercept structure (the `prior` row proves it: perdrug -0.285, interaction 0.000).
`report_variants.py` needs `PYTHONPATH=.` or `python -m` to run at all.

## Child tasks

- [`check2-leakage-aware-drug-aligned`](../check2-leakage-aware-drug-aligned/design.md) — ORPHANED; registry driver correct, bypassed by the `check2_plan/score_one/gather` path.
