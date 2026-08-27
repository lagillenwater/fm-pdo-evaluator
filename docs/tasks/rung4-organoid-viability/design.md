# Rung 4 — organoid viability (embargoed, frozen holdout)

**Status** OPEN — blocked; the live experimental item. No result promoted.
**Steps** load, build, restrict, fit, score, promote, release.
**Parent** [`docs/PROJECT_SPEC.md`](../../PROJECT_SPEC.md); **state** [`docs/PROJECT_STATE.md`](../../PROJECT_STATE.md).
**Design source** [`docs/transfer_ladder_protocol.md`](../../transfer_ladder_protocol.md), rung 4 (the old Path B), plus decisions D1/D2 in [`docs/decisions/2026-08-25-ladder-round.md`](../../decisions/2026-08-25-ladder-round.md).

## What this rung establishes

The number the project exists to produce: whether cell-line-trained prediction transfers to patient-derived organoids — the frozen sarcoma organoid holdout, scored on GDSC2's drug axis per D1.
Embargo applies throughout (project rule 9); the cohort is a frozen holdout under D2's unfreeze conditions.

## Current state

Blocked on one remaining defect of five (the first four are fixed; commits `24c6240`, `82e0d6b`, `ad34b29`): `data/static/drug_xref.parquet` still carries the pre-rename `source="soragni"` labels, so the loader's `source=="sarcoma_organoids_2024"` filter matches zero rows and the design comes back empty.
Next action: rebuild the crosswalk with `--refresh`, verify the `source` column, commit, push, pull to Alpine, resubmit `12_sarcoma_organoids_2024_score.sbatch`.
After that: D2's two audit gaps (`prov_params`, `prov_panel` — the latter real: no panel guard in `score_viability_adapters.py`).

## Before the next submission

This spec must grow the rung's control list (floor and positive control per project rule 6) and its metric declaration, so the first promoted organoid number lands with the same apparatus every other rung carries.

## Child tasks

None yet.
