# Rung 1: floor, positive control, and cross-validated capacity

**Status** OPEN.
**Steps** build, fit, null, promote.
**Project rules relied on** 6 (floor + positive control), 7 (capacity), 8 (re-promotion), 1 and 3 (unchanged, but the re-run must not disturb them).
**Scope** task-specific — one rung, one re-run, one re-promotion. The rules it satisfies are project-wide; the violation is rung 1's alone.

## The problem

Two defects, both in the same rung, both requiring the same cluster job to fix.

**No controls.** `scripts/rung1_plan.py` sets `BASELINES = ("observed_delta", "knn", "pca", "nmf")`.
There is no `prior` (floor), no `planted` (positive control), and no `*_random` (noise) row anywhere in the promoted `docs/results/rung1_check1_fidelity.csv`.
The table therefore cannot separate "the baselines genuinely beat Stack" from "this harness cannot fit anything, and the baselines are closer to zero than Stack is."
Rung 2 and rung 3 both carry `prior` and `planted`; rung 1 is the outlier.

Worse, `scripts/audit_ladder.py` reports `controls_floor=True` and `controls_positive=True` for rung 1 — a regex over `score_generation_eval.py` and `de_permutation_null.py`, **neither of which produced the promoted result** (its own sidecar names `rung1_gather.py`).
Anyone reading the audit CSV would conclude the controls are present.

**Pinned capacity.** `scripts/alpine/34a_rung1_plan.sbatch` passes `--folds 5 --k 10`, fixing the component count for `pca`/`nmf`/`knn`, while rung 2 cross-validates it.
Comparing a tuned arm against a pinned one measures the pinning as much as the method.
The library already supports CV selection — this is the unthreaded half of `cross-check-fairness-and-capacity`.

## The change

1. Add `prior`, `planted`, and one `*_random` per learned source to `rung1_plan.py`'s plan, built the way rung 2 builds them: `planted` threaded into the **fit target**, not substituted at scoring time.
   Rung 2's positive control scored ≈ -0.005 for exactly that reason before `4c23f60` — it was never shown the signal it was asked to recover.
   Build `planted` per drug, with an independent random direction per drug: a single global direction makes every row correlate at ±1 and no fit can clear the null.
2. Drop `--k 10` from `34a_rung1_plan.sbatch` and let capacity be cross-validated, identically for every learned source.
3. Record the **selected** capacity per source as a column in `rung1_check1_fidelity.csv`, so "tuned or pinned?" is answerable from the artifact instead of from a driver's command line.
4. Re-run `34a → 34 → 34b`, re-promote `rung1_check1_fidelity.csv` with a fresh sidecar, and put a correction banner on every document reporting the old numbers — `docs/l1000_imputation_fidelity.md`'s banner is the template. Never a silent edit.
5. Fix `audit_ladder.py`'s rung-1 provenance row to read the scripts that actually produce the result, or drop the columns. A regex that points at the wrong files is worse than no column.

## Tests this task must pass

Task-specific, in `tests/test_rung1_plan.py`:

- The plan contains `prior`, `planted`, and one `*_random` per learned source. Pure plan-builder assertion, no cluster.
- On synthetic data with a planted signal, `planted` scores above the floor and `prior` does not clear its null — the known-answer pair, calling the real builders rather than re-deriving them.
- `planted` is built per drug: two drugs' planted directions are not collinear.
- The selected-capacity column is present, and is not a constant 10 across sources on data where the CV-optimal k differs.

Project rules, from `tests/test_project_rules.py`:

- `test_rule_06_comparison_tables_carry_a_floor_and_a_positive_control[rung1_check1_fidelity.csv]` — `xfail(strict=True)` today. This task flips it and **deletes the marker in the same change**.
- `test_rule_07_rung_drivers_do_not_pin_capacity_to_a_literal[scripts/alpine/34a_rung1_plan.sbatch]` — same.
- `-m "step_build or step_fit or step_null or step_promote"`, plus the full file.

## Promotion to a project rule

The four task-specific tests are about rung 1's plan builder and stay here.
The third one — a positive control must be built per condition, not once globally — is a candidate for promotion: the same defect has now occurred at rung 2 and would occur anywhere a planted control is added.
Promote it to a project rule if a third rung needs it, and say so in this section when that happens.

## Done when

`rung1_check1_fidelity.csv` carries floor, positive control and noise rows and a selected-capacity column; both `xfail`s are gone; the result is re-promoted with a fresh sidecar; every document reporting the old numbers carries a correction banner; `docs/PROJECT_STATE.md`'s rung-1 entry reports the new numbers with its spec/code/output links.
