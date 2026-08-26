# Transfer ladder — project state

**As of:** 2026-08-26, commit `f251b21`, branch `worktree-modular-harness-core`.

## What this document is

This is the single current-state document: what is true right now. Every claim is traceable to a promoted artifact, a file:line, and/or a job id.
Every new entry opens with three links — **Spec** (the task's `design.md` under `docs/tasks/<task-slug>/`), **Code** (the scripts and commits that implement it), **Outputs** (the promoted CSV, its provenance sidecar, the job id, the figure) — so a number, the intent behind it, and the artifact it came from are one hop apart in either direction.
Where one of the three is genuinely absent, the entry says so rather than omitting the line.

It does **not** replace `docs/transfer_ladder_protocol.md` (the rungs, the six invariants, the per-rung baseline/model/control lists — that is the standing design) or `docs/decisions/2026-08-25-ladder-round.md` (D1-D6, why each was decided that way) — both stay authoritative for *design intent*; this document is authoritative for *current implementation status* and is expected to go stale again as soon as more work lands.
Update it constantly. Don't let the details drift — that drift, repeated across a dozen documents instead of one, is the problem `docs/PROJECT_SPEC.md` (project rules + spec tree) and `docs/PROCESS.md` (how work actually gets done, session to session) exist to fix.
Read those two first if you haven't.

**Read this before trusting any number in a deck or a paper draft.**
A number not listed here as promoted is not evidence yet.

---

## 1. Per-rung status

### Rung 0 — replicate ceiling

**Spec** — [`transfer_ladder_protocol.md`](transfer_ladder_protocol.md), rung 0 (standing protocol; no task folder).
**Code** — [`scripts/delta_reproducibility.py`](../scripts/delta_reproducibility.py), [`scripts/alpine/delta_reproducibility.sbatch`](../scripts/alpine/delta_reproducibility.sbatch); commits `c1fa798` (panel pin + p-value), `b7b1d72` (stratified null), `6a7a7cf`.
**Outputs** — [`rung0_delta_reproducibility.csv`](results/rung0_delta_reproducibility.csv) + [sidecar](results/rung0_delta_reproducibility.provenance.json), job 31676846; figure [`rung0_ceiling.png`](figures/rung0_ceiling.png).

**Promoted, correctly panelled.**
`docs/results/rung0_delta_reproducibility.csv` [job 31676846].
Split-half median **0.109**, Spearman-Brown full-data **0.197**, both clearing their diff-drug and same-drug nulls (p=0.0005).

The ceiling is computed on rung 1's declared 14,121-gene panel: `scripts/alpine/delta_reproducibility.sbatch` passes `--panel-file` (`c1fa798`).
The earlier 0.299/0.461 figures were measured on an unpinned top-2000-HVG panel and are superseded, not merely re-measured.
`p_vs_same_drug` uses the bootstrapped aggregate statistic (same commit); it previously used the aggregate-vs-per-item form that `p_vs_null`, 20 lines away in the same file, had already been corrected to avoid.

**Open:**
- Rung 0 aggregates by **median**; rung 1 aggregates by **mean** (ladder protocol invariant 3; `PROJECT_SPEC.md` project rule 2).
  This is why rung 1 does not get a "fraction of ceiling" in the ladder summary — it is marked **blocked** rather than computed wrong.
  Rung 0's `splithalf_mean_r` is already in the CSV; switching the headline to it (or switching rung 1 to a median) is the fix, whichever the team picks.
- Rung 0 scores Pearson; rung 2 scores Spearman (ladder protocol invariant 2; `PROJECT_SPEC.md` project rule 2).
  Same class of problem, blocks a rung-2-vs-rung-0 comparison specifically.

### Rung 1 — held-out Tahoe line, delta fidelity

**Spec** — [`tasks/stack-drug-alignment-and-check1/design.md`](tasks/stack-drug-alignment-and-check1/design.md) (ORPHANED — its registry driver is not what runs) and [`tasks/stack-faithful-generation-and-de-metrics/design.md`](tasks/stack-faithful-generation-and-de-metrics/design.md) for the DE metrics; [`transfer_ladder_protocol.md`](transfer_ladder_protocol.md), rung 1.
**Code** — [`scripts/rung1_plan.py`](../scripts/rung1_plan.py), [`rung1_build_one.py`](../scripts/rung1_build_one.py), [`rung1_gather.py`](../scripts/rung1_gather.py); sbatch [`34a`](../scripts/alpine/34a_rung1_plan.sbatch)/[`34`](../scripts/alpine/34_rung1_array.sbatch)/[`34b`](../scripts/alpine/34b_rung1_gather.sbatch).
The registry path the spec designed, [`check1_registry_driver.py`](../scripts/check1_registry_driver.py), is not invoked by any of them.
**Outputs** — [`rung1_check1_fidelity.csv`](results/rung1_check1_fidelity.csv) + [sidecar](results/rung1_check1_fidelity.provenance.json), job 31675161; DE table [`de_permutation_null_both_checkpoints.csv`](results/de_permutation_null_both_checkpoints.csv) + [sidecar](results/de_permutation_null_both_checkpoints.provenance.json); figure [`rung1_de_null.png`](figures/rung1_de_null.png).

**Promoted but incomplete against its own protocol row.**
`docs/results/rung1_check1_fidelity.csv` [job 31675161].
`knn`/`pca`/`nmf` (~0.28–0.32) clearly beat `stack_cytokine`/`stack_drug_aligned` (~0.018/0.040) — directionally real, matched by the DE-fidelity table's `within_drug` p-values (baselines p≈0.68–0.80, Stack/reference p≈0.005).

**Open:**
- `rung1_plan.py` never builds `prior` (floor), `planted` (positive control), or `*_random` (noise controls) — `BASELINES = ("observed_delta", "knn", "pca", "nmf")`, no floor/positive- control row exists anywhere in the promoted CSV. Without them there is no way to confirm the rung-1 harness itself is working, as distinct from the baselines genuinely beating Stack.
- `scripts/audit_ladder.py`'s rung-1 provenance check reads `score_generation_eval.py` and `de_permutation_null.py` — **neither produced the promoted rung-1 result** (its own sidecar names `rung1_gather.py`).
  The audit's `controls_floor=True`/`controls_positive=True` for rung 1 is not evidence of anything; it is a regex match against files the rung does not run.
- Owned by [`tasks/rung1-controls-and-capacity`](tasks/rung1-controls-and-capacity/design.md), which closes both of the two bullets above in one re-run.
- Baseline capacity (`--k`) is fixed at 10 for rung 1's pca/nmf/knn while rung 2 CV-selects it — an unfair comparison the 2026-08-21 capacity-fairness spec was written to remove.

### Rung 2 — cross-platform (map fit on L1000, tested on Tahoe)

**Spec** — [`transfer_ladder_protocol.md`](transfer_ladder_protocol.md), rung 2, plus decision D3 in [`decisions/2026-08-25-ladder-round.md`](decisions/2026-08-25-ladder-round.md).
No task folder; this rung was built inside the ladder round.
**Code** — [`scripts/rung2_plan.py`](../scripts/rung2_plan.py), [`rung2_score_one.py`](../scripts/rung2_score_one.py), [`rung2_gather.py`](../scripts/rung2_gather.py); sbatch [`30a`](../scripts/alpine/30a_rung2_plan.sbatch)/[`30`](../scripts/alpine/30_rung2_array.sbatch)/[`30b`](../scripts/alpine/30b_rung2_gather.sbatch); helpers `fmharness.deltas.shuffled_target_base` and `fmharness.statistics.bootstrap_aggregate_pvalue`; commit `4c23f60`.
**Outputs** — [`rung2_grid.csv`](results/rung2_grid.csv) + [sidecar](results/rung2_grid.provenance.json), [`rung2_transfer_penalty.csv`](results/rung2_transfer_penalty.csv) + [sidecar](results/rung2_transfer_penalty.provenance.json), jobs 31677382/31677383/31677384; Stack's arm [`rung2_l1000_context_generation.csv`](results/rung2_l1000_context_generation.csv) + [sidecar](results/rung2_l1000_context_generation.provenance.json), job 31678008; figure [`rung2_transfer.png`](figures/rung2_transfer.png).

**Promoted, after four control/correctness fixes and a re-run.**
`docs/results/rung2_grid.csv`, `rung2_transfer_penalty.csv` [jobs 31677382/31677383/31677384, commit `4c23f60`].
The rung previously produced nothing usable: its cluster array crashed and three of its controls were broken.

The four defects, each verified against a synthetic plan dir (including the exact singleton-fold shape that crashed the cluster) before the real run:
1. **The crash**: the shuffled control's line-relabeling matched the held-out line with probability `|fold|/50`, not 1 — essentially never at 5-fold.
   Replaced with `fmharness.deltas.shuffled_target_base`, a tested derangement helper.
2. **Positive control never fitted**: `planted` substituted its truth only at *scoring* time; every arm still trained on the real delta, so it had no way to recover a signal it was never shown (scored ~-0.005).
   Now built per-drug (independent random direction and drug-mean vector per drug — a single global direction had made every row correlate at exactly ±1, making "clear the null" impossible even for a perfect fit) and threaded into the actual fit target.
   Recovers cleanly now (~0.93-0.95 across arms).
3. **Negative control identical to the model**: `bulk_target`'s `shuffled` cell had no branch and fell through to the same call as `pca`.
4. **`bulk_target` leakage**: fit on the full Tahoe set, predicted the same lines' GDSC2 bulk profiles — every target's own delta was in its own fit.
   Unified with `in_platform` through one 5-fold helper (`_folded_predictions`) sharing the one fold split required by ladder protocol invariant 5 and `PROJECT_SPEC.md` project rule 1.

Also fixed: the mismatched-pair null was unstratified (same class of bug fixed in rung 0's `b7b1d72`, reintroduced here) — now diff-drug-stratified, and its p-value goes through the same `fmharness.statistics.bootstrap_aggregate_pvalue` helper (see §2).
And `rung2_plan.py`'s L1000 training set previously included Tahoe's own 7 overlapping lines, so `cross_platform` saw up to 14% of its evaluation lines' own responses during fit — excluded now.

**First valid numbers**: every real baseline loses 0.17–0.53 of correlation moving from Tahoe-fit to L1000-fit maps; cross-platform scores sit close to the shuffled (wrong-line) control (e.g. pca 0.035 vs shuffled 0.033) — cross-platform transfer for the fitted baselines is barely distinguishable from a scrambled baseline.

**Stack's own rung-2 arm** (decision D3: rebuild its generation context from L1000, query baseline held at Tahoe) is built and scored: `docs/results/rung2_l1000_context_generation.csv` [job 31678008].
Scoring it needs a `pert_id`→PubChem-CID map, because the generation output is named by L1000's Broad `pert_id` (`BRD-K...`) while the scorer works in Tahoe's PubChem-CID convention; that map is built from L1000's own `pert_info` table, and no generated file is renamed (`build_generated_deltas` matches by filename stem against a mapping dict).
Both checkpoints null (r=-0.001 cytokine, r=0.011 drug-aligned), matching rung 1's Tahoe-context result within noise — Stack's failure is not a context/platform artifact.

**Open:** `cmapPy` is a hard dependency of rung 2 (four files import it) and is declared in no dependency file — not reproducible from a fresh `uv sync`.

### Rung 3 — GDSC2 viability

**Spec** — [`tasks/check2-leakage-aware-drug-aligned/design.md`](tasks/check2-leakage-aware-drug-aligned/design.md) (ORPHANED — registry driver bypassed) and [`tasks/cross-check-fairness-and-capacity/design.md`](tasks/cross-check-fairness-and-capacity/design.md); [`transfer_ladder_protocol.md`](transfer_ladder_protocol.md), rung 3.
**Code** — [`scripts/check2_plan.py`](../scripts/check2_plan.py), [`check2_score_one.py`](../scripts/check2_score_one.py), [`check2_gather.py`](../scripts/check2_gather.py), [`report_variants.py`](../scripts/report_variants.py); sbatch [`18a`](../scripts/alpine/18a_check2_plan.sbatch)/[`18`](../scripts/alpine/18_check2_array.sbatch)/[`18b`](../scripts/alpine/18b_check2_gather.sbatch); figure fix in [`plot_ladder_results.py`](../scripts/plot_ladder_results.py), commits `8a5badb`/`f251b21`.
**Outputs** — [`rung3_check2_grid.csv`](results/rung3_check2_grid.csv) + [sidecar](results/rung3_check2_grid.provenance.json), [`rung3_declared_variants.csv`](results/rung3_declared_variants.csv) + [sidecar](results/rung3_declared_variants.provenance.json), [`rung3_label_ceiling.csv`](results/rung3_label_ceiling.csv) + [sidecar](results/rung3_label_ceiling.provenance.json), job 31665927; figure [`rung3_check2.png`](figures/rung3_check2.png).

**Solid.**
`docs/results/rung3_check2_grid.csv`, `rung3_declared_variants.csv` [job 31665927].
`base` embedding under L2 penalty is the only representation clearing Bonferroni across 24 declared variants (interaction 0.137, p=0.001, z_random=2.66) — ~30% of the 0.457 screen-agreement ceiling.
`report_variants.py`'s Bonferroni check computes this correctly.
The figure (`plot_ladder_results.py`) reports the same number, restricted to the significant, non-control row; it previously took an unfiltered max over all 24+ control rows (`8a5badb`/`f251b21`).

**Open:**
- `perdrug` and `global` are computed on non-residualized predictions while `interaction` residualizes out the per-drug mean (`evaluation.py`'s `score_predictions`).
  Proof it matters: `prior` — one constant feature, zero line information by construction — scores `perdrug=-0.285` while its `interaction` is exactly 0.000.
  A reader comparing the `perdrug` column across rows is reading fold-intercept structure, not per-drug ranking signal.
- `report_variants.py` cannot be run the obvious way — `ModuleNotFoundError` on its root-level shim import; needs `PYTHONPATH=.` or `python -m`.

### Rung 4 — organoid viability (embargoed, frozen holdout)

**Spec** — [`tasks/rung4-organoid-viability/design.md`](tasks/rung4-organoid-viability/design.md), the rung branch of `PROJECT_SPEC.md`'s spec tree, alongside [`transfer_ladder_protocol.md`](transfer_ladder_protocol.md)'s rung-4 row plus decisions D1/D2 in [`decisions/2026-08-25-ladder-round.md`](decisions/2026-08-25-ladder-round.md).
The rung's spec is now [`tasks/rung4-organoid-viability/design.md`](tasks/rung4-organoid-viability/design.md); it must grow the rung's control list and metric declaration before the next submission.
**Code** — [`scripts/score_viability_adapters.py`](../scripts/score_viability_adapters.py), [`scripts/alpine/12_sarcoma_organoids_2024_score.sbatch`](../scripts/alpine/12_sarcoma_organoids_2024_score.sbatch), [`src/fmharness/data/loaders/sarcoma_organoids_2024.py`](../src/fmharness/data/loaders/sarcoma_organoids_2024.py), [`scripts/build/build_drug_xref.py`](../scripts/build/build_drug_xref.py), [`scripts/alpine/migrate_soragni_rename.sh`](../scripts/alpine/migrate_soragni_rename.sh); commits `24c6240` (D1 + fold map), `82e0d6b` (sbatch `set -u` crash), `ad34b29` (raw-data path migration).
**Outputs** — **none promoted.**
Feasibility only: [`rung4_feasibility.csv`](results/rung4_feasibility.csv) + [sidecar](results/rung4_feasibility.provenance.json) and [`rung4_table_granularity.csv`](results/rung4_table_granularity.csv).
The scored result `docs/results/rung4_viability.csv` does not exist yet; [`rung4_viability.png`](figures/rung4_viability.png) is wired to it and will render the moment it does.

**Blocked — this is the live item.**
Decision D2's unfreeze condition ("once rung 3 is promoted") is met, but D2 also names two audit gaps to close first: `prov_params`, `prov_panel`.
Five defects stand between the rung and a first result; four are fixed, the fifth is open:

1. **D1 not implemented**: the only rung-4 script effectively used L1000's drug coverage (the option D1 rejected), not GDSC2's (the option D1 chose), because nothing restricted the organoid target to GDSC2's screened compounds before scoring.
   **Fixed** — `design` is now restricted to `set(dg["drug"])` (GDSC2's own AUC design, already loaded for the training join) immediately after it's built.
   Commit `24c6240`.
2. **Hand-written fold splits** (ladder protocol invariant 5, `PROJECT_SPEC.md` project rule 1): two `{ln: i % n_folds ...}` fold maps instead of the shared `fold_assignment` helper.
   **Fixed**, same commit — behavior is unchanged at the `--folds 5` this script actually runs with; it only differed at the unused `--folds 1` edge case, where the shared helper's LOO degeneracy is correct.
3. **sbatch crash**: `scripts/alpine/12_sarcoma_organoids_2024_score.sbatch`'s `Resolved:` log line referenced `$GCTX`/`$GENDIR`/etc. *before* their default-value assignments, so any submission without explicit `--export` for every one of them hit `set -u` and died in the same second it started (job 31679368).
   **Fixed**, commit `82e0d6b` — moved the echo after the defaults.
4. **Stale raw-data path**: the 2026-08-25 rename migration (`e886685`) covered repo-root generated artifacts only; `data/raw/soragni/` was never renamed to `data/raw/sarcoma_organoids_2024/`, which the loader hardcodes with no override, so `load_sarcoma_organoids_2024()` failed closed with "raw manifest missing" (job 31679380).
   **Fixed on Alpine and locally** — extended `scripts/alpine/migrate_soragni_rename.sh` with the same `mv -vn` (never-overwrite, guarded) pattern, ran it both places.
   Commit `ad34b29`.
5. **Stale drug crosswalk — OPEN.**
   `data/static/drug_xref.parquet` still tags every organoid drug `source="soragni"` from before the rename; the loader filters for `source=="sarcoma_organoids_2024"` (`sarcoma_organoids_2024.py:213`).
   Zero rows match, so `build_sample_design(..., drug_key="pubchem_cid")` returns an empty design — "0 of 0 drugs" (job 31679480, after fixes 1–4 landed).
   The code that BUILDS the crosswalk (`scripts/build/build_drug_xref.py:242`) already writes the correct new label; only the *committed parquet* is stale — a data-artifact staleness problem, not a code bug, and the same *class* of problem this whole document exists to stop recurring.
   The rebuild has not been run: `data/static/drug_xref.parquet` and `manifest.json` still carry the old labels on disk.
   It takes several minutes — `--refresh` hits PubChem for ~650 compounds at a rate-limited 0.25s/call.
   **Next action: rerun `PYTHONPATH=src uv run python scripts/build/build_drug_xref.py --refresh`, verify the `source` column reads `sarcoma_organoids_2024` for the organoid rows, commit (it's public reference data per `release_manifest.yaml`), push, pull to Alpine, resubmit `12_sarcoma_organoids_2024_score.sbatch`.**

**Still open after that unblocks it:** the two named audit gaps, `prov_params`/`prov_panel` — the latter is real (no `common_gene_panel`/`assert_common_genes` call anywhere in `score_viability_adapters.py`), not a blunt regex check, and is not attempted yet: it touches embargoed-data code and needs a real run to verify.
`docs/figures/sarcoma_organoids_2024_pathb_summary.png` is deleted — it came from a retired pipeline (adapters "szalai"/"xgboost" that exist nowhere in current code).
Its replacement, `rung4_viability.png`, is wired up and will render once rung 4 produces `docs/results/rung4_viability.csv`; that filename does not exist yet, and the loader script may need a matching `--out-csv` name or a small `plot_ladder_results.py` load-path adjustment once rung 4's output schema is known.

---

## 2. The p-value bug family — one cause, fixed centrally

Four scripts had computed p by comparing a reported AGGREGATE (a mean or median over many pairs) against the spread of INDIVIDUAL null draws, instead of against the bootstrapped sampling distribution of that same aggregate at the observed pair count.
An aggregate's standard error is roughly √n tighter than a single draw's, so this inflated every affected p by one to two orders of magnitude:

- `delta_reproducibility.py`'s `p_vs_same_drug` (the `p_vs_null` beside it, 20 lines away, was already correct — commit `6a7a7cf` fixed one and not the other).
- `l1000_imputation_fidelity.py` — flips the headline: landmark genes go from p=0.2438 ("not established") to p=0.0005 (real).
- `l1000_tahoe_agreement_diagnosis.py`'s transform sweep — all seven transforms now clear their null (were p=0.13–0.28, "none clears it").
- `rung2_score_one.py` — see rung 2 above.

Fixed with one shared, tested implementation, `fmharness.statistics.bootstrap_aggregate_pvalue`, so this class of bug can only be reintroduced by NOT using the helper, which is now the visible, greppable anomaly rather than the invisible default.
**`docs/l1000_imputation_fidelity.md` and `docs/transfer_ladder_protocol.md` were corrected in place with explicit banners** rather than silently rewritten — a reader who saved the old conclusion needs to see it was wrong and why.

**Closed: the systematic audit for remaining occurrences.** `tests/test_project_rules.py::test_rule_05_edge_every_manual_pvalue_site_is_allowlisted` scans the whole repo for the shape and fails on any site not allowlisted with a verified reason; the six current sites (five scripts plus `evaluation.py`'s permutation form) are each listed with why they are correct — their nulls hold permutation replicates of the same aggregate, so the inflation cannot occur.

---

## 3. Provenance / release-gate gaps — all open

- **No promoted result carries a `LeakageProfile`.**
  `filter_leakage` exists and is correctly implemented, but is only ever called from `check1_registry_driver.py`/ `check2_registry_driver.py` — neither is invoked by any current sbatch. Every promoted rung-1 and rung-3 number comes from a parallel path (`rung1_plan/build_one/gather.py`, `check2_plan/score_one/gather.py`) that bypasses the registry abstraction entirely, including this guarantee.
  Concretely: nobody has checked whether the test cell lines at rungs 1/3 were in Stack's pretraining corpus.
  See `docs/PROJECT_SPEC.md`'s spec tree (branches marked ORPHANED) for how this happened.
- **`check_release.py` scans column NAMES, not cell VALUES.**
  `SAMPLE_COLUMNS` (`scripts/check_release.py:46-49`) triggers row-level scanning only when a table has a column literally named `patient`/`line`/`cell_line`/etc. `docs/results/rung4_table_granularity.csv` has the schema `table, rows, columns, dose_or_replicate_columns, ...`
  — none of those names trip the check — but one row's `columns` cell is a semicolon-joined **string value** that contains `SARC0128_Tumor;SARC0129_Tumor;SARC0120_Organoids` as substring content, since that row describes `normalized_gene_counts.parquet`'s own column list.
  The file is committed with those identifiers present and is still in the repo.
  Owned by [`tasks/embargo-gate-cell-values`](tasks/embargo-gate-cell-values/design.md), which fixes the gate and remediates this file.
- **The gate is not enforced anywhere** — no pre-commit hook installed, no CI step, no test invokes it.
  `.github/workflows/ci.yml` runs ruff → ruff format → pyright → pytest, in that order, and a lint failure blocks the run before pytest ever executes — the last count on this branch is 251 ruff errors, 69 files needing reformat, and 21 pyright errors, so CI has never reported a pass/fail on the test suite for this branch.
- **`promote_result.py` records `HEAD` at promotion time** (owned by [`tasks/project-rule-enforcement`](tasks/project-rule-enforcement/design.md), change 1), not the commit the result was produced at, and does not check the working tree was clean when promoting.
  Two promoted sidecars carry input/log paths under another machine's `/private/tmp/` scratchpad and `/Users/lucas/...` home — the sha256 still verifies, but the recorded location is unrecoverable from here.
- **`audit_ladder.py`'s `controls_*` columns are a regex over script source text**, not evidence a control ran.
  Confirmed correct in spirit (the file's own docstring says as much) but easy to over-read from the CSV alone — see the rung-1 example above, where it is actively misleading because it points at the wrong scripts entirely.

---

## 4. Confirmed correct — do not re-litigate

Each item below was checked directly and survived an independent attempt to refute it; several were confirmed correct rather than merely left unchallenged.
Treat them as settled unless new evidence appears:

- `fmharness.deltas.fold_assignment` genuinely is one shared, sorted, deterministic partition — `loo_baseline_source` holds out the whole fold as a group (not per-line), and rung 1/rung 3 agree on the real data despite partitioning on different label spaces, verified by direct reproduction (not by construction).
- The (patient, drug) support restriction (`restrict_common_support`) is real and applied before scoring at both rung 1 and rung 3 — the exact bug class (`d9f94ec`) it was built to prevent does not currently recur there.
- `check_release.py` verifies embargo **per value** against the public cell-line registry, not per declaration, as claimed.
- Decision D6 (the `additive`→`observed_delta` rename with a back-compat alias) is fully and correctly implemented, including its stated "reverse by" path.
- Decision D5's staleness map (`SUPERSEDED_BEFORE` in `audit_ladder.py`) is real and precise — flags exactly the one result it should, not more, not fewer.
- The 2026-08-21 capacity-fairness and common-support specs are fully implemented at the six sites they were meant to reach (rung 3's fixed-readout and penalized-grid paths, both).
- `rung1_gather.py`/`check2_gather.py` both genuinely refuse to score an incomplete source set, for the correct reason (missing a source changes every OTHER source's number too).

---

## 5. Project-rule violations currently exempted

`tests/test_project_rules.py`'s `KNOWN_GAPS` is the machine-readable version of this list; each entry is a strict `xfail`, so closing one turns the test into an unexpected pass and forces the entry out.
Three are classified — two owned by a task, one exempt with a stated reason.
Four surfaced when the tests started discovering their cases instead of running against hardcoded lists, and each needs one of three answers — fix it, exempt it with a stated reason, or record that it is retired.
None of them may be guessed.
The discovery run also cleared one suspect: `check2_grid_5fold_corrected.csv` carries `prior` and `planted` and passes rule 6 — its provisional exemption turned into a strict-xfail *unexpected pass* and was removed, which is the mechanism doing its job.

| Case | Rule | Status |
|---|---|---|
| `rung1_check1_fidelity.csv` | 6 | Owned by [`tasks/rung1-controls-and-capacity`](tasks/rung1-controls-and-capacity/design.md) |
| `scripts/alpine/34a_rung1_plan.sbatch` | 7 | Owned by [`tasks/rung1-controls-and-capacity`](tasks/rung1-controls-and-capacity/design.md) |
| `de_permutation_null_both_checkpoints.csv` | 6, 3 | **Unclassified** — a permutation-null table may not be a method comparison; if not, the discovery predicate needs narrowing rather than the table needing controls. One classification decision resolves both entries |
| `rung3_declared_variants.csv` | 3 | Exempt with reason: `report_variants.py` re-reports `rung3_check2_grid.csv`, whose producing family carries the guards upstream of its input. Becomes a real gap only if it ever rescores raw deltas |
| `scripts/alpine/02_merge_score.sbatch` | 7 | **Unclassified** — pins `--k 10`; predates the ladder, may be retired |
| `scripts/alpine/05_stack_score.sbatch` | 7 | **Unclassified** — pins `--k 10`; predates the ladder, may be retired |
| `scripts/alpine/07_stack_emb_score.sbatch` | 7 | **Unclassified** — pins `--k 10`; predates the ladder, may be retired |

An unclassified exemption is worse than a failing test if it sits here: it looks handled.
Classifying these four is the first work item of [`tasks/project-rule-enforcement`](tasks/project-rule-enforcement/design.md).

---

## 6. Where things live

- **Results**: `docs/results/*.csv` + matching `.provenance.json` sidecar (job id, git sha, input hashes).
  No sidecar, no evidence — this project's own standing rule.
- **Figures**: `docs/figures/*.png`, generated by `scripts/plot_ladder_results.py`, current as of `f251b21` — see the `8a5badb`/`f251b21` commit messages for what was wrong with the previous set (two panels were blank placeholders never regenerated after promotion; the ladder summary showed one rung; one figure was from a retired pipeline).
- **Audit**: `docs/results/ladder_audit.csv` (per-rung control/provenance checks — read §3's caveat about what these columns do and don't prove), `docs/results/promoted_provenance.csv` (staleness).
- **Design**: per task in `docs/tasks/<task-slug>/design.md` (+ `plan.md`), a branch of `docs/PROJECT_SPEC.md`'s spec tree; the cross-task standing protocol is `docs/transfer_ladder_protocol.md`.
  **Decisions**: `docs/decisions/YYYY-MM-DD-<slug>.md` for reversals and cross-task calls (currently `2026-08-25-ladder-round.md`, `2026-06-16-revert-coderdata-loaders.md`); task-local ones live in that task's folder.
  **This document**: current implementation state, update it as things change rather than writing a new dated handoff, and keep each entry's spec/code/output links current with it.
