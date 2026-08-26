# Project spec — fm-pdo-evaluator

It answers three questions : **what must every piece of analysis in this repo respect, no matter which task produced it; what document is currently authoritative for a given area; and what happened to the documents that used to be authoritative and no longer are.**

It changes rarely — project rules get amended deliberately, the tree gets a new branch when a task starts, entries get reclassified when superseded.
It is not where you look for "what's true right now" (that's `docs/PROJECT_STATE.md`) or "what's the current experimental design" (that's `docs/transfer_ladder_protocol.md`, itself indexed below as the current active spec), and it is not "how do we actually work" (that's `docs/PROCESS.md` — lifecycle, tooling, git/collaboration mechanics; this file's own "Process for new specs" section below is the one piece of that which belongs here instead, since it's specifically about the task index this file owns).
It is where you look before starting new design work, to find out what's already been decided and where.

## Mission

Does a foundation model's cell-line drug-response prediction transfer to patient-derived organoids?
Two scores matter: an in-silico score now, a prospective score later, and the gap between them is the number the project exists to produce.
The evaluation has gone through three framings as understanding of the problem improved — **Check 1 / Check 2** (2026-08-07 onward, registry-driven, per-model correctness checks) → **Path A / Path B** (cell-line vs organoid substrate split) → **the transfer ladder** (2026-08-25 onward, rungs 0-4, one distribution shift added per rung).
These are not three different projects; they are the same question, reframed each time the previous framing turned out not to isolate *why* a prediction failed.
The mapping:

| Old name | Rung | What it tests |
|---|---|---|
| (new) | 0 | Tahoe replicate ceiling — is the target itself reproducible |
| Check 1 / Check 1b | 1 | held-out Tahoe cell line, delta fidelity |
| (new) | 2 | cross-platform: map fit on L1000, tested on Tahoe |
| Check 2 | 3 | GDSC2 viability (cell line) |
| Path B (Sarcoma organoids) | 4 | organoid viability — embargoed, frozen holdout |

A spec written before 2026-08-25 that says "Check 1" means rung 1; "Check 2" means rung 3; "Path B" or "Soragni" means rung 4.
There is no rung 0 or rung 2 equivalent in the old framing — both are new diagnostic rungs added when Check 1/Check 2 alone couldn't distinguish "the model failed" from "the ceiling was never established" or "the platform shifted."

## Structure — the transfer ladder

The project's structure is the ladder: five rungs, one distribution shift added per rung, ending at the organoid number the project exists to produce.
The spec tree below branches from this document: **one spec per rung**, and each rung's spec links its child tasks — the pieces of work that build or repair that rung.
Cross-cutting branches hold what no single rung owns: the harness library and the rule-enforcement work.

Each branch carries only its status — OPEN or CLOSED — and a blurb of what it wants to accomplish; why a branch is still open lives in its own spec, and the numbers live in [`docs/PROJECT_STATE.md`](PROJECT_STATE.md).
**A branch is CLOSED when** its outputs are promoted with provenance, `PROJECT_STATE.md` records its numbers, and the project-rule tests for its steps pass, including any `xfail` it was meant to flip; "implemented" is not closed, which is how three children below ended up implemented-but-bypassed.

- **Rung 0 — replicate ceiling** · **OPEN** · [spec](tasks/rung0-replicate-ceiling/design.md) · [state](PROJECT_STATE.md)
  Establish how reproducible the target itself is: split Tahoe's replicate pool in half, correlate the two halves' per-(line, drug) expression deltas on the same 14,121-gene panel rung 1 is scored on, and publish that ceiling so every higher rung reports a fraction of it instead of a raw correlation against an imaginary 1.0.
- **Rung 1 — held-out line, delta fidelity** · **OPEN** · [spec](tasks/rung1-delta-fidelity/design.md) · [state](PROJECT_STATE.md)
  Establish whether any method — Stack's generation (cytokine and drug-aligned checkpoints) against pca/nmf/knn baselines — can predict a held-out Tahoe cell line's expression delta within the same platform: the easiest transfer, one unseen line and nothing else, the rung a foundation model must clear before any harder shift means anything.
  - [`rung1-controls-and-capacity`](tasks/rung1-controls-and-capacity/design.md) · **OPEN**
    Add the rows rung 1's table is missing — a `prior` floor that must fail, a per-drug `planted` signal threaded into the fit target that must be recovered, noise controls — and replace the pinned `--k 10` with cross-validated capacity, in one re-run and re-promotion.
  - [`stack-drug-alignment-and-check1`](tasks/stack-drug-alignment-and-check1/design.md) · **OPEN, ORPHANED**
    Fine-tune Stack's generation head on drug perturbations (sci-Plex) instead of its cytokine-only alignment, and evaluate through the registry driver so pretraining-corpus leakage is filtered before any line is scored.
  - [`stack-faithful-generation-and-de-metrics`](tasks/stack-faithful-generation-and-de-metrics/design.md) · **OPEN (mostly done)**
    Run Stack's generation with its real scheduled in-context procedure (`--mode mdm`, not the vanilla workaround), and score fidelity by Wilcoxon differential-expression recovery — which genes actually move — alongside the Pearson-delta correlation.
- **Rung 2 — cross-platform** · **OPEN** · [spec](tasks/rung2-cross-platform/design.md) · [state](PROJECT_STATE.md)
  Measure what crossing a measurement platform costs: fit the same line-to-delta mapping on L1000 instead of Tahoe, test on Tahoe, and report the retained fraction per method against shuffled and planted controls — isolating the platform shift that the organoid transfer will also contain.
- **Rung 3 — GDSC2 viability** · **OPEN** · [spec](tasks/rung3-gdsc2-viability/design.md) · [state](PROJECT_STATE.md)
  Establish whether any representation — Stack embeddings, expression, PCA/NMF — predicts GDSC2 drug response (AUC) beyond each drug's mean, scored as drug×line interaction under one shared CV partition, still on cell lines so that only the substrate shift remains for rung 4.
  - [`check2-leakage-aware-drug-aligned`](tasks/check2-leakage-aware-drug-aligned/design.md) · **OPEN, ORPHANED**
    Score GDSC2 viability through a registry path that drug-aligns Stack's checkpoint and filters train/test contamination (lines in the pretraining corpus, shared drugs) by construction rather than by per-script care.
- **Rung 4 — organoid viability** · **OPEN, BLOCKED** · [spec](tasks/rung4-organoid-viability/design.md) · [state](PROJECT_STATE.md)
  Produce the number the project exists for: train on GDSC2 cell lines, predict the frozen Soragni sarcoma organoid screen on GDSC2's drug axis, and report the transfer gap — under embargo, on a holdout that stays frozen until D2's preconditions are met.
- **Cross-cutting — harness and rules**
  - [`modular-harness-core`](tasks/modular-harness-core/design.md) · **OPEN, ORPHANED**
    Build the shared evaluation library — Encoder/Generator protocols, model and readout registries, `filter_leakage`, one `fold_assignment` partition — so every rung scores through the same machinery and a guarantee implemented once holds everywhere.
  - [`cross-check-fairness-and-capacity`](tasks/cross-check-fairness-and-capacity/design.md) · **OPEN**
    Make every cross-method comparison fair by construction: every arm scored on the same (patient, drug) pairs and gene panel (`restrict_common_support`, `common_gene_panel`), matched-width random-feature controls, and capacity CV-selected identically instead of one arm tuned against another's hardcoded k.
  - [`project-rule-enforcement`](tasks/project-rule-enforcement/design.md) · **OPEN**
    Close the edge cases the project-rule tests cannot see by making the guarantees structural: a metric declaration the scorers read (rule 2), support/panel guards inside the scoring entry point (rules 1, 3), a selected-capacity column in every table (rule 7), and `clean_tree` plus the producing commit in every sidecar (rule 8).
  - [`embargo-gate-cell-values`](tasks/embargo-gate-cell-values/design.md) · **OPEN**
    Make `check_release.py` scan cell values of every text column against the public-line registry rather than only columns named like identifiers, and remediate the one committed table that carries organoid ids inside a schema-description cell.
  - `arm2-harness-validation` · **MISSING**
    Recover the Arm-2 harness-validation spec (commit `4aca11f`, other branches only) that four documents cite as the source of their Phase-1 blockers, or rewrite those citations to state their dependencies directly.

No branch is CLOSED yet, measured against the bar above — worth seeing plainly, since the repo has looked "mostly done" for weeks because implementation and closure were not distinguished.

### Project-level documents (not part of the tree)

| Doc | Status | Note |
|---|---|---|
| [`docs/transfer_ladder_protocol.md`](transfer_ladder_protocol.md) | **ACTIVE** | The rung designs the tree's specs are written against — the six ladder-scoped invariants, per-rung baseline/model/control lists. Its own Invariant 2 is **wrong** (rungs 0/1 are Pearson by design; only rung 2 is Spearman) and needs a correction. Its list is numbered separately from this file's project rules; cite which you mean. |
| [`docs/decisions/2026-08-25-ladder-round.md`](decisions/2026-08-25-ladder-round.md) | **ACTIVE, one entry stale** | D1-D6; decisions stay dated because a decision is an event, not a task. D1 implemented only at `24c6240`; D2's precondition still unmet. |
| [`docs/decisions/2026-06-16-revert-coderdata-loaders.md`](decisions/2026-06-16-revert-coderdata-loaders.md) | **ACTIVE** | The recovered CoderData reversal entry (project rule 10). |
| [`docs/HANDOFF-2026-08-26.md`](HANDOFF-2026-08-26.md) | **HISTORICAL** | Superseded by `PROJECT_STATE.md`; do not update. |
| [`docs/adapter_contract.md`](adapter_contract.md), [`docs/models.md`](models.md), [`docs/environment.md`](environment.md) | **ACTIVE** | Reference docs; not comprehensively audited against current code. |

**Every plan file has zero checked boxes**, including ones fully implemented — the checkbox mechanism is not a reliable status signal in this repo; this tree, not a plan's checkboxes, is the status source.

## Project rules — what every task must satisfy

Every script that computes or reports a number must satisfy these.
**Amend this section deliberately when a genuine exception is needed — silently violating one is unacceptable.**

Each rule names the harness step it governs and the test that enforces it.
**The step is the unit, not the task**: any task touching that step inherits its tests, whichever rung or dataset the task is about.
A task runs `uv run pytest tests/test_project_rules.py` in full — six of these are repository- or artifact-wide scans, which catch a violation the task never intended — and adds `-m` for the steps it touched, e.g. `uv run pytest -m "step_split or step_score"`.
The ten step markers are registered in `pyproject.toml`.
Each entry also names the **edge case its test cannot see, and the mechanism that closes it** — a scan over source text is weaker evidence than a behavioural check, and reading the two as equivalent is its own failure mode.
These rules stay generic: which table or driver violates one today is current state, and belongs in `docs/PROJECT_STATE.md` and the task that owns the fix, not here.
Current violations are exempted only through `KNOWN_GAPS` in `tests/test_project_rules.py` — each entry a strict `xfail` naming its owner, so a fixed one forces its own removal.

1. **One shared way of splitting samples into cross-validation folds, everywhere.**
   `fmharness.deltas.fold_assignment` produces it: sorted, deterministic, and it becomes leave-one-out once there are at least as many folds as cell lines.
   Never write the split by hand in a script (`{ln: i % n_folds ...}`), even when it looks equivalent.
   It stops being equivalent the moment two scripts happen to list the cell lines in a different order, nothing announces that, and two rungs simply stop being comparable.
   **Step** split. **Enforced by** `tests/test_project_rules.py::test_rule_01_fold_split_is_order_free_deterministic_and_degenerates_to_loo`, `::test_rule_01_no_analysis_code_writes_its_own_fold_map` (`-m step_split`).
   **Edge case** a split written in another form (`np.array_split`, a groupby, a hand-rolled loop) is invisible to a text scan.
   **Edge-case test** `::test_rule_01_edge_no_alternative_split_mechanism_in_scripts` — enforced: the named alternative constructors (`KFold`, `ShuffleSplit`, `array_split`, …) fail the scan too, with exemptions listed by file and reason.
   The residual hole, a bespoke loop no pattern matches, closes structurally when the scoring entry point takes folds only from the shared helper.
2. **One correlation and one way of averaging it, applied identically wherever two numbers are compared.**
   If rung A reports a median Pearson and rung B a mean Spearman, then `A/B` or `A - B` measures nothing — it is noise with a sign.
   Fix the correlation and the averaging once, for the comparison, rather than separately inside each script that reports a piece of it.
   **Step** score. **Enforced by** `tests/test_project_rules.py::test_rule_02_transfer_penalty_is_a_difference_of_like_quantities`, `::test_rule_02_ladder_summary_publishes_no_number_without_saying_what_it_is_of` (`-m step_score`).
   **Edge case** nothing yet states each rung's correlation and averaging in a form code can read, so two numbers can be divided by accident and the result still looks like a measurement.
   **Edge-case test** `::test_rule_02_edge_metric_declaration_exists_and_covers_every_rung_table` — strict xfail: the acceptance criterion for the declaration the scorers will read, owned by `tasks/project-rule-enforcement`.
3. **Compare only on the samples, drugs, and genes that every method can cover.**
   Every method in one table must be scored on the same (cell line or organoid, drug) pairs and the same gene panel, never each method against whatever subset it happens to share with the measurements.
   (`restrict_common_support`, `common_gene_panel` + `assert_common_genes`.)
   **Step** build, restrict. **Enforced by** `tests/test_project_rules.py::test_rule_03_every_method_is_scored_on_the_identical_support`, `::test_rule_03_a_method_on_a_different_gene_panel_is_rejected` (`-m "step_build or step_restrict"`).
   **Edge case** the guarantee holds only where a caller remembers to invoke the helpers; a new scoring path that forgets them produces a well-formed table comparing different things.
   **Edge-case test** `::test_rule_03_edge_producing_code_restricts_support_and_panel` — enforced: every promoted comparison table's producing pipeline (the script its own sidecar names, its stage family, and their `fmharness` imports) must invoke both guard families.
   Moving the calls inside the scoring entry point remains the structural close, and turns this scan into a backstop.
4. **Every reported statistic needs a test with a known answer, and that test must call the same function the analysis calls.**
   Plant a signal and require it to be recovered; plant none and require the result to come back null.
   A test that re-derives the calculation inside itself passes even when the function it is supposed to be checking is broken — which has happened here: a control's own test stayed green while the control crashed on the cluster.
   **Step** score. **Enforced by** `tests/test_project_rules.py::test_rule_04_every_reported_statistic_has_a_known_answer_test` (`-m step_score`).
   **Edge case** a test can name a statistic without planting anything, so "a test exists" is weaker than "a signal was planted and recovered."
   **Edge-case test** `::test_rule_04_edge_known_answer_tests_carry_the_marker` — enforced: every reported statistic must be named in a test file declaring `pytest.mark.known_answer`, the author's signed claim that a signal is planted and recovered, reviewed once per file.
5. **The null for an average must be built by resampling that average, not by comparing it against single draws.**
   Use `fmharness.statistics.bootstrap_aggregate_pvalue`.
   A mean or median over n pairs varies roughly √n less than a single pair does, so scoring it against individual null draws inflates p by one to two orders of magnitude.
   The same mistake appeared independently in four scripts in one week before a shared, tested version of it existed.
   **Step** null. **Enforced by** `tests/test_project_rules.py::test_rule_05_aggregate_null_recovers_a_planted_shift_and_stays_null_without_one`, plus the fuller battery in `tests/test_statistics_recover_known_answers.py` (`-m step_null`).
   **Edge case** the defect is a mismatch of kinds, not a text pattern: the same expression is correct when the null holds replicates of the same aggregate and wrong when it holds single draws, so no search distinguishes them.
   **Edge-case test** `::test_rule_05_edge_every_manual_pvalue_site_is_allowlisted` — enforced: any p-value computed outside the shared helper fails unless its site is allowlisted with the verified reason it is correct.
   This retires the open "systematic audit" item: the audit is now the allowlist, enforced instead of pending.
6. **Every comparison table needs a floor, and a positive control wherever one can be built.**
   A floor is something that must fail (shuffled labels, a constant prediction); a positive control is something that must succeed (a planted signal).
   Without both, a table of real methods cannot separate "no method predicts this" from "nothing could have been detected here at all" — and that separation is what gives a null result its meaning.
   **Step** build, null. **Enforced by** `tests/test_project_rules.py::test_rule_06_comparison_tables_carry_a_floor_and_a_positive_control`, one case per discovered comparison table (`-m "step_build or step_null"`).
   **Edge case** a control can be present in the table and still prove nothing — a positive control substituted at scoring time rather than threaded into the fit target scores at chance, and is indistinguishable from a control that genuinely failed.
   **Edge-case test** `::test_rule_06_edge_positive_control_is_recovered_not_just_present` — enforced: in every table carrying both, the planted row must beat every floor row on the table's primary score; a positive control at chance fails.
7. **However much flexibility a model is allowed, every method in the same table gets it the same way.**
   Choose the number of components and the penalty strength by cross-validation, or fix identical values for every method — never one method tuned and another left at a value written into the script, even when that value came first and still looks reasonable.
   **Step** fit. **Enforced by** `tests/test_project_rules.py::test_rule_07_drivers_do_not_pin_capacity_to_a_literal`, one case per submission script (`-m step_fit`).
   **Edge case** a driver can avoid pinning capacity on the command line and still pin it in code.
   **Edge-case test** `::test_rule_07_edge_tables_record_selected_capacity` — strict xfail: the acceptance criterion for the selected-capacity column, owned by `tasks/project-rule-enforcement`.
8. **Every promoted result carries its `.provenance.json` record beside it** — commit, job id, the exact arguments used, and checksums of the inputs and the log — written from a checkout with no uncommitted changes, at the commit that produced the result.
   No record, not evidence.
   **Step** promote. **Enforced by** `tests/test_project_rules.py::test_rule_08_every_promoted_result_carries_a_complete_provenance_record` (`-m step_promote`).
   **Edge case** the clean-checkout half cannot be reconstructed after the fact — nothing recorded it.
   **Edge-case test** `::test_rule_08_edge_sidecars_record_clean_tree_and_producing_commit` — strict xfail: the acceptance criterion for the promoter recording both, owned by `tasks/project-rule-enforcement`.
9. **Embargo is checked value by value, on what the cells contain, not on what the columns or the files are named.**
   A column named `line` is not the only place a patient identifier appears, and a table that merely lists another table's column names carries those identifiers as ordinary text.
   `data/release_manifest.yaml`'s three tiers cover every file: anything not explicitly classified stays embargoed.
   **Step** load, release. **Enforced by** `tests/test_project_rules.py::test_rule_09_embargo_gate_sees_an_identifier_inside_a_cell_value` (`-m "step_load or step_release"`).
   **Edge case** an identifier can sit in a cell of a column whose name looks innocuous, which a header-driven scan never opens.
   **Edge-case test** `::test_rule_09_embargo_gate_sees_an_identifier_inside_a_cell_value` — strict xfail: the gate must find an embargoed identifier inside an innocuously-named column's cell, owned by `tasks/embargo-gate-cell-values`.
10. **Reversing an analysis design, a method, or a data source needs a written decision, not just a commit.**
    If work undoes something a spec or plan established — a data source, a scoring rule, a control — write one paragraph in `docs/decisions/` saying what changed and why, however terse and however late.
    The CoderData → custom-loaders reversal (a full revert, `1bfb922`→`93dc76f`, merged as "restore-custom-loaders", 2026-06-16) left *zero* trace in any document here for over two months; the only record was git log and one person's memory.
    See `docs/decisions/2026-06-16-revert-coderdata-loaders.md` for the recovered entry.
    It is the sharpest example of the problem this file exists to prevent, and it must not repeat.
    **Step** document. **Enforced by** `tests/test_project_rules.py::test_rule_10_every_task_and_decision_is_named_in_the_spec_tree` (`-m step_document`).
    **Edge case** a reversal nobody writes down is undetectable mechanically, and no CI heuristic changes that.
    **Edge-case test** none, deliberately: an unwritten reversal is undetectable mechanically, and stays a human read of git log against `docs/decisions/`. The primary test catches only the next failure along — a document that exists but is indexed nowhere.

## Known drift points — read before you touch these areas

Concrete case studies, kept here permanently (not in PROJECT_STATE.md, which only tracks current open/fixed status) because the *lesson* stays useful after the specific confusion is resolved.

- **A newer, more authoritative-looking document can itself be the source of drift.**
  `transfer_ladder_protocol.md` is the newest, most-referenced design doc in the repo, and its Invariant 2 is the one that's wrong — it was written without checking the two older specs that correctly describe the code it's supposedly formalizing.
  Recency is not the same as accuracy; a new spec must check existing specs and code, not just state intent.
- **Renaming a concept without a forward pointer leaves every prior document silently ambiguous.**
  `additive` → `observed_delta` (D6) and `soragni` → `sarcoma_organoids_2024` (`a6c8976`) both have a working *code* alias/rename, and zero *documentation* trail pointing old-name readers to what changed.
  A rename is a two-line fix in the doc that used the old name; do it at rename time, not retroactively.
- **A parallel implementation that bypasses an abstraction silently drops that abstraction's guarantees.**
  The registry-driver orphaning (spec tree above) is the concrete instance: three specs' worth of leakage-filtering design is not running on any promoted number, and discovering that took a targeted review, not a doc anyone could just read.
- **An architectural reversal with no decision entry is invisible to everyone but git log.**
  The CoderData → custom-loaders revert sat undocumented for over two months.
  See project rule 10 and `docs/decisions/2026-06-16-revert-coderdata-loaders.md`.

## Process for new tasks

1. **Every task gets its own spec.**
   Create `docs/tasks/<task-slug>/design.md` — one folder per task, slug named for the work (`cross-check-fairness-and-capacity`), never date-prefixed — and add its branch to the spec tree above — under its rung, or under cross-cutting — linking the spec, in the same change.
   The plan (`plan.md`) and any task-local decisions (`decisions.md`) live beside it in that folder.
   The slug is the task's identity everywhere: the tree keys on it, `docs/PROJECT_STATE.md` links to it, and commit messages can name it.
2. **Before writing that spec**, read this file's project rules and spec tree.
   If the new work overlaps an ACTIVE task, extend that task's `design.md` or explicitly supersede it — don't open a second task on the same subject.
3. **A new spec's header names the harness steps the task touches, the project rules it relies on, and the tests it must pass.**
   The steps are the ten registered as markers in `pyproject.toml`: load, build, restrict, split, fit, score, null, promote, release, document.
   Naming them is what tells the next reader, and whoever reviews the PR, which tests this task had to pass — without anyone reconstructing it from the diff.
4. **If the new work needs an exception to a project rule**, that exception is written into *this* file (a numbered sub-point under the rule, with the reason), not left implicit in the new spec alone — otherwise the next task that touches the same code has no way to know the exception exists and will "fix" it back, or worse, not know it was ever a deliberate choice.
5. **When a new spec/decision supersedes or reverses an older one**, do both of:
   - Add a one-line dated banner at the top of the *old* document pointing to the new one (the pattern already used ad hoc in `docs/l1000_imputation_fidelity.md`'s correction banner and the 2026-08-13 handoff's inline status note — make it the standard, not the exception).
   - Update this file's spec-tree branch for the old task.
6. **A task's own tests live in its spec; a test that should bind the whole project is promoted here.**
   Write the task-specific tests into `docs/tasks/<task-slug>/design.md` — they are what shows *that* update satisfied the rules it touches.
   When the work establishes that every task should pass one of them, move it into the project rules above, with its step, its enforcing test, and what a pass does not prove; the task's spec records that it was promoted.
   A test that never generalises simply stays in the task, which is the normal case.
7. **If the new work changes a currently-reported number**, update `docs/PROJECT_STATE.md` in the same change — a spec describing new intent and a state doc still showing the old number is exactly the drift this structure exists to prevent.
   That state entry links back to the task's `design.md`, the commits that changed the code, and the promoted output, so the number, the intent behind it, and the artifact it came from are one hop apart in either direction.
