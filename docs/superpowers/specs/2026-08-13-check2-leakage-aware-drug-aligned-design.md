# Leakage-aware Check 2, drug-aligned Stack checkpoint — design

**Date:** 2026-08-13
**Status:** approved, pending implementation plan

## Purpose

Resolve the open question in `docs/superpowers/specs/2026-08-13-check2-leakage-aware-drug-aligned.md`:
Check 2 (end-to-end GDSC2 AUC prediction) has no leakage filtering, unlike Check 1
(`scripts/check1_registry_driver.py`), so scoring the drug-aligned Stack checkpoint through it
today would be silently blind to the checkpoint's measured pretraining overlap with the eval
cohort (A549 / 5 drugs, `doubly_exposed_frac=0.003` on the Check-1 cohort).

Concretely: build a leakage-aware, registry-driven Check-2 driver (continuing the harness-core
direction `check1_registry_driver.py` established), run it for both the cytokine-aligned and
drug-aligned checkpoints, and propagate the results into `docs/tahoe_generation_results.md` and
`docs/prospective_evaluation_harness_overview.pptx`.

## What already exists and does not need to change

- `filter_leakage` / `LeakageQueryable` (`src/fmharness/leakage.py`) — the tiered drop rule
  (always drop doubly-exposed, drop single-axis only when `task_signal_in_pretrain="direct"`).
  Unchanged; this design reuses it against a different frame (see below), not a different rule.
- `PregeneratedStackGenerator` (`src/fmharness/models/stack_generator.py`) — the
  `LeakageQueryable` declaration over Stack's pre-generated output. Unchanged.
- `score_generation_eval.py`'s Check-2 machinery — the fixed-signature-readout scoring, the
  representation-controlled penalized grid (`RidgeCV`/`LassoCV`/`ElasticNetCV`, tuned per
  representation), `regret_norm_at_k`/`interaction_rho`/`per_drug_spearman` (all
  `fmharness/evaluation.py`). Unchanged in behavior — Task 1 below relocates it, does not rewrite
  it.
- `loo_baseline_source`, `learned_gene_panel`, `build_generated_deltas`, `build_tahoe_deltas`
  (`fmharness/deltas.py`) — the delta-source builders both checks already share. Unchanged.
- The measured pretraining-overlap corpus from Check 1 (line `ACH-000681`; drugs CID
  `6918289,11626560,104741,11707110,3385`) — reused as-is for the drug-aligned checkpoint's
  corpus declaration.
- The "gate" (Hallmark-vs-random-genes readout-power check, `score_generation_eval.py`) — scores
  the real Tahoe delta against AUC, not any model's output, so a checkpoint's pretraining overlap
  does not bear on its validity. Explicitly **not** filtered, and out of scope for this work.

## What is new

### The leakage-filtering point (the design decision this doc resolves)

Check 1 filters `real_key` (Tahoe's ground-truth key) because `real_key` **is** Check 1's
evaluation set. Check 2's evaluation set is the GDSC2 AUC `design` frame
(`patient, drug, y` — `patient`=DepMap id, `drug`=PubChem CID — from `build_sample_design`),
structurally identical in shape to `real_key`. The same `filter_leakage` call, pointed at
`design` instead of `real_key`, is the entire mechanism:

- Filter `design` once, via the same `PregeneratedStackGenerator` declaration Check 1 uses.
- `design_target` (today: `design[design["drug"].isin(target_drugs)]` in
  `score_generation_eval.py`) is derived from the **already-filtered** `design` — no second
  filter call.
- Every representation (`expr`, `additive`, `knn`, `pca`, `nmf`, `stack`, any `--stack-emb`) is
  scored via a merge/groupby against `design`/`design_target` downstream, so filtering it once
  uniformly restricts every representation to the same surviving pairs — the same
  same-pair-count parity Check 1's table already has, without touching how any individual source
  is built.
- The delta **sources** themselves stay built from the full, unfiltered Tahoe triple
  (`real_delta`/`real_key`/`base`). A source's prediction for a contaminated pair sitting unused
  in the `sources` dict is harmless; it never gets scored once `design` excludes that pair.
- No positional `_ROW` tag is needed (unlike Check 1's `filter_leakage` call, which has to
  survive a `reset_index()` and be re-aligned to `real_delta` by position) — `score_check2`'s
  logic works entirely through `(patient, drug)` merges/groupbys, never positional slicing.

### Architecture

- **New `src/fmharness/check2.py`.** The constants (`PROLIFERATION`, `FIXED_READOUTS`,
  `PENALTY_NAMES`) and helpers currently private in `scripts/score_generation_eval.py`
  (`_make_penalty` → `make_penalty`, `_load_line_matrix` → `load_line_matrix`, `_repr_by_drug` →
  `repr_by_drug`, `_penalized_preds` → `penalized_preds`), plus a new
  `score_check2(sources, real_key, base, hvg, design, *, hallmark, fixed_methods=FIXED_READOUTS,
  penalties=PENALTY_NAMES, folds=5, stack_emb=None, n_permutations=1000) -> pd.DataFrame` — the
  composition of today's fixed-readout loop and representation-grid loop
  (`score_generation_eval.py`'s current `main()`, lines building `out`). `score_check2` stays
  leakage-agnostic, exactly like `score_delta_sources` in `evaluation.py`: it scores whatever
  `design` it is handed, and does not know `filter_leakage` exists.
- **`scripts/score_generation_eval.py`**: pure refactor to import from `fmharness.check2`
  instead of defining these inline — the same move `loo_baseline_source`/`learned_gene_panel`
  already made into `fmharness/deltas.py`. No behavior change; its own CLI/output stay identical.
- **New `scripts/check2_registry_driver.py`**: mirrors `check1_registry_driver.py`'s shape.
  `run_check2(real_delta, real_key, base, *, query_baseline, generated_dir, pert_to_drug,
  checkpoint_label, hallmark_path, auc_design, n_hvg=2000, k=10, fixed_methods=FIXED_READOUTS,
  penalties=PENALTY_NAMES, folds=5, stack_emb=None, n_permutations=1000, pretraining_lines=None,
  pretraining_drugs=None, task_signal_in_pretrain="none") -> pd.DataFrame`:
  1. Builds the `PregeneratedStackGenerator` from the corpus args (same as `run_check1`).
  2. `filter_leakage(auc_design, model)` — directly, no `_ROW` tag.
  3. Prints the same `basis=measured/unknown, doubly_exposed_frac=...` line `run_check1` prints.
  4. Builds `sources` (additive/knn/pca/nmf/stack) from the **unfiltered** Tahoe triple, exactly
     as `score_generation_eval.py` does today.
  5. Calls `score_check2(sources, real_key, base, hvg, filtered_design, hallmark=..., ...)`.
  `main()` is a thin CLI wrapper: `--context`/`--deltas-bundle`, `--query-baseline`,
  `--generated-dir`, `--pert-map`, `--checkpoint-label`, `--auc-tranche`, `--n-hvg`, `--k`,
  `--hallmark-path`, `--methods`, `--penalties`, `--folds`, `--stack-emb`, `--corpus-lines`,
  `--corpus-drugs` — the union of `check1_registry_driver.py`'s and `score_generation_eval.py`'s
  own flag sets, nothing new invented.
- **`fmharness/leakage.py` gains** `corpus_declared_partially`, `ground_truth_source_declared_ambiguously`,
  and `parse_corpus_set` — moved from `check1_registry_driver.py` (currently module-level
  functions there) so both driver scripts import one copy instead of
  `check2_registry_driver.py` importing from a sibling script. `check1_registry_driver.py`'s
  existing tests for these three (`test_parse_corpus_set_*`, `test_corpus_declared_partially_*`,
  `test_ground_truth_source_declared_ambiguously_*`) move to wherever `leakage.py`'s tests live,
  import-path only — no behavior change.

### Testing

- `tests/test_check2.py` (new): `make_penalty`/`repr_by_drug`/`penalized_preds` unit tests, and
  a `score_check2` test against small synthetic fixtures (shape: columns, one row per
  representation × method).
- `tests/test_check2_registry_driver.py` (new), mirroring `test_check1_registry_driver.py`'s
  structure: reports rows for every representation; a declared corpus drops the same
  (patient, drug) pair from **every** representation's scored set, not just `stack` (the parity
  property the filtering-point decision above depends on); an undeclared corpus leaves every
  representation's pair count unchanged; the leakage-basis print line is correct in both cases.
- Existing `tests/test_check1_registry_driver.py` and score-generation-eval tests (if any) keep
  passing unmodified except for the import-path move noted above.

### Running for real, and propagating results

Once the driver exists, run it three times against data already pulled into this worktree
(`generated/` = cytokine-aligned, 33 files; `generated_sciplex/` = drug-aligned, 33 files; GDSC2
tranche data already present under `data/raw/coderdata/`):

1. `--generated-dir generated --checkpoint-label cytokine-aligned` (no corpus flags — this
   checkpoint was aligned on CELLxGENE + Parse PBMC cytokines only, no Tahoe/GDSC2 line or drug
   overlap to declare).
2. `--generated-dir generated_sciplex --checkpoint-label drug-aligned-unfiltered` (no corpus
   flags — same run, deliberately undeclared, for the filtered-vs-unfiltered comparison).
3. `--generated-dir generated_sciplex --checkpoint-label drug-aligned` `--corpus-lines
   ACH-000681 --corpus-drugs 6918289,11626560,104741,11707110,3385` (leakage-filtered).

**Table shape.** Both `docs/tahoe_generation_results.md`'s existing tables (Check 1, and Check
2's fixed-readout + representation-ladder tables) and the deck's mirrors of them
(`scripts/update_harness_overview_slides.py`'s `CHECK1_ROWS`/`SIG_ROWS`/`LADDER_ROWS`) are
one-row-per-source; a wide per-checkpoint-variant column layout (like the scratch comparison
table in the handoff doc) does not fit them, since only the `stack` row has checkpoint variants
and every baseline row (`additive`/`knn`/`pca`/`nmf`/`expr`) has exactly one. **Long format**:
replace the single `stack (gen)` row with three rows — `stack (gen, cytokine-aligned)`,
`stack (gen, drug-aligned, unfiltered)`, `stack (gen, drug-aligned, leak-excluded)` — everything
else in each table unchanged. This also closes the still-open item from the handoff doc (Check
1's drug-aligned row was never written into `docs/tahoe_generation_results.md`) using the same
long-format convention.

**Deck.** `docs/prospective_evaluation_harness_overview.pptx` exists only in the main worktree
(`/Users/gillenlu/Repositories/fm-pdo-evaluator/docs/`, gitignored — this worktree never got a
copy). Copy it into this worktree, update
`scripts/update_harness_overview_slides.py`'s hardcoded row constants and
`scripts/plot_generation_eval_summary.py`'s hardcoded dicts with the new numbers (both scripts
already document that they hand-transcribe `docs/tahoe_generation_results.md`'s reported values
— no new plumbing, same pattern as today's cytokine-only numbers), re-run both scripts, then
copy the finished `.pptx` back to the main worktree.

**Acceptance:**
- `check2_registry_driver.py` reproduces `score_generation_eval.py`'s existing (cytokine-aligned,
  unfiltered) Check-2 numbers exactly when run with no corpus declared, proving the refactor
  changed no behavior.
- The drug-aligned leak-excluded run's `doubly_exposed_frac` is printed and matches (or is
  explained if it differs from) Check 1's measured 0.003 on the same declared corpus.
- `docs/tahoe_generation_results.md` and the deck both show all three `stack` variants for both
  Check 1 and Check 2, long-format, with a written takeaway sentence (matching the doc's existing
  style) on whether leakage filtering changed anything for Check 2 the way it didn't for Check 1.
