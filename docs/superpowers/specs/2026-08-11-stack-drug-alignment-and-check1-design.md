# Stack drug-alignment + registry-driven Check 1 — design

**Date:** 2026-08-11
**Status:** approved, pending implementation plan
**ORPHANED, 2026-08-26:** `check1_registry_driver.py` correctly implements this spec but is not
invoked by any current sbatch — the promoted rung-1 result (Check 1 in this doc's terms) comes
from a parallel path that bypasses it. See `docs/PROJECT_SPEC.md`'s spec index.

## Purpose

Resume `docs/superpowers/specs/2026-08-06-arm2-harness-validation-design.md`'s Phase 1
(sci-Plex drug-alignment of the Stack generation checkpoint) and the Check-1 slice of its
Phase 2 (registry-driven ladder reproduction), now that the modular-harness-core plan has
landed the `Encoder`/`Generator`/`LeakageQueryable` registries that Phase 2 originally sketched
in a different, earlier shape.

Concretely: get a real, drug-aligned Stack generation checkpoint (the base `bc_large.ckpt` has
no generation head at all -- the alignment step is what adds one, and the only generator that
currently exists is aligned on cytokines, not drugs), and report its Check-1 delta-Pearson next
to the existing table (`docs/tahoe_generation_results.md`) through the harness-core `Generator`
protocol, proving the registries are swappable for the exact use case they were built for.

## Why Check 1 only, not the full ladder

`docs/tahoe_generation_results.md` bundles two separate evaluations under "the ladder": Check 1
(delta-Pearson generation quality -- what Phase 1's own acceptance criterion asks for) and
Check 2 (a much larger end-to-end evaluation: penalized L1/L2/EN regression against GDSC2 AUC,
selection gap@k, per-drug/p_label, plus a proposed-but-never-built MOA-stratification
extension). Check 2 is a substantially bigger, separable unit of work -- most of what the
original Arm-2 spec's Phase 2 described as its own phase. This spec scopes to Check 1: it is
directly what "did the Stack update work" asks, and it is what actually needs the swappable
`Generator` registry (Check 2's baseline rows -- additive/knn/pca/nmf -- are statistical delta
sources, not model generators, and are untouched by this work). Check 2 remains future,
separate work.

## What already exists and does not need to change

- `Generator` protocol (`src/fmharness/model_protocols.py`) -- `generate(baseline, perturbation)
  -> AnnData`, `context_coverage(perturbations) -> set[str]`, already proven against
  `MockGenerator`. The cytokine-aligned Stack checkpoint's existing generation path
  (`scripts/alpine/04_stack_generate.sbatch`) is the real, non-mock implementation this
  protocol was written to cover; the drug-aligned checkpoint will be a second instance of the
  same shape.
- `delta_fidelity` (`src/fmharness/evaluation.py:318`) -- the exact Check-1 scoring function:
  per-(patient, drug) Pearson between predicted and real log-fold-change, plus `r_offdiag` and
  specificity `rank`. Already tested (`tests/test_evaluation.py::test_delta_fidelity_restricts_to_hvgs`).
  Unchanged.
- `build_additive_deltas` / `build_learned_deltas` (`src/fmharness/deltas.py`) -- the
  additive/knn/pca/nmf baseline delta sources. Unchanged; these are not the swappable
  dimension this work is about.
- `filter_leakage` (`src/fmharness/leakage.py`) -- directly covers the spec's own stated
  handling for Check 1's 6 doubly-exposed (line, drug) pairs ("exclude the 6 doubly-exposed
  pairs from Check 1, or report with and without") -- exactly the tiered-drop rule this
  function already implements, now with a real caller (wired earlier this session).

## What is new

### Phase 1 — sci-Plex drug alignment (Alpine aa100)

Four blockers were found (2026-08-06) but never fixed; `/scratch/alpine/$USER/sciplex_finetune/`
is still empty and no drug-aligned checkpoint exists.

1. **Gene-panel mismatch.** `scripts/alpine/08_sciplex_prep.sbatch` currently downloads
   chemCPA's `sciplex_complete_middle_subset.h5ad`, pre-subset to 2,000 HVGs -- Stack's panel
   is 15,012 genes, Tahoe's context is 14,725. Fine-tuning the generation head on 2,000 genes
   and then generating over 14,725+ is a train/test mismatch the original spec calls out as
   confounding. Fix: point `08` at the scPerturb-hosted `SrivatsanTrapnell2020_sciplex3.h5ad`
   (Zenodo record 13350497, 2.5 GB -- far too large for 650k cells at 2,000 genes, so almost
   certainly a much larger gene set, plausibly full transcriptome).
   `scripts/build_sciplex_finetune.py` already auto-detects this exact file's schema (its
   "scPerturb" flavor, documented in the script's own docstring) -- the switch is a URL change,
   not new parsing code. **Verify the actual shape once downloaded before relying on it**; if it
   turns out to also be gene-subset, fall back to reprocessing sci-Plex 3's raw GEO series
   (GSE139944) directly.
2. **Raw-counts unverified.** `08`'s log prints a literal `(VERIFY these are raw counts, not
   normalized)` warning that nothing ever acted on. Stack is a count model (NB likelihood);
   normalized input breaks it. Add an actual check (non-negative, integer-valued up to float
   rounding) against whichever layer/`.X` the new source uses, failing loudly if it does not
   hold, rather than only warning.
3. **Drug-name truncation.** Found in the chemCPA source specifically (`pert_id` truncated at
   first whitespace: "AZ", "GSK", etc.). Re-check against the new source's own perturbation
   column -- may not carry over to a different upstream file; do not assume the old finding
   still applies.
4. **`GLIBCXX_3.4.29` import crash.** The `stack` conda env's numpy is built against a newer
   libstdc++ than the Alpine node provides. Fix via `LD_LIBRARY_PATH` ordering
   (`$CONDA_PREFIX/lib` ahead of `/lib64`) in the sbatch scripts that invoke `stack-finetune` /
   `stack-generate`, not by mutating the shared conda env.

Then: promote `scripts/alpine/09_stack_finetune.sbatch` from its current smoke-test
configuration (`--qos=gpu-testing`, `--max_epochs 1`, `--time 1:00:00`) to a real run --
`gpu-normal`, real epoch count, `--time` set from the observed per-epoch time on the smoke run.

**Alpine mechanics (unchanged from the existing spec):** `ralpine` is read-only by design --
Claude authors and commits sbatch scripts, Lucas submits them, Claude polls `ralpine sq` /
`sacct` / `log` and pulls results with `ralpine pull`. No change to this loop; this spec does
not touch Alpine access/safety, which remains explicitly deferred from earlier this session.

**Acceptance:** a drug-aligned checkpoint exists at `/scratch/alpine/$USER/sciplex_finetune/`;
pulled locally for Phase 2 to score.

### Phase 2 — Check 1 through the registries (local)

A small driver (new script, name TBD at planning time) that:

1. Wraps the cytokine-aligned and drug-aligned Stack checkpoints as two `Generator` instances
   (thin adapters over the existing `04_stack_generate.sbatch`-produced output, or a direct
   `generate()` call if the checkpoint is available locally -- to be settled at planning time
   based on whether generation still needs to run on Alpine or can run from a pulled
   checkpoint).
2. Loads the real Tahoe pseudobulk delta and the existing additive/knn/pca/nmf baseline deltas
   exactly as `scripts/score_generation_eval.py` already does (no reimplementation).
3. Applies `filter_leakage` against each Stack `Generator`'s declared pretraining corpus,
   producing the table with and without the 6 doubly-exposed pairs.
4. Scores every source (baselines unchanged, both Stack variants via their `Generator.generate()`
   output) through the existing, unchanged `delta_fidelity`.
5. Reports a table in the same shape as `docs/tahoe_generation_results.md`'s Check-1 table,
   with the drug-aligned row added.

**Acceptance:** re-running Check 1 through this driver reproduces the existing table's
additive/nmf/pca/knn/stack(cytokine-aligned) numbers exactly, or every difference is explained
and justified in writing (matching the original Phase-2 acceptance bar) -- plus a new
stack(drug-aligned) row, with and without the 6 leaked pairs.

## Sequencing

Alpine GPU turnaround is real, async wall-clock time this session cannot shortcut (submit,
queue, train, pull). Rather than block on it: do Phase 1's local prep (the 4 fixes above) and
hand off the corrected sbatch scripts, then start Phase 2's driver immediately against the
existing cytokine-aligned checkpoint and baselines (already available, reproduces
today's published numbers on its own as a correctness check of the driver itself),
and fold the drug-aligned checkpoint in as the new row once training completes.

## Out of scope

- Check 2 (end-to-end GDSC2 AUC ladder, selection gap@k, MOA stratification) -- separate,
  future work per the original spec's own phase boundary.
- Phases 3-6 of the original Arm-2 spec (positive controls/MDE, n~500 re-anchor, panel-scale
  Stack embeddings, full LOO/LODO CV) -- untouched, still future.
- Alpine safety hardening -- explicitly paused earlier this session, not reopened here.
- Any change to the Soragni/Arm-1 cohort or PDTO application.
