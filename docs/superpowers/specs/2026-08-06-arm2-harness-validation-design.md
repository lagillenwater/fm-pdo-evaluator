# Arm-2 harness validation — design

**Date:** 2026-08-06
**Status:** approved, pending implementation plan

## Purpose

Validate the evaluation harness and the RNA -> sensitivity adapter on cell lines, before
either is applied to the Soragni PDTO cohort. Nothing here is about PDTOs. The PDTO
application is downstream and out of scope for this spec.

## Why the harness needs validating first

The project has two arms that have never been run on each other's cohort:

| | Arm 1 (PDTO-facing) | Arm 2 (Tahoe) |
|---|---|---|
| cohort | GDSC2 **sarcoma only** | Tahoe-100M, **pan-cancer** |
| n lines | 28 native / 57 CoderData | ~50 |
| test set | Soragni sarcoma PDTOs, n=17 | — |
| headline | Stack <= PCA; screen-free top-1 fails | base Stack embedding interaction **+0.119** |

The sarcoma restriction is hardcoded at `scripts/gdsc_representation_increment.py:56`
(subtype list at `:32-39`) and defaulted at `scripts/transfer_gdsc_soragni.py:101`, whose
docstring at `:11-12` records that the full pan-cancer panel washes the interaction out.

So the single positive result in the project (Arm 2, +0.119) and the main negative
(Arm 1, Stack <= PCA) sit on different cohorts at different n. They are not comparable, and
neither has been reproduced on the other's data.

This spec commits to **Arm 2** as the harness-validation vehicle. Arm 1 is tabled.

### Why the cohort is all cancer

Both ends bind independently. GDSC2, CTRP and PRISM are cancer-cell-line panels by
construction -- panel-scale dose-response viability does not exist for non-cancer lines.
And Tahoe-100M is itself 50 cancer lines. "Cancer" is the intersection of the only two
datasets carrying both single-cell drug perturbation and matched viability. This is an
external-validity limit to state in the writeup, not a defect to fix. The harness itself is
cohort-agnostic: `--auc-tranche` (`scripts/score_generation_eval.py:248`) and
`--stack-emb label=path` (`:297`) are both pluggable.

## What is currently unvalidated

1. **No positive control in Arm 2.** `plant_response` / `plant_interaction`
   (`src/fmharness/controls/`) are wired only into `scripts/per_patient_eval.py:72,508`,
   which is Arm 1. `score_generation_eval.py` carries only the permutation negative
   (`p_label`). We know what the harness reports when there is no signal; we do not know what
   it reports when there is. +0.119 is uncalibrated upward -- no minimum detectable effect,
   no statement of what fraction of recoverable signal was captured.

2. **The selection metric may be invalid.** `docs/tahoe_generation_results.md:167-175`
   argues the potency prior (rank drugs by training-fold mean AUC, ignore the cell line)
   should score gap@1 ~0.06-0.11 against 0.22-0.36 for every representation -- i.e. the prior
   wins. The prior is already inside the fitted models: `_penalized_preds` uses
   `StandardScaler` + `fit_intercept=True` (`score_generation_eval.py:179-182`), so
   `model.intercept_` *is* the training mean. It has never been scored because the per-pair
   prediction frame is built, scored and discarded (`:465-468`).

3. **No lineage baseline.** Arm 1 has a subtype one-hot (interaction 0.138). The Arm-2 ladder
   has expr/pca/nmf/additive/knn and nothing for tissue of origin. Across 50 pan-cancer lines
   an embedding that encodes lineage will show line-specific response partly *because* lineage
   predicts response.

4. **Fold assignment is unseeded and unreplicated.** `fold_of = {ln: i % n_folds}`
   (`score_generation_eval.py:410`) is deterministic, unstratified, and never averaged over
   draws. +0.119 is one fold assignment, not a mean.

5. **n = 50.** Every Arm-2 number rests on 50 lines, against a
   `results/label_ceiling.csv` ceiling computed on 513.

## Assets already local (no download required)

- `data/raw/gdsc2_sarcoma/depmap/OmicsExpressionRawReadCountHumanProteinCodingGenes.csv`
  -- full DepMap 26Q1 panel, **raw counts**, all lineages, 129 MB, checksummed in
  `manifest.json`. Raw counts are what Stack expects.
- `data/raw/gdsc2_sarcoma/gdsc2/GDSC2_fitted_dose_response_27Oct23.xlsx` -- full release 8.5.
- `data/raw/gdsc2_sarcoma/gdsc2/screened_compounds_rel_8.5.csv` -- 621 compounds carrying
  `TARGET` and `TARGET_PATHWAY`. This is the MOA join that
  `docs/tahoe_generation_results.md:139-140` records as "not in the current context map";
  it is present locally, under the current GDSC column names rather than the
  `PUTATIVE_TARGET` / `PATHWAY_NAME` the doc cites.
- `load_gdsc2_sarcoma()` **already defaults to the full panel** (`:106-109`); `sarcoma_only`
  is the opt-in. The n~500 pan-cancer cohort is loadable with existing code.
- `results/label_ceiling.csv` -- GDSC2<->CTRPv2 interaction **0.466** over 513 lines / 71
  drugs; GDSC2<->PRISM 0.307 over 311 lines / 105 drugs. The known-answer benchmark.

## Cross-validation: two different problems

**Leave-one-line-out is a flag.** `--folds 999` gives
`n_folds = min(args.folds, len(uniq_lines))` and one line per fold
(`score_generation_eval.py:409-410`). Cost is linear in folds -- `_penalized_preds` is
n_drugs x n_folds fits -- so ~10x at n=50 and ~100x at n~500. Perfectly parallel over drugs.

**Leave-one-drug-out is a different estimator.** `_penalized_preds` fits an independent ridge
per drug (`:159-186`). A held-out drug has no model; LODO is undefined under that
architecture, not merely expensive. It requires sharing information across drugs through drug
features -- `bilinear_features` (`src/fmharness/bilinear.py:17`),
`AUC(s,d) = ridge([z_s, g_d, z_s (x) g_d])` -- which exists but lives only in Arm 1
(`per_patient_eval.py:224`, `transfer_pharmaformer_lite.py`).

**Which carries the claim.** The PDTO application screens *known* drugs on a *new* organoid.
Leave-one-line-out is therefore the primary generalization test; leave-one-drug-out is the
stress test.

**Policy.** Development runs at 5-fold. Full CV (LOO + LODO) runs once, on the frozen
configuration, in Phase 6. The MDE is fold-scheme-dependent -- a recovery curve computed at
5-fold does not transfer to LOO -- so Phase 3 plants at both schemes at n=50.

## Modular architecture (Phase 2)

Today every swap is a script edit: representations are an inline dict of lambdas
(`score_generation_eval.py:455-462`), folds a dict comprehension (`:410`), penalties a list,
metrics hardcoded in `score_predictions`. Four registries, each a Protocol plus a spec-string
loader, so features can be swapped without touching the driver.

| registry | protocol | must admit |
|---|---|---|
| Representation | `name -> (drug: str) -> DataFrame[line x feature]` | expr, PCA, NMF, lineage one-hot, Stack embedding (drug-independent); additive/knn/pca/nmf/stack deltas (drug-dependent) |
| Estimator | `fit(features, drugs, y, groups)` / `predict_parts(...)` | per-drug penalized ridge (current); shared-drug bilinear (Phase 6) |
| CV scheme | `splits(design) -> iterable[(train_idx, test_idx)]` | leave-line-out, leave-drug-out, repeat-k-fold(seed) |
| Readout | `(preds) -> dict[str, float]` | raw-AUC gap@k, percentile-within-drug gap@k, MOA hit-rate@k, interaction/global/per-drug rho |

Notes:
- The representation signature already exists implicitly -- `:462` builds
  `lambda _drug: e`. This formalizes it.
- The estimator protocol must **reuse** `ProbeBase` (`src/fmharness/probe/`), which already
  defines `fit`/`predict_parts` for Arm 1. Do not introduce a second estimator protocol.
- The CV registry replaces the unseeded `i % n_folds` assignment and is what makes Phase 3's
  repeat-CV and Phase 6's LOO/LODO configuration changes rather than code changes.

## Phases

Phase 0 and Phase 1 are independent and start together -- 0 is local CPU, 1 holds a GPU.

### Phase 0 — close the loop on Tahoe (local)

Surgical, ~10 lines plus analysis. Deliberately **not** blocked on the Phase 2 refactor.

1. Emit `y_prior` in the `_penalized_preds` fold loop and dump
   `(source, penalty, patient, drug, y_true, y_pred, y_prior)` to
   `results/check2_preds.parquet`. `y_prior` is the training-fold mean AUC for that drug --
   the fitted model with coefficients zeroed.
2. Score the potency prior with the same gap@k on the same folds. **Expected: the prior
   wins** (~0.06-0.11 vs 0.22-0.36).
3. Shortlist concentration table: distinct drugs ever ranked #1, modal drug and its share,
   share of #1 picks that are broadly active -- against the observed reference. Reuse
   `_personalization` (`per_patient_eval.py:444`), which already computes these columns and
   carries the observed row. Reference at n=50 is ~6 distinct drugs (95% band 4-8), modal
   share ~0.69 (band 0.58-0.80); across 955 GDSC2 lines the observed best is one of only 13
   compounds, Staurosporine for 69%.
4. If concentrated, add the percentile-within-drug selection metric (out-of-fold rank within
   drug). Random is exactly 1/(k+1) and the prior lands there by construction.
5. MOA readouts: join `TARGET` / `TARGET_PATHWAY` from
   `screened_compounds_rel_8.5.csv` on `DRUG_ID`/`DRUG_NAME`. MOA hit-rate@k, and
   interaction stratified targeted vs
   cytotoxic. Control hit-rate against the pan-active base rate (shuffled shortlist).
   At 50 lines this is illustrative, not powered -- report it as such.

**Why first:** it decides which selection metric Phases 4-6 are scored with. Running n~500
or a GPU embedding job under a metric about to be discarded is wasted compute.

**Acceptance:** `results/check2_preds.parquet` exists; prior-vs-representation gap@k table
produced; concentration table produced with the observed reference row; a decision recorded
on whether selection is scored in raw AUC or percentile-within-drug.

### Phase 1 — sci-Plex drug alignment (Alpine aa100)

The base `bc_large.ckpt` has no generation head (`query_pos_embedding`, `cls` absent) -- the
alignment step adds it. So the cytokine-aligned generator is currently the *only* generator
in existence, and there is no unaligned control. Drug-aligning on sci-Plex produces a second
generator whose alignment domain matches the test domain. This is the missing control for
"is the Check-1 null the cytokine domain, or generation itself." It is also the long pole
(GPU queue + training) and depends on nothing.

**Leakage check — DONE 2026-08-06, result below.** Not a blocker.

Measured from `tahoe_query.h5ad` obs (50 rows, 49 with a DepMap id + 1 `None`),
`context_by_drug/pert_to_cid.tsv` (33 rows, 32 unique compounds -- Trametinib appears twice,
both CID 11707110), and `sciplex_finetune.h5ad` obs (3 lines, 188 pert categories incl.
`control` -> 187 compounds).

- **Line overlap: 1 of 49.** A549 (`ACH-000681`). K562 and MCF7 are not in Tahoe.
- **Drug overlap: 6 of 32.** Temsirolimus, Crizotinib, Fulvestrant, Trametinib (exact);
  Fluorouracil = Tahoe `5-Fluorouracil`, Azacitidine = Tahoe `5-Azacytidine` (same
  compounds, different naming). One unresolved: sci-Plex `AZ` is truncated upstream and
  could conceivably be Tahoe's `AZD-8055`; treat as a 7th until checked.
- **Doubly-exposed pairs (A549 AND one of the 6): 6 of 1,568 = 0.4%.** The eval grid is
  49 lines x 32 drugs = 1,568, which matches the `stack` row's pair count in Check 1.
  Drug-only exposure is 294 pairs (19%); line-only is 32 (2%).

**Action:** exclude the 6 doubly-exposed pairs from Check 1, or report with and without.
At 0.4% the disjointness assertion at `08_sciplex_prep.sbatch:15` is close enough to true
that leakage cannot explain a Check-1 result either way.

**Blockers found while checking, which DO matter:**

1. **The finetune never ran.** `/scratch/alpine/$USER/sciplex_finetune/` is empty; job
   30859593 died at import with
   `ImportError: /lib64/libstdc++.so.6: version GLIBCXX_3.4.29 not found` raised from
   `numpy/fft/_pocketfft_umath`. The `stack` conda env's numpy is built against a newer
   libstdc++ than the node provides. Fix by installing `libstdcxx-ng` from conda-forge into
   the env, or by putting `$CONDA_PREFIX/lib` ahead of `/lib64` in `LD_LIBRARY_PATH`.
   Environment bug, not a modelling problem.
2. **Gene-panel mismatch.** `sciplex_finetune.h5ad` is **2,000 genes**; `tahoe_context.h5ad`
   is **14,725**; Stack's panel is 15,012. The scPerturb sciplex3 release is pre-subset to
   2,000 HVGs. Aligning the generation head on 2,000 genes and then generating over 14,725
   is a train/test mismatch that would confound the result. Resolve before running: source
   full-gene sci-Plex counts from GEO GSE139944, or state the restriction explicitly.
3. **Raw counts unverified.** `08`'s log prints
   `counts=.X (VERIFY these are raw counts, not normalized)` and that verification never
   happened. Stack is a count model with an NB likelihood; normalized input breaks it.
4. **Upstream name truncation.** sci-Plex `pert_id` values are cut at the first whitespace
   (`AZ`, `GSK`, `ZM`, `Sodium`, `Aurora`, `Tie2`, `Valproic`, `Trichostatin`). This comes
   from scPerturb's `condition` column -- `build_sciplex_finetune.py:86` passes it through
   unmodified. 187 compounds observed against sci-Plex 3's published 188 suggests at most
   one collapse, but if two compounds share a first token they are merged into a single
   perturbation label. Check before training.

**Then:**

1. Fix (1)-(4) above.
2. Promote `09_stack_finetune.sbatch` from smoke test to real run: `--qos=gpu-testing` ->
   `gpu-normal`, `--max_epochs 1` -> real epochs, `--time 1:00:00` -> set from observed
   epoch time (`09:4,10,43`).
3. Generate on Tahoe with the drug-aligned checkpoint via 04's `CKPT`/`OUTDIR` overrides,
   score with `--generated-dir generated_sciplex` (`09:51-54`), excluding the 6 leaked pairs.

**Acceptance:** a drug-aligned checkpoint exists; Check 1 delta-Pearson for the drug-aligned
generator reported next to cytokine-aligned (0.012), additive (0.225) and the 0.30 raw /
0.46 Spearman-Brown ceiling, with and without the 6 leaked pairs.

### Phase 2 — modular refactor (local)

Implement the four registries above. Port the existing sources and metrics onto them with no
change in numbers.

**Acceptance:** re-running the Arm-2 ladder through the registries reproduces the published
table (`docs/tahoe_generation_results.md:78-88`), or every difference is explained and
justified in writing. A changed number is either a bug or a documented fix -- it is never
left unexplained.

### Phase 3 — calibrate the harness (Alpine amilan)

1. Wire `plant_interaction` into the Arm-2 driver as a synthetic-label mode.
2. Effect-size sweep on the real n=50 Tahoe design -> recovery curve (planted interaction ->
   recovered `interaction_rho`) and the MDE. Read +0.119 off that curve.
3. Run the sweep at **both** 5-fold and LOO, since the MDE does not transfer between schemes.
4. Repeat-CV over seeds to quantify fold-assignment variance on the existing +0.119.
5. Add the lineage one-hot as a ladder row.

Embarrassingly parallel over (effect x penalty x seed x scheme) -> SLURM array, CPU.

**Acceptance:** an MDE at n=50 for both schemes; +0.119 expressed as a fraction of
recoverable signal; a fold-variance interval on +0.119; lineage-one-hot interaction reported
alongside it.

**Kill criterion:** if the MDE at n=50 exceeds 0.119, the Arm-2 ladder could never have
detected that effect and every number in it is noise. Record it and proceed to Phase 4, where
n is larger.

### Phase 4 — re-anchor at n~500 (Alpine amilan)

1. Wire `load_gdsc2_sarcoma(sarcoma_only=False)` -- the full DepMap ∩ GDSC2 panel from local
   raw counts -- as an Arm-2 cohort.
2. Run the baseline ladder (expr, PCA, NMF, lineage, prior) at n~500 under the Phase-0 metric.
3. Validate against the known ceiling: GDSC2<->CTRP interaction 0.466 at 513 lines / 71 drugs.
4. Subset to the 50 Tahoe lines and check the metrics reproduce -- this separates "n" from
   "representation" and tests whether n=50 is why expr/PCA/NMF all sit at ~0.

**Acceptance:** ladder at n~500 with the CTRP ceiling on the same axes; the n=50 subset
table next to the n~500 table.

### Phase 5 — Stack embeddings at panel scale (Alpine aa100)

1. Build the full-panel Stack input from the same local DepMap raw counts
   (`scripts/prep_stack_input.py`).
2. Embed with the base checkpoint. `06_stack_embed.sbatch` does 50 lines in <=1h, so this is
   a chunked GPU array.
3. Feed into the Phase-4 ladder.

**Acceptance:** +0.119 restated as an n~500 measurement against the 0.466 ceiling, with the
lineage one-hot in the same table.

### Phase 6 — freeze, then full CV (Alpine amilan)

Runs once, on the settled configuration. This is the only phase that pays the ~100x.

1. Leave-one-line-out on the frozen representation set (`--folds 999`).
2. Port `bilinear_features` into the Arm-2 estimator registry and run leave-one-drug-out as
   the stress test.

**Acceptance:** LOO and LODO tables for the frozen configuration, reported against the CTRP
and PRISM ceilings.

## Alpine mechanics

`ralpine` is read-only by design -- `sbatch` and `scancel` are deliberately absent
(`scripts/alpine/ralpine:7-9`) because one spends an allocation and the other destroys running
work. The loop for every Alpine phase is therefore:

1. Claude authors and commits the sbatch script.
2. Lucas submits it.
3. Claude polls `ralpine sq` / `ralpine sacct` / `ralpine log` and pulls results with
   `ralpine pull`.

Requires an open ControlMaster socket -- `ssh alpine` once per session; `ralpine status`
reports it down otherwise.

Partitions: `amilan` (CPU) for Phases 3, 4, 6. `aa100` (GPU) for Phases 1 and 5. The existing
sbatch files all request `aa100`; the CPU phases must not inherit that.

## Out of scope

- The Soragni PDTO cohort and all Arm-1 analyses.
- Growing GDSC2 with Sanger raw counts.
- Any claim about non-cancer tissue.

## Known defect, not blocking

`scripts/gdsc_representation_increment.py --stack-gdsc` intersects to **0 rows**:
`stack_gdsc.csv` is keyed by the restored native loader (28 lines) while the script loads
CoderData ids (`:56-60`, 57 sarcoma lines; measured id overlap is zero). Either fix the id join or
retire the script in favour of the Arm-2 ladder. It is Arm 1 and therefore tabled.
