# Cross-check fairness and capacity design

2026-08-21

## Background

`scripts/score_viability_adapters.py` (Path B) had a real bug, fixed today (commit
`d9f94ec`): every delta source (`additive`/`pca`/`nmf`/`stack`) was independently inner-joined
against the real Soragni AUC design, so each source ended up scored on a *different*
(patient, drug) support (`stack` n=202 vs `additive`/`pca`/`nmf` n=150 in one run; n=37 vs 150
in another). Fixed via `fmharness.deltas.restrict_common_support`, which intersects every
source down to the pairs all sources share and that carry a real label, before scoring.

A follow-up audit (`superpowers:dispatching-parallel-agents`-style workflow, find + adversarial
verify) checked whether the same bug class exists in Check 1 and Check 2 -- the Tahoe/GDSC2
generation-eval pipelines whose numbers are already reported (`docs/tahoe_generation_results.md`)
-- and in the other Soragni baseline scripts. It found and verified 6 more instances, 2 of them
already visibly manifested in published results tables (`docs/tahoe_generation_results.md`
reports the mismatched pair counts in prose rather than fixing them in code).

Separately, extending the readout adapters to compare against Stack's `base`/`aligned`
embedding checkpoints (mirroring Check 2's own base-vs-aligned comparison) raised a second,
distinct fairness question: PCA/NMF/kNN's number of latent variables is capped by the small
eval cohort's sample size, while a foundation model's embedding width comes from its
pretraining corpus. Forcing them to the same raw dimensionality is neither meaningful (PCA/NMF
are non-identifiable past ~n-1 components) nor fair in the other direction (truncating Stack's
embedding down to PCA's k throws away exactly what the FM contributes). This design covers both
problems together since they'll be touched in the same files.

## Part A -- data-support fairness (6 confirmed findings)

Reuse/generalize `fmharness.deltas.restrict_common_support` (`sources: dict[str, tuple[delta,
key]]`, `design` -> row-filtered sources on the shared, labeled (patient/line, drug) support)
rather than writing a parallel mechanism per site.

1. **Check 1** (`fmharness.evaluation.score_delta_sources`/`delta_fidelity`). Callers:
   `scripts/score_generation_eval.py`, `scripts/check1_registry_driver.py`. Fix: call
   `restrict_common_support(sources, real_key)` right before `score_delta_sources(...)` in both
   scripts. Delete `check1_registry_driver.py`'s misleading "no separate pre-filter needed"
   comment (it only argues leakage-filtering is enough, which is true for that but not for
   cross-source coverage parity).

2. **Check 2 part (a)** (fixed-signature readouts, `score_check2`, `check2.py` lines ~174-192).
   Fix: `sources = restrict_common_support(sources, design)` right before the fixed-readout
   loop, inside `score_check2` itself (so every caller gets it for free). Print the native-vs-
   common diagnostic the same way `score_viability_adapters.py` does.

3. **Check 2 part (b)** (representation-controlled penalized grid, `penalized_preds`). Different
   data shape than (1)/(2) -- representations are `{drug: DataFrame[line x genes]}` dicts or
   drug-independent callables, not (delta, key) row-pairs, so `restrict_common_support` can't be
   reused verbatim. New function `restrict_representation_support` (`check2.py`, next to
   `penalized_preds`): materializes every representation to `{drug: frame}` over the drugs
   `design_target` covers, then per drug intersects every representation's line index with the
   design-labeled patients for that drug, drops the drug if any representation lacks it or the
   intersection is empty, and returns every representation restricted to that common per-drug
   line set. Call it once in `score_check2` right before the `for repr_name, feat in
   representations.items()` loop, replacing the raw dict/callable mix with the restricted,
   uniformly-`dict`-shaped output.

4. **`scripts/predict_expression_baselines.py`**. Fix: `methods = restrict_common_support
   (methods, design)` before the scoring loop; fix the printed `n` to reflect the *restricted*
   key length (currently the misleading pre-merge broadcast size).

5. **`scripts/baselines_soragni.py`**. Fix: compute `shared` (the GDSC2-screened intersection)
   *before* building `rows`, and filter drug-mean/l1000 rows to `shared` too, not just pca/nmf --
   so every row in the printed table shares one drug support instead of pca/nmf alone being
   narrowed.

6. **`scripts/biomarker_anchored.py`**. Fix: restrict the biomarker (`bm_all`) and global-model
   (`glob`) predictions to their shared (patient, drug) support before the head-to-head
   comparison, and print `n` for both sides (today prints neither).

Each fix gets a test mirroring `tests/test_deltas.py`'s two `restrict_common_support` tests
(deliberately-different native coverage -> only the shared, labeled pairs survive; empty
intersection -> raises).

## Part B -- capacity/latent-variable fairness

**Principle (no dimensionality truncation):** a foundation-model embedding's width reflects its
pretraining corpus, not the small eval cohort. Truncating it down to match PCA/NMF's component
count throws away real information; inflating PCA/NMF up to the FM's width fits noise (PCA/NMF
are non-identifiable past ~n_train-1 components). Resolve this by comparing *effective capacity
after CV-tuned regularization*, not raw dimension counts -- the same principle `make_penalty`
(RidgeCV/LassoCV/ElasticNetCV) already applies to the penalty strength. Ridge is well-posed for
p >> n; `penalized_preds`/`load_line_matrix` already feed the *full, untruncated* embedding into
`make_penalty` with no truncation -- confirmed, no code change needed there.

What *is* still a fixed, uncontrolled hardcode today, needing the same CV-tuning treatment:

7. **PCA/NMF's `n_components`** (`fmharness.deltas.build_learned_deltas`, default `k=20`,
   shared by Check 1's `loo_baseline_source` and Check 2's delta sources) and **kNN's `k`**
   (`build_knn_deltas`, default `k=10`), plus the scattered `args.n_components` / hardcoded-10
   defaults in `predict_expression_baselines.py`, `baselines_soragni.py`, `biomarker_anchored.py`.
   Fix: inner-CV-select `n_components`/`k` per representation/fold from a small shared candidate
   grid (e.g. `{2, 5, 10, 15, 20}`, capped below the training-fold's own sample size so a
   candidate is never invalid), scored by CV-averaged predictive error on the training fold only
   (`sklearn.model_selection.cross_val_score`, small inner fold count) -- never against the held-
   out line/patient, so no leakage into the outer LOO evaluation. Also replace
   `build_learned_deltas`'s fixed `Ridge(alpha=alpha)` correction-step with `make_penalty("l2")`
   (CV-tuned), the same inconsistency class the user flagged ("tuning parameters of the various
   ridge models").

8. **Matched-width random-feature negative control** for high-dimensional FM embeddings (Stack
   now, scFoundation later): for each FM-embedding representation in the penalized grid, also
   fit/score a same-shape (`n_samples x n_features`) i.i.d. Gaussian-feature representation
   through the identical CV-tuned model. Report both rows; the FM row must clear the random-
   control row by a real margin to attribute any apparent win to learned structure rather than
   raw parameter count. Mirrors `fmharness.signatures.score_signatures`'s existing
   `rnd_p95`/`p_vs_random` same-size-random-gene-set control -- same idea, generalized from gene-
   set size to embedding width, seeded and reproducible. Lives in `check2.py` (used by part b)
   and `score_viability_adapters.py`'s `--stack-emb` path.

9. **Document, don't change**: Stack's embedding already flows into `make_penalty` untruncated
   (point 7's `load_line_matrix` confirmation) -- record this explicitly in `check2.py`'s and
   `score_viability_adapters.py`'s module docstrings as the resolution to the capacity-parity
   question, so a future reader doesn't reintroduce a truncation step by mistake.

## Files touched

- `src/fmharness/deltas.py` -- `build_learned_deltas` (CV-tuned `n_components` + `make_penalty`
  correction step), `build_knn_deltas` (CV-tuned `k`).
- `src/fmharness/check2.py` -- `restrict_representation_support` (new), wire into
  `score_check2`; matched-width random-feature control in the penalized grid.
- `src/fmharness/evaluation.py` -- no logic change; `score_delta_sources`'s callers gain the
  `restrict_common_support` call, not the function itself.
- `scripts/score_generation_eval.py`, `scripts/check1_registry_driver.py`,
  `scripts/check2_registry_driver.py` -- wire in `restrict_common_support` where not already
  inherited from `score_check2`'s own fix.
- `scripts/predict_expression_baselines.py`, `scripts/baselines_soragni.py`,
  `scripts/biomarker_anchored.py` -- per-script fixes above.
- `scripts/score_viability_adapters.py` -- matched-width random control for `--stack-emb`;
  docstring note (point 9).
- Tests: `tests/test_deltas.py` (CV-tuned n_components/k), `tests/test_check2.py`
  (`restrict_representation_support`, random-feature control), plus one regression test per
  fixed script pattern where the script has an existing test file.

## Sequencing

1. Library changes first (`deltas.py`, `check2.py`), TDD, full local test suite green.
2. Wire into each of the 6 script call sites, one at a time, `py_compile` + pyright per file.
3. Commit (checked with the user first, per standing convention) and push (pre-authorized).
4. Alpine: pull, rerun the affected Check 1 / Check 2 / Soragni-baseline jobs, compare before/
   after -- the two already-published discrepancies (`docs/tahoe_generation_results.md`) should
   either close or the doc's caveat should be replaced with corrected numbers.
5. Update `docs/tahoe_generation_results.md` and `docs/soragni_pathb_results.csv`/figure once
   corrected numbers land, same verify-before-reporting discipline as today's Path B correction.

## Out of scope (explicitly deferred)

- Repo-wide sweep of every historical/exploratory script (`per_patient_eval.py`,
  `transfer_pharmaformer_lite.py`, `harness_core_demo.py`, `plot_global_rho.py`,
  `plot_pearson.py`) -- the audit's "sweep" phase checked these and found them already fair
  (shared support computed once, or fail-loud `.loc` indexing instead of silent per-source
  joins); not part of Check 1/2 or the Soragni Path-B family this design covers.
- scFoundation integration itself -- point 8's random-width control is designed to generalize to
  it, but adding the actual scFoundation embedding pipeline is a separate future task.
