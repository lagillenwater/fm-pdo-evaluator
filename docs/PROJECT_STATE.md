# Transfer ladder — project state

**As of:** 2026-08-26, commit `f251b21`, branch `worktree-modular-harness-core`.

## What this document is, and what it replaces

This is the single current-state document: what is true right now, rung by rung, with every
claim traceable to a promoted artifact, a file:line, or a job id. It folds in the findings of a
7-dimension adversarial review run this session (63 sub-agents: one reviewer per dimension,
independent verifiers attempting to refute each finding before it counts) plus everything fixed
in response.

It replaces `docs/HANDOFF-2026-08-26.md` as the living status doc — that file is a point-in-time
handoff written mid-round and is now stale in most of its specifics (it predates every fix
below). It does **not** replace `docs/transfer_ladder_protocol.md` (the rungs, the six
invariants, the per-rung baseline/model/control lists — that is the standing design) or
`docs/decisions/2026-08-25-ladder-round.md` (D1-D6, why each was decided that way) — both stay
authoritative for *design intent*; this document is authoritative for *current implementation
status* and is expected to go stale again as soon as more work lands. Update it, don't let it
drift — that drift, repeated across a dozen documents instead of one, is the problem
`docs/PROJECT_SPEC.md` (invariants + spec index) and `docs/PROCESS.md` (how work actually gets
done, session to session) exist to fix. Read those two first if you haven't.

**Read this before trusting any number in a deck or a paper draft.** A number not listed here as
promoted is not evidence yet, per this project's own standard.

---

## 1. Per-rung status

### Rung 0 — replicate ceiling

**Promoted, correctly panelled.** `docs/results/rung0_delta_reproducibility.csv` [job
31676846]. Split-half median **0.109**, Spearman-Brown full-data **0.197**, both clearing their
diff-drug and same-drug nulls (p=0.0005).

This session fixed two defects: the ceiling was being computed on an unpinned top-2000-HVG
panel instead of rung 1's declared 14,121-gene panel (`scripts/alpine/delta_reproducibility.sbatch`
never passed `--panel-file`; commit `c1fa798`) — the earlier 0.299/0.461 figures were on that
wrong panel and are superseded, not just re-measured. And `p_vs_same_drug` used the
aggregate-vs-per-item statistic the file's own `p_vs_null` had already been corrected to avoid,
20 lines away (same commit).

**Open, not fixed this session:**
- Rung 0 aggregates by **median**; rung 1 aggregates by **mean** (invariant 3). This is why
  rung 1 does not get a "fraction of ceiling" in the ladder summary — it is marked **blocked**
  rather than computed wrong. Rung 0's `splithalf_mean_r` is already in the CSV; switching the
  headline to it (or switching rung 1 to a median) is the fix, whichever the team picks.
- Rung 0 scores Pearson; rung 2 scores Spearman (invariant 2). Same class of problem, blocks a
  rung-2-vs-rung-0 comparison specifically.

### Rung 1 — held-out Tahoe line, delta fidelity

**Promoted but incomplete against its own protocol row.** `docs/results/rung1_check1_fidelity.csv`
[job 31675161]. `knn`/`pca`/`nmf` (~0.28–0.32) clearly beat `stack_cytokine`/`stack_drug_aligned`
(~0.018/0.040) — directionally real, matched by the DE-fidelity table's `within_drug` p-values
(baselines p≈0.68–0.80, Stack/reference p≈0.005).

**Open, not fixed this session:**
- `rung1_plan.py` never builds `prior` (floor), `planted` (positive control), or `*_random`
  (noise controls) — `BASELINES = ("observed_delta", "knn", "pca", "nmf")`, no floor/positive-
  control row exists anywhere in the promoted CSV. Without them there is no way to confirm the
  rung-1 harness itself is working, as distinct from the baselines genuinely beating Stack.
- `scripts/audit_ladder.py`'s rung-1 provenance check reads `score_generation_eval.py` and
  `de_permutation_null.py` — **neither produced the promoted rung-1 result** (its own sidecar
  names `rung1_gather.py`). The audit's `controls_floor=True`/`controls_positive=True` for rung
  1 is not evidence of anything; it is a regex match against files the rung does not run.
- Baseline capacity (`--k`) is fixed at 10 for rung 1's pca/nmf/knn while rung 2 CV-selects it —
  an unfair comparison the 2026-08-21 capacity-fairness spec was written to remove.

### Rung 2 — cross-platform (map fit on L1000, tested on Tahoe)

**Dead at session start (crashed cluster array, 3 broken controls). Fixed and re-run this
session.** `docs/results/rung2_grid.csv`, `rung2_transfer_penalty.csv` [jobs
31677382/31677383/31677384, commit `4c23f60`].

Four defects fixed, all verified against a synthetic plan dir (including the exact
singleton-fold shape that crashed the cluster) before touching real data:
1. **The crash**: the shuffled control's line-relabeling matched the held-out line with
   probability `|fold|/50`, not 1 — essentially never at 5-fold. Replaced with
   `fmharness.deltas.shuffled_target_base`, a tested derangement helper.
2. **Positive control never fitted**: `planted` substituted its truth only at *scoring* time;
   every arm still trained on the real delta, so it had no way to recover a signal it was never
   shown (scored ~-0.005). Now built per-drug (independent random direction and drug-mean
   vector per drug — a single global direction had made every row correlate at exactly ±1,
   making "clear the null" impossible even for a perfect fit) and threaded into the actual fit
   target. Recovers cleanly now (~0.93-0.95 across arms).
3. **Negative control identical to the model**: `bulk_target`'s `shuffled` cell had no branch
   and fell through to the same call as `pca`.
4. **`bulk_target` leakage**: fit on the full Tahoe set, predicted the same lines' GDSC2 bulk
   profiles — every target's own delta was in its own fit. Unified with `in_platform` through
   one 5-fold helper (`_folded_predictions`) sharing the invariant-5 partition.

Also fixed: the mismatched-pair null was unstratified (same class of bug fixed in rung 0's
`b7b1d72`, reintroduced here) — now diff-drug-stratified, and its p-value goes through the same
`fmharness.statistics.bootstrap_aggregate_pvalue` helper (see §2). And `rung2_plan.py`'s L1000
training set previously included Tahoe's own 7 overlapping lines, so `cross_platform` saw up to
14% of its evaluation lines' own responses during fit — excluded now.

**First valid numbers**: every real baseline loses 0.17–0.53 of correlation moving from
Tahoe-fit to L1000-fit maps; cross-platform scores sit close to the shuffled (wrong-line)
control (e.g. pca 0.035 vs shuffled 0.033) — cross-platform transfer for the fitted baselines is
barely distinguishable from a scrambled baseline.

**Stack's own rung-2 arm** (decision D3: rebuild its generation context from L1000, query
baseline held at Tahoe) was built on 2026-08-25 but never scored — the generation output was
named by L1000's Broad `pert_id` (`BRD-K...`) while the scorer needed Tahoe's PubChem-CID
convention. Built the missing map from L1000's own `pert_info` table (no renaming of generated
files needed — `build_generated_deltas` already matches by filename stem against a mapping
dict) and scored it this session: `docs/results/rung2_l1000_context_generation.csv` [job
31678008]. Both checkpoints null (r=-0.001 cytokine, r=0.011 drug-aligned), matching rung 1's
Tahoe-context result within noise — Stack's failure is not a context/platform artifact.

**Open, not fixed this session:** `cmapPy` is a hard dependency of rung 2 (four files import
it) and is declared in no dependency file — not reproducible from a fresh `uv sync`.

### Rung 3 — GDSC2 viability

**Solid, unaffected by anything found this session except a figure-level fix.**
`docs/results/rung3_check2_grid.csv`, `rung3_declared_variants.csv` [job 31665927]. `base`
embedding under L2 penalty is the only representation clearing Bonferroni across 24 declared
variants (interaction 0.137, p=0.001, z_random=2.66) — ~30% of the 0.457 screen-agreement
ceiling. This was already computed correctly by `report_variants.py`'s Bonferroni check; the fix
this session was in the FIGURE (`plot_ladder_results.py`), which previously took an unfiltered
max over all 24+control rows for its headline number — now restricted to the significant,
non-control row.

**Open, not fixed this session:**
- `perdrug` and `global` are computed on non-residualized predictions while `interaction`
  residualizes out the per-drug mean (`evaluation.py`'s `score_predictions`). Proof it matters:
  `prior` — one constant feature, zero line information by construction — scores
  `perdrug=-0.285` while its `interaction` is exactly 0.000. A reader comparing the `perdrug`
  column across rows is reading fold-intercept structure, not per-drug ranking signal.
- `report_variants.py` cannot be run the obvious way — `ModuleNotFoundError` on its root-level
  shim import; needs `PYTHONPATH=.` or `python -m`.

### Rung 4 — organoid viability (embargoed, frozen holdout)

**Still blocked. This is the live item.** Decision D2's unfreeze condition ("once rung 3 is
promoted") is now met, but D2 also names two audit gaps to close first — `prov_params`,
`prov_panel` — and this session surfaced a chain of five separate defects, four fixed, the fifth
in progress when the session was interrupted:

1. **D1 not implemented**: the only rung-4 script effectively used L1000's drug coverage (the
   option D1 rejected), not GDSC2's (the option D1 chose), because nothing restricted the
   organoid target to GDSC2's screened compounds before scoring. **Fixed** — `design` is now
   restricted to `set(dg["drug"])` (GDSC2's own AUC design, already loaded for the training
   join) immediately after it's built. Commit `24c6240`.
2. **Invariant-5 violation**: two hand-rolled `{ln: i % n_folds ...}` fold maps instead of the
   shared `fold_assignment` helper. **Fixed**, same commit — behavior is unchanged at the
   `--folds 5` this script actually runs with; it only differed at the unused `--folds 1` edge
   case, where the shared helper's LOO degeneracy is correct.
3. **sbatch crash**: `scripts/alpine/12_sarcoma_organoids_2024_score.sbatch`'s `Resolved:` log
   line referenced `$GCTX`/`$GENDIR`/etc. *before* their default-value assignments, so any
   submission without explicit `--export` for every one of them hit `set -u` and died in the
   same second it started (job 31679368). **Fixed**, commit `82e0d6b` — moved the echo after
   the defaults.
4. **Stale raw-data path**: the 2026-08-25 rename migration (`e886685`) covered repo-root
   generated artifacts only; `data/raw/soragni/` was never renamed to
   `data/raw/sarcoma_organoids_2024/`, which the loader hardcodes with no override, so
   `load_sarcoma_organoids_2024()` failed closed with "raw manifest missing" (job 31679380).
   **Fixed on Alpine and locally** — extended `scripts/alpine/migrate_soragni_rename.sh` with
   the same `mv -vn` (never-overwrite, guarded) pattern, ran it both places. Commit `ad34b29`.
5. **Stale drug crosswalk — IN PROGRESS, NOT FIXED.** `data/static/drug_xref.parquet` still
   tags every organoid drug `source="soragni"` from before the rename; the loader filters for
   `source=="sarcoma_organoids_2024"` (`sarcoma_organoids_2024.py:213`). Zero rows match, so
   `build_sample_design(..., drug_key="pubchem_cid")` returns an empty design — "0 of 0 drugs"
   (job 31679480, after fixes 1–4 landed). The code that BUILDS the crosswalk
   (`scripts/build/build_drug_xref.py:242`) already writes the correct new label; only the
   *committed parquet* is stale — a data-artifact staleness problem, not a code bug, and the
   same *class* of problem this whole document exists to stop recurring. A rebuild
   (`--refresh`, hits PubChem for ~650 compounds at a rate-limited 0.25s/call, several minutes)
   was running when the session was interrupted; `data/static/drug_xref.parquet` and
   `manifest.json` are unchanged on disk. **Next action: rerun
   `PYTHONPATH=src uv run python scripts/build/build_drug_xref.py --refresh`, verify the
   `source` column reads `sarcoma_organoids_2024` for the organoid rows, commit (it's public
   reference data per `release_manifest.yaml`), push, pull to Alpine, resubmit
   `12_sarcoma_organoids_2024_score.sbatch`.**

**Still open after that unblocks it:** the two named audit gaps, `prov_params`/`prov_panel` —
the latter is real (no `common_gene_panel`/`assert_common_genes` call anywhere in
`score_viability_adapters.py`), not a blunt regex check, and was deliberately not attempted this
session (touches embargoed-data code, needs a real run to verify, more risk than this pass
should take under the time available). `docs/figures/sarcoma_organoids_2024_pathb_summary.png`
was removed this session (it was from a fully retired pipeline — adapters "szalai"/"xgboost"
that exist nowhere in current code); its replacement, `rung4_viability.png`, is wired up and
will render correctly the moment rung 4 produces `docs/results/rung4_viability.csv` — that exact
filename does not exist yet and the loader script may need a matching `--out-csv` name or a
small `plot_ladder_results.py` load-path adjustment once rung 4's actual output schema is known.

---

## 2. The p-value bug family (fixed this session, one shared cause)

Four independent scripts compared a reported AGGREGATE (a mean or median over many pairs)
against the spread of INDIVIDUAL null draws, instead of against the bootstrapped sampling
distribution of that same aggregate at the observed pair count. An aggregate's standard error is
roughly √n tighter than a single draw's, so this inflated every affected p by one to two orders
of magnitude:

- `delta_reproducibility.py`'s `p_vs_same_drug` (the `p_vs_null` beside it, 20 lines away, was
  already correct — commit `6a7a7cf` fixed one and not the other).
- `l1000_imputation_fidelity.py` — flips the headline: landmark genes go from p=0.2438 ("not
  established") to p=0.0005 (real).
- `l1000_tahoe_agreement_diagnosis.py`'s transform sweep — all seven transforms now clear their
  null (were p=0.13–0.28, "none clears it").
- `rung2_score_one.py` — see rung 2 above.

Fixed with one shared, tested implementation, `fmharness.statistics.bootstrap_aggregate_pvalue`,
so this class of bug can only be reintroduced by NOT using the helper, which is now the visible,
greppable anomaly rather than the invisible default. **`docs/l1000_imputation_fidelity.md` and
`docs/transfer_ladder_protocol.md` were corrected in place with explicit banners** rather than
silently rewritten — a reader who saved the old conclusion needs to see it was wrong and why.

**Not done: a systematic audit for any remaining occurrence of this pattern.** Four instances
were found because they were in the files this session's work touched; nothing has grepped the
whole repo for the shape (`np.sum(null_draws >= observed_aggregate) / null_draws.size` or
equivalent) to confirm there isn't a fifth.

---

## 3. Provenance / release-gate gaps (confirmed by the review, none fixed this session)

- **No promoted result carries a `LeakageProfile`.** `filter_leakage` exists and is correctly
  implemented, but is only ever called from `check1_registry_driver.py`/
  `check2_registry_driver.py` — neither is invoked by any current sbatch. Every promoted rung-1
  and rung-3 number comes from a parallel path (`rung1_plan/build_one/gather.py`,
  `check2_plan/score_one/gather.py`) that bypasses the registry abstraction entirely, including
  this guarantee. Concretely: nobody has checked whether the test cell lines at rungs 1/3 were
  in Stack's pretraining corpus. See `docs/PROJECT_SPEC.md`'s spec index (marked ORPHANED) for
  how this happened.
- **`check_release.py` scans column NAMES, not cell VALUES.** `SAMPLE_COLUMNS`
  (`scripts/check_release.py:46-49`) triggers row-level scanning only when a table has a column
  literally named `patient`/`line`/`cell_line`/etc. `docs/results/rung4_table_granularity.csv`
  has the schema `table, rows, columns, dose_or_replicate_columns, ...` — none of those names
  trip the check — but one row's `columns` cell is a semicolon-joined **string value** that
  contains `SARC0128_Tumor;SARC0129_Tumor;SARC0120_Organoids` as substring content, since that
  row describes `normalized_gene_counts.parquet`'s own column list. The file is committed today
  with those identifiers present. Confirmed still in the repo as of this document.
- **The gate is not enforced anywhere** — no pre-commit hook installed, no CI step, no test
  invokes it. `.github/workflows/ci.yml` runs ruff → ruff format → pyright → pytest, in that
  order, and a lint failure blocks the run before pytest ever executes — confirmed still true
  (251 ruff errors / 69 files needing reformat / 21 pyright errors on this branch as of the
  review), meaning CI has not actually reported a pass/fail on the test suite for this branch at
  any point.
- **`promote_result.py` records `HEAD` at promotion time**, not the commit the result was
  produced at, and does not check the working tree was clean when promoting. Two promoted
  sidecars this session already carry input/log paths under another machine's `/private/tmp/`
  scratchpad and `/Users/lucas/...` home — the sha256 still verifies, but the recorded location
  is unrecoverable from here.
- **`audit_ladder.py`'s `controls_*` columns are a regex over script source text**, not
  evidence a control ran. Confirmed correct in spirit (the file's own docstring says as much)
  but easy to over-read from the CSV alone — see the rung-1 example above, where it is actively
  misleading because it points at the wrong scripts entirely.

---

## 4. Confirmed correct — do not re-litigate

From the review's adversarial-verify pass (independent attempts to refute each finding; these
survived and several non-obvious ones were separately confirmed correct rather than merely
unchallenged):

- `fmharness.deltas.fold_assignment` genuinely is one shared, sorted, deterministic partition —
  `loo_baseline_source` holds out the whole fold as a group (not per-line), and rung 1/rung 3
  agree on the real data despite partitioning on different label spaces, verified by direct
  reproduction (not by construction).
- The (patient, drug) support restriction (`restrict_common_support`) is real and applied before
  scoring at both rung 1 and rung 3 — the exact bug class (`d9f94ec`) it was built to prevent
  does not currently recur there.
- `check_release.py` verifies embargo **per value** against the public cell-line registry, not
  per declaration, as claimed.
- Decision D6 (the `additive`→`observed_delta` rename with a back-compat alias) is fully and
  correctly implemented, including its stated "reverse by" path.
- Decision D5's staleness map (`SUPERSEDED_BEFORE` in `audit_ladder.py`) is real and precise —
  flags exactly the one result it should, not more, not fewer.
- The 2026-08-21 capacity-fairness and common-support specs are fully implemented at the six
  sites they were meant to reach (rung 3's fixed-readout and penalized-grid paths, both).
- `rung1_gather.py`/`check2_gather.py` both genuinely refuse to score an incomplete source set,
  for the correct reason (missing a source changes every OTHER source's number too).

---

## 5. Where things live

- **Results**: `docs/results/*.csv` + matching `.provenance.json` sidecar (job id, git sha,
  input hashes). No sidecar, no evidence — this project's own standing rule.
- **Figures**: `docs/figures/*.png`, rebuilt this session (`scripts/plot_ladder_results.py`) —
  see the commit messages on `8a5badb`/`f251b21` for what was wrong with the previous set (two
  panels were blank placeholders never regenerated after promotion; the ladder summary showed
  one rung; one figure was from a retired pipeline).
- **Audit**: `docs/results/ladder_audit.csv` (per-rung control/provenance checks — read §3's
  caveat about what these columns do and don't prove), `docs/results/promoted_provenance.csv`
  (staleness).
- **Design**: `docs/transfer_ladder_protocol.md`. **Decisions**:
  `docs/decisions/2026-08-25-ladder-round.md`. **This document**: current implementation state,
  update it as things change rather than writing a new dated handoff.
