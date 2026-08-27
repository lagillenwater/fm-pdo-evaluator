# Decisions taken without asking — ladder round, 2026-08-25

Every entry is a question I would normally have brought to Lucas. Standing instruction for this
round: document the question, the options, and proceed with the recommendation. Each is
reversible; the "how to reverse" line says what to change.

---

## D1. Which cohort supplies rung 4's drug axis

**Question.** The sarcoma organoid panel has 34 drugs. L1000 covers 25, GDSC2 covers 21, both
cover 19 [job 31660822]. Which defines rung 4's drug axis?

**Options.** (a) L1000, for maximum coverage — 25 drugs. (b) GDSC2, for 21. (c) The 19 covered
by both.

**Chosen: (b) GDSC2.** Rung 4 has no ceiling of its own — verified, no dose or replicate column
in any of the seven tables [job 31663218] — so rung 3 is its *only* reference frame and every
rung-4 statement is a ratio against it. GDSC2 already supplies rung 3's labels, so taking rung
4's drug axis from GDSC2 keeps both rungs on the same drugs. A shared drug axis is worth more
than four extra drugs when the comparison is the entire interpretation. Option (c) stays
available as a controlled sub-analysis.

**Reverse by:** pointing rung 4's drug selection at the L1000 pert map instead; the coverage
numbers are already measured.

---

## D2. Freezing rung 4 rather than running it now

**Question.** Rung 4 is buildable. Run it in this round, or hold it?

**Options.** (a) Run it now for a complete ladder. (b) Freeze until rungs 0–3 are locked.

**Chosen: (b) freeze.** Rung 4 is the embargoed holdout and nothing has been fit to it —
`check_release.py` now enforces that per value against a public cell-line registry. Since it has
no ceiling, a rung-4 number is only interpretable against a finished rung 3; running it first
spends the holdout before the thing that contextualises it exists. The alignment of the data
policy with the evaluation design is also a defensible claim in a write-up, and running early
forfeits it.

**Reverse by:** unfreezing once rung 3 is promoted — but close rung 4's two audit gaps first
(`prov_params`, `prov_panel`), which are recorded and not yet fixed.

---

## D3. How Stack gets a cross-platform arm at rung 2

**Question.** Stack is not fitted, so it has no training set to swap and would sit in rung 2 as
a constant. What is the matched cross-platform arm for it?

**Options.** (a) Leave it out. (b) Swap its query baseline to L1000. (c) Rebuild its CONTEXT
from L1000 and regenerate.

**Chosen: (c).** Initially I chose (a), and Lucas rejected it — correctly, since it means
learning nothing about Stack at the rung that matters. I then proposed (b), which is wrong in a
subtler way: it shifts what the model is *conditioned on*, while the baselines' arm shifts where
their *map was learned*. Stack's context is its in-context learning corpus — the examples it
reads a drug's effect from — so it is the true analogue of training data. (c) holds the query
line at Tahoe and swaps only the corpus.

**Known limitation, carried into the write-up:** only 15 of Tahoe's 33 drugs matched L1000
[job 31664466], so Stack's cross-platform arm is a 14-drug comparison against the baselines'
wider one, and the two must not be tabulated as if equally powered.

**Reverse by:** dropping the `l1000_context_by_drug` generation arms; the Tahoe-context arms are
unaffected.

---

## D4. No cross-validation on rung 2's cross-platform arm

**Question.** The in-platform arm is 5-fold. Should the cross-platform arm resample too?

**Options.** (a) No CV — train on all of L1000, predict all Tahoe lines. (b) Bootstrap over
target lines to match the in-platform arm's variance.

**Chosen: (a), with the limitation stated.** There is no leakage to guard against: the training
data is an entirely separate platform. But the transfer penalty is `cross_platform −
in_platform`, and subtracting a single point estimate from a fold-averaged one leaves a
variance mismatch inside the quantity the rung exists to measure.

**Recommendation for the next round:** add the bootstrap. It is the one place in the ladder
where I know the comparison is imperfect and chose not to fix it in this pass, because the
point estimate is still informative and the fix is additive rather than corrective.

---

## D5. Staleness is matched per fix to the scripts that fix touched

**Question.** A result can carry perfect provenance and still be invalid because a defect has
since been fixed. How wide should the invalidation be?

**Options.** (a) Flag every result older than the earliest fix. (b) Match each fix to the
scripts it touched.

**Chosen: (b).** (a) marked all 15 promoted results superseded, including gene-overlap tables
that never used the panel — noise that buries the results genuinely at risk. Under (b) exactly
one is flagged: `check2_grid_5fold_corrected.csv`, job 31655278, produced before the panel was
wired and before the positive control was restored.

**Reverse by:** widening `SUPERSEDED_BEFORE` in `scripts/audit_ladder.py`; the fix-to-script map
is explicit there.

---

## D6. `additive` renamed to `observed_delta`, old name still dispatches

**Question.** Renaming a source label changes every future result table. Break the old name?

**Options.** (a) Hard rename. (b) Rename, keep the old string working.

**Chosen: (b).** `loo_baseline_source` dispatches on this string, so a hard rename would send an
old call site silently down the `knn` branch rather than failing. A test asserts both names
produce identical output, and `model_matrix.yaml` records the alias so `check_matrix` still
resolves tables written under either.

**Reverse by:** deleting the alias once no caller uses it — the test will fail loudly, which is
the point.
