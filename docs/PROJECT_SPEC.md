# Project spec — fm-pdo-evaluator

This is the trunk. It answers three questions that no single existing document answered before
it, which is why bugs got re-solved: **what must every piece of analysis in this repo respect,
no matter which task produced it; what document is currently authoritative for a given area; and
what happened to the documents that used to be authoritative and no longer are.**

It changes rarely — invariants get amended deliberately, the index gets a new row when a spec
lands, entries get reclassified when superseded. It is not where you look for "what's true right
now" (that's `docs/PROJECT_STATE.md`) or "what's the current experimental design" (that's
`docs/transfer_ladder_protocol.md`, itself indexed below as the current active spec). It is where
you look before starting new design work, to find out what's already been decided and where.

## Mission

Does a foundation model's cell-line drug-response prediction transfer to patient-derived
organoids? Two scores matter: an in-silico score now, a prospective score later, and the gap
between them is the number the project exists to produce. The evaluation has gone through three
framings as understanding of the problem improved — **Check 1 / Check 2** (2026-08-07 onward,
registry-driven, per-model correctness checks) → **Path A / Path B** (cell-line vs organoid
substrate split) → **the transfer ladder** (2026-08-25 onward, rungs 0-4, one distribution shift
added per rung). These are not three different projects; they are the same question, reframed
each time the previous framing turned out not to isolate *why* a prediction failed. The mapping:

| Old name | Rung | What it tests |
|---|---|---|
| (new) | 0 | Tahoe replicate ceiling — is the target itself reproducible |
| Check 1 / Check 1b | 1 | held-out Tahoe cell line, delta fidelity |
| (new) | 2 | cross-platform: map fit on L1000, tested on Tahoe |
| Check 2 | 3 | GDSC2 viability (cell line) |
| Path B (Soragni/sarcoma organoids) | 4 | organoid viability — embargoed, frozen holdout |

A spec written before 2026-08-25 that says "Check 1" means rung 1; "Check 2" means rung 3;
"Path B" or "Soragni" means rung 4. There is no rung 0 or rung 2 equivalent in the old framing —
both are new diagnostic rungs added when Check 1/Check 2 alone couldn't distinguish "the model
failed" from "the ceiling was never established" or "the platform shifted."

## Invariants — permanent law

Every script that computes or reports a number must satisfy these. They generalize
`transfer_ladder_protocol.md`'s six rung-scoped invariants to the whole project, plus rules this
session's review found violated in ways the ladder's own list didn't cover. **Amend this section
deliberately when a genuine exception is needed — silently violating one is how the Pearson/
Spearman split and the four independent p-value bugs happened.**

1. **One shared partition for anything resembling cross-validation.**
   `fmharness.deltas.fold_assignment` — sorted, deterministic, degenerates to leave-one-out at
   `n_folds >= len(lines)`. Never a hand-rolled `{ln: i % n_folds ...}`, even if it looks
   equivalent — it silently stops being equivalent the moment two call sites iterate lines in a
   different order, and nothing will tell you until a cross-comparison quietly breaks.
2. **One statistic, one aggregation, applied identically wherever two numbers are compared.**
   If rung A is Pearson-median and rung B is Spearman-mean, `A/B` or `A - B` is not a
   measurement, it's noise with a sign. State the metric and the aggregation per comparison
   *once*, upstream of every script that reports it, not per-script.
3. **Restrict to common support before comparing.** Every arm being compared in one table must
   be scored on the same (unit, condition) pairs and the same feature panel it and every other
   arm can natively cover — never each arm against its own native intersection with the target.
   (`restrict_common_support`, `common_gene_panel` + `assert_common_genes`.)
4. **Every reported statistic ships with a known-answer test that imports the real function.**
   Plant a signal, require it recovered; plant nothing, require null. The test must call the
   production code, not reimplement its logic inline — a reimplementation passes even when the
   real function is broken, which already happened once this project (the rung-2 shuffled-
   control "fix" that still crashed the cluster while its own unit test stayed green).
5. **An aggregate's null must be resampled to that aggregate, not compared against individual
   draws.** Use `fmharness.statistics.bootstrap_aggregate_pvalue`. Comparing a mean/median over
   n items against single null draws inflates p by roughly √n — this exact bug independently
   recurred in four scripts in one week before it got a shared, tested implementation.
6. **Every comparison table needs a floor and, where feasible, a positive control.** A table of
   real sources with no control that must fail and none that must succeed cannot distinguish
   "the effect is real" from "this pipeline can't fit anything" — this is not optional
   decoration, it is what makes a null result mean something.
7. **Model/representation capacity must be tuned the same way across everything compared in one
   table.** CV-select it (component counts, penalty strength) or fix it identically for every
   arm — never one arm tuned and another hardcoded, even if the hardcoding predates the tuned
   arm and "still basically works."
8. **Every promoted result carries a provenance sidecar** (git sha, job id, resolved args,
   input hashes, log hash) written from a clean tree at the commit that produced it. No sidecar,
   not evidence — this project's own standing rule, already true, restated here so it's provable
   from one document instead of tribal knowledge.
9. **Embargo is enforced per value, fail-closed, on cell content — not per column name and not
   per file path.** A column literally named `line` is not the only place a patient identifier
   can leak; a meta-table describing another table's *schema* can leak the same identifiers as
   free-text cell content. `data/release_manifest.yaml`'s three tiers apply to every artifact by
   default-embargoed; classify explicitly or it stays embargoed.
10. **A reversal of architecture, method, or data source gets a decision entry, not just a
   commit.** If work this session or any future session reverts something a spec/plan
   established (a data layer, a scoring path, a control), write one paragraph in
   `docs/decisions/` saying what changed and why — even a terse one, even written months late.
   The CoderData → custom-loaders reversal (a full PR-level architectural revert,
   `1bfb922`→`93dc76f`, merged as "restore-custom-loaders", 2026-06-16) had *zero* trace in any
   doc in this repository for over two months — the only record was git log and one person's
   memory — until this rule was written; see `docs/decisions/2026-06-16-revert-coderdata-loaders.md`
   for the now-recovered entry. It remains the sharpest example of the process problem this file
   exists to fix, and it must not recur for the next one.

## Spec index

Every design/plan/decision document in the repo, classified. **ACTIVE** = currently authoritative
for its area, safe to build on. **SUPERSEDED** = superseded by a named later document; the old
one's header should carry a one-line pointer (add one if it doesn't yet). **ORPHANED** =
implemented as designed, but the pipeline that was supposed to use it has been bypassed by a
newer parallel path, and the spec's guarantee is not currently enforced anywhere — worse than
superseded, because nothing marks it as not-in-effect. **HISTORICAL** = describes work that is
done and stayed done; keep for context, nothing to reconcile.

| Doc | Date | Status | Note |
|---|---|---|---|
| `docs/transfer_ladder_protocol.md` | 2026-08-25 | **ACTIVE** | Current experimental design — the rungs, the six ladder-scoped invariants, per-rung baseline/model/control lists. Its Invariant 2 ("Spearman per (line, drug) over the panel") is itself **wrong** — it asserts something neither the code nor the two older specs describing the same code ever intended (rungs 0/1 are Pearson by design; only rung 2 is Spearman). Needs a correction, the same way any other doc would. |
| `docs/decisions/2026-08-25-ladder-round.md` | 2026-08-25 | **ACTIVE, one entry stale** | D1-D6. D1 (GDSC2 drug axis) was decided but not implemented until 2026-08-26's `24c6240` — a full day where the doc read as settled while the only rung-4 script did the opposite. D2's unfreeze precondition (prov_params/prov_panel closed) is still unmet. Neither is reflected in the doc; see `docs/PROJECT_STATE.md` for current status. |
| `docs/superpowers/specs/2026-08-07-modular-harness-core-design.md` + plan | 2026-08-07 | **ORPHANED** | Library layer (Encoder/Generator protocols, registries, leakage filtering, CV scheme) is fully implemented. Its acceptance-target driver, `check1_registry_driver.py`, is not invoked by any current sbatch script — every promoted rung-1 result comes from a parallel `rung1_plan/build_one/gather.py` path written 2026-08-25/26 that calls the underlying library directly and does **not** run `filter_leakage`. The spec's stated leakage guarantee is therefore not in effect for any promoted number, and nothing documents that this happened. |
| `docs/superpowers/specs/2026-08-11-stack-drug-alignment-and-check1-design.md` + plan | 2026-08-11 | **ORPHANED** | Same mechanism as above — `check1_registry_driver.py` exists, correctly implements the spec, is bypassed in production. |
| `docs/superpowers/specs/2026-08-13-check2-leakage-aware-drug-aligned-design.md` + plan | 2026-08-13 | **ORPHANED** | Same mechanism — `check2_registry_driver.py` bypassed by `check2_plan/score_one/gather.py`. |
| `docs/superpowers/specs/2026-08-13-check2-leakage-aware-drug-aligned.md` (no `-design` suffix) | 2026-08-13 | **SUPERSEDED**, self-mislabeled | A handoff doc, still self-labeled `Status: not started` despite the work being designed, planned, and now superseded via the orphaning above. Carries a load-bearing inline correction (the drug-aligned checkpoint actually used is `epoch=5/val_loss=6.1078`, not the `epoch=4/5.0847` every other doc names) that nothing indexes — `data/model_matrix.yaml` is currently the only pointer to it. |
| `docs/superpowers/specs/2026-08-18-stack-faithful-generation-and-de-metrics-design.md` + plan | 2026-08-18 | **ACTIVE (mostly done)** | vanilla→`--mode mdm` fix and DE metrics both implemented and consistently applied in every current generation sbatch. DE metrics live in a separate scoring function rather than merged into the primary delta table as designed — `rung1_check1_fidelity.csv` carries no DE columns; report the two tables together until that's reconciled. |
| `docs/superpowers/specs/2026-08-21-cross-check-fairness-and-capacity-design.md` (no plan file) | 2026-08-21 | **ACTIVE, partially threaded through** | `restrict_common_support` and matched-width random-feature controls fully implemented at every intended site. The capacity-fairness fix (CV-select k/n_components instead of hardcoding) is implemented in the *library* but the driver scripts for rungs 1 and 3 still pass a hardcoded `--k 10`, disabling it exactly where invariant 7 above says it must apply. |
| `docs/HANDOFF-2026-08-26.md` | 2026-08-26 | **HISTORICAL** | Point-in-time handoff, superseded as the living status doc by `docs/PROJECT_STATE.md`. Keep as a record of what the round looked like mid-flight; do not update further. |
| `docs/adapter_contract.md`, `docs/models.md`, `docs/environment.md` | various | **ACTIVE** | Not re-audited this pass; no conflict found against current code in the areas this session touched. |

**Every plan file has zero checked boxes**, including ones fully implemented — the checkbox
mechanism is not a reliable signal of status in this repo and should not be trusted at face
value; this index, not the plan's own checkboxes, is the status source.

## Known drift points — read before you touch these areas

Concrete case studies, kept here permanently (not in PROJECT_STATE.md, which only tracks current
open/fixed status) because the *lesson* stays useful after the specific confusion is resolved.

- **A newer, more authoritative-looking document can itself be the source of drift.**
  `transfer_ladder_protocol.md` is the newest, most-referenced design doc in the repo, and its
  Invariant 2 is the one that's wrong — it was written without checking the two older specs that
  correctly describe the code it's supposedly formalizing. Recency is not the same as accuracy;
  a new spec must check existing specs and code, not just state intent.
- **Renaming a concept without a forward pointer leaves every prior document silently
  ambiguous.** `additive` → `observed_delta` (D6) and `soragni` → `sarcoma_organoids_2024`
  (`a6c8976`) both have a working *code* alias/rename, and zero *documentation* trail pointing
  old-name readers to what changed. A rename is a two-line fix in the doc that used the old
  name; do it at rename time, not retroactively.
- **A parallel implementation that bypasses an abstraction silently drops that abstraction's
  guarantees.** The registry-driver orphaning (spec index above) is the concrete instance: three
  specs' worth of leakage-filtering design is not running on any promoted number, and discovering
  that took a targeted review, not a doc anyone could just read.
- **An architectural reversal with no decision entry is invisible to everyone but git log.**
  The CoderData → custom-loaders revert sat undocumented for over two months. See invariant 10
  and `docs/decisions/2026-06-16-revert-coderdata-loaders.md`.

## Process for new specs

1. **Before writing a new design spec**, read this file's invariant list and spec index. If the
   new work overlaps an ACTIVE spec, extend it or explicitly supersede it — don't start a third
   parallel document on the same subject.
2. **If the new work needs an exception to an invariant**, that exception is written into *this*
   file (a numbered sub-point under the invariant, with the reason), not left implicit in the new
   spec alone — otherwise the next task that touches the same code has no way to know the
   exception exists and will "fix" it back, or worse, not know it was ever a deliberate choice.
3. **When a new spec/decision supersedes or reverses an older one**, do both of:
   - Add a one-line dated banner at the top of the *old* document pointing to the new one (the
     pattern already used ad hoc in `docs/l1000_imputation_fidelity.md`'s correction banner and
     the 2026-08-13 handoff's inline status note — make it the standard, not the exception).
   - Update this file's spec index row for the old document.
4. **If the new work changes a currently-reported number**, update `docs/PROJECT_STATE.md` in
   the same change — a spec describing new intent and a state doc still showing the old number
   is exactly the drift this structure exists to prevent.
5. **A new per-task spec's own header should name which invariants it relies on**, so a future
   reader doesn't have to reverse-engineer which rules were in scope when it was written.
