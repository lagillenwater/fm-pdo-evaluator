# Decision — revert the data layer off CoderData, back to native raw-artifact loaders

**Written retroactively, 2026-08-26**, closing the gap `docs/PROJECT_SPEC.md` invariant 10
names as the starkest example of an undocumented architectural reversal in this repo: this
decision was made and executed on 2026-06-16 (`93dc76f`, merged as PR #6
"restore-custom-loaders") with no decision record anywhere — the only trace was git log and one
person's memory. Reconstructed from the revert commit's own message, which is thorough; this
file exists so the *decision* is discoverable without already knowing to look at git log for a
two-month-old commit.

**Question.** The data layer had been swapped onto CoderData (`1bfb922`, 2026-05-29) three
weeks earlier. Keep it, or revert to loaders that read the raw per-dataset artifacts directly?

**What was found.** CoderData fed GDSC2 as gene-length-normalized TPM but the Soragni/sarcoma
organoid cohort as CPM — an asymmetry invisible unless you specifically check each dataset's
normalization, and one that matters here more than in most pipelines: Stack is a single-cell
count model (log1p input, negative-binomial decoder) that expects length-free, count-derived
input. Feeding it TPM for one cohort and CPM for another confounds the model's own normalization
assumption with a real biological or platform effect — exactly the kind of silent mismatch this
project's invariant 3 (common support / consistent aggregation) exists to catch, just one layer
lower, in the data loader instead of the scoring step.

**Chosen: revert.** Restore loaders that read raw artifacts directly, both put on the same
length-free normalization: GDSC2 from DepMap raw counts via DESeq2 median-of-ratios (raw counts
kept in `layers['raw_counts']`); the organoid cohort's deposited matrix is already CPM
(length-free, verified summing to 1e6) — CoderData's normalization label for it was corrected
from a false `'tpm'` to `'cpm'` in the same pass, and its response metric corrected from
CoderData's mislabeled `'auc'` to the actual `'viability'`. `load_tranche` adapts both native
bundles to the shared `build_sample_design` contract, so nothing downstream needed to change.

**Consequence for reading older specs.** `docs/tasks/modular-harness-core/design.md`
and other specs written after this revert already assume the native loaders and are accurate.
The initial CoderData swap (`1bfb922`) predates every design spec in this repo by five to ten
weeks and left no spec of its own either — there is nothing to correct in retrospect there,
only this entry recording that it was tried and reverted.

**Reverse by:** re-adopting CoderData would require first fixing its GDSC2/Soragni normalization
asymmetry upstream (or normalizing both to a common length-free basis after loading from
CoderData, inside `load_tranche`) — reverting back without that fix reintroduces the exact
confound this decision removed.
