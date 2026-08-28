# Rung 0 — review record

**As of** 2026-08-28.

How this branch was reviewed and what came of it. Every task went through a two-stage gate — an
implementation-scoped review (spec compliance + code quality) with fix rounds re-verified by a
scoped re-review — followed by one whole-branch final review before the pull request. Findings
below are grouped by where they were found; each carries its disposition. Numbers live in
[`verification.md`](verification.md) and the promoted artifact, not here.

## Per-task reviews

- **Dependencies & design amendment** — 2 findings: the dependency manifest's comment became
  false when packages landed ahead of their importing code (fixed: comment now covers code
  landing on the task's branch); `docs/DATA.md` edits broke the file's line-wrap convention
  (fixed, and a stacked double-parenthetical smoothed). Both re-verified.
- **Statistics** — 1 finding, plan-mandated: the Spearman-Brown test exercised a local closure
  instead of shipped code. Resolved by shipping `fmharness.statistics.spearman_brown` as the one
  implementation of the lift, with the test importing it; the reporting layer consumes it. The
  round also brought the branch under the CI gates the plan had omitted (ruff check/format,
  strict pyright) — added to the plan's Global Constraints mid-run.
- **Measurement core** — reviewed at the strongest tier; the masked-Pearson algebra, the three
  null strata with their cross-half construction, and both planted closed forms were verified
  independently and found correct. 4 findings, all latent: an interface-name drift
  (`repl` → `repl_col`), a variable rebinding that would have broken the tercile control's row
  alignment, a missing column intersection that could silently misalign gene sets for direct
  callers, and a short-stratum sampler that bootstrapped with replacement instead of exhausting
  the available pairs. All fixed and re-verified; the last two were ruled against the plan's own
  text.
- **Reporting layer** — clean (no Critical/Important findings). The planted end-to-end fixture
  recovered its known reliability through the real CLI.
- **Promotion** — the reviewer's one Critical (a suspected false lint-gate claim) was refuted by
  the controller running the gate directly; one report-completeness gap fixed. The implementer
  caught a genuine defect in the planned code: capturing the clean-tree flag after writing
  outputs would have made it unconditionally false.
- **Tranche ingestion** — 1 finding, plan-inherited and load-bearing: shards with no
  download-time metadata entry entered the manifest unverified. Ruled a refusal; fixed and
  re-verified before the real 1,026-shard registration ran.
- **Alpine plumbing** — the `switch` verb's injection-safety was probed empirically and held;
  2 findings on stale content in the rewritten download job (vestigial dependency check,
  memory sized for removed work, header describing removed aggregation) — fixed.
- **Execution & promotion** — the evidence-critical review verified the chain end to end:
  recorded hashes against recomputed ones, byte-identity of the task-side and promoted CSV,
  the tranche `content_hash` against `data_commit`, and the pool arithmetic reconciling exactly
  to the scored pair count. 1 Critical + 3 Important, all in the prose around the number
  (a stale "produces no measurement" claim, the number missing from STATE's own row, an
  unreconciled producing-vs-promoting commit, and a "commands as run" entry that was not run);
  all fixed and re-verified verbatim against the artifacts.

## Final whole-branch review

Five passes on the strongest tier, with independent re-derivation of the provenance chain
(own checksums, own recounts, full suite at zero skips). Verdict: the scientific artifact is
sound; six self-consistency findings, all fixed in one wave with the promoted record untouched:
unlanded-lineage vocabulary in the shipped measurement script and its job header; the task-status
contradiction (ruled: OPEN until merge); this document's absence; a stale test-module docstring;
`ralpine help` printing nothing; and promotion inputs keyed by ephemeral paths (fixed forward —
durable labels for future promotions; rung 0's record deliberately stands with its documented
mapping).

## Standing follow-ups (triaged non-blocking at the final review)

Recorded here so they are findable, with where they bite: derive the planted-reliability
expectation from the realized plate split and widen the zero-signal control's seed robustness
(test hardening); a scale note for the O(n²) mismatched-pair enumeration; tranche-ingestion
edges (a `.cache` decoy test, clean failure on truncated metadata, atomic record+sidecar
writes); the promotion script's subprocess calls on the refusal path; `00`/`01` job scripts
still using the `module load anaconda` pattern the other jobs document as failing (bites only
on a scratch-purge rebuild); `docs/environment.md` §4's asserted-but-absent static-asset
manifest (reconcile when rung 4 registers GDSC2); Python-level bootstrap loops in
`fmharness.statistics` (vectorizable); the schema docstrings' `PredictionRecord` references
(rung 1's reconciliation); and the pre-existing `.gitignore` trailing-comment hazard — three
patterns (`reports/`, `containers/*.sif`, `.fmharness/`) carry inline comments and so match
nothing.

One observation worth carrying to rung 1's design rather than fixing here: the design's Steps
line omits `restrict` and `split` although the run restricts (panel, drugs) and splits
(plates); their behavior is exercised inside the build/score controls and measured by the pool
description, but a wrong CID list would pass every declared control — the declared-panel hashes
recorded at promotion are the guard.

## Drift audit (2026-08-28)

An independent audit of this branch against its own documents (`design.md`, `docs/DATA.md`,
`summary.md`, `verification.md`, `plan.md`) scored 68 clauses: **47 aligned, 3 recorded
deviations, 18 drift**. Auditor's judgment: the measurement is faithful and independently
re-derivable — tranche hash, input hashes, CSV byte-identity, the pool arithmetic
(1,650 − 50 = 1,600), and CSV-to-log-to-record agreement all recomputed and exact, suite green at
zero skips — and the drift found was documentary, clustered in unrecorded execution rulings, plus
one substantive panel finding. No promoted number was touched by the audit or by the fix wave
below.

Findings and dispositions, this fix wave:

1. **Declared vs scored panel** — reconciled in `design.md`'s drug-panel bullet, `docs/DATA.md`'s
   Restriction paragraph, and `summary.md`'s measured-quantity section: 32 CIDs resolve to 33
   Tahoe names, 1,650 candidate conditions, Ribociclib unscoreable, 1,600 = 1,650 − 50 (not
   32 × 50); dated 2026-08-28 entry recorded at `design.md`'s foot. FIXED.
2. **"~2× MDE" claim scope mismatch** (statistics-layer plant 2.03×, verified against the closed
   form; measurement-core plant 54.9×, with no test relating the plant to the MDE) — **not
   addressed by this wave**; carried forward as an open item.
3. **Unrecorded execution rulings** — one dated block at `design.md`'s foot records the two rulings
   that belong at the design level (`spearman_brown` as a third shipped statistics function;
   `clean_tree`'s redefinition to tracked-files-only, captured before writes); one dated block at
   `plan.md`'s foot records all six plus `pyarrow>=15`. FIXED/RECORDED.
4. **Steps line omits `restrict`/`split`** — `design.md`'s Steps line now reads `build, restrict,
   split, score, null, promote, document`; the Controls section gained **restrict** and **split**
   entries; two new tests (`test_restrict_positive_panel_subset_scores_exactly_the_subset`,
   `test_restrict_negative_disjoint_panel_scores_nothing`) exercise the restrict controls against
   the real `score_split_half`. FIXED.
5. **Missing per-gene figure** — `write_per_gene_figure` added to `scripts/delta_reproducibility.py`
   and wired into `main()`; the figure generated from the committed CSV
   (`rung0_per_gene_reliability.png`); `verification.md` now reads the diagnostic with the actual
   numbers (97.0% of genes r > 0, median per-gene r 0.146). FIXED.
6. **Unreachable `task-8-facts.md` citations** — both replaced in `verification.md` with
   self-contained statements naming the controller's execution log and noting the two input
   hashes are independently recomputable (they also appear in, and match, the promoted provenance
   record). FIXED.
7. **Decision-history section-title mismatch** — `design.md`'s dated entry now quotes the section's
   real title, "Why use the DE statistics as published". FIXED.
8. **Pilot commit misattribution** — `design.md`'s Expected-result line now distinguishes the
   recorded commit (`4c23f609`) from the archive branch tip (`ff88bba`). FIXED.
9. **Control-table number that doesn't reproduce** — `summary.md`'s row corrected to "recovered
   (0.800-0.809 across seeds; tolerance plus/minus 0.05)". FIXED.
10. **Post-promotion prose rewrite (`c6f1baf`) unrecorded** — one sentence added to
    `verification.md`'s commit-reconciliation paragraph; the script docstring's nonexistent-path
    example (`data/static/tahoe_drug_names.txt`) fixed to describe `--drug-names-file`
    generically. FIXED.
11. **Bare Alpine-only path forms** — `data/static/tahoe_target_cids.txt` is not tracked on this
    branch; `design.md` and `docs/DATA.md` now say so rather than citing it bare; the
    `/projects`-path decision entry reworded to "at its repository-relative path on Alpine".
    FIXED.
12. **`pyarrow>=15` and unticked checkboxes** — recorded in `plan.md`'s dated block; all 55
    checkboxes ticked (every task executed). FIXED/RECORDED.

Reverse-direction items landed but undescribed — `ralpine help`'s verb rewrite and
`register_tranche`'s etag/unverified-shard refusal — are now named in `design.md`'s
ported-apparatus table.

One process change came with this wave, beyond the individual fixes: the work lifecycle is
reordered so a drift audit like this one runs *before* promotion on future tasks, not after —
closing the gap that let this wave's findings accumulate undetected past promotion.
