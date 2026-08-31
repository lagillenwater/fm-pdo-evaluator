# Rung 0 — review record

**As of** 2026-08-31.

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
  round also brought the branch under the continuous integration (CI) gates the plan had omitted (ruff check/format,
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
  recovered its known reliability through the real command-line interface (CLI).
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
  recorded hashes against recomputed ones, byte-identity of the task-side and promoted comma-separated values (CSV),
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
manifest (reconcile when rung 4 registers Genomics of Drug Sensitivity in Cancer, release 2 (GDSC2)); Python-level bootstrap loops in
`fmharness.statistics` (vectorizable); the schema docstrings' `PredictionRecord` references
(rung 1's reconciliation); the pre-existing `.gitignore` trailing-comment hazard — three
patterns (`reports/`, `containers/*.sif`, `.fmharness/`) carry inline comments and so match
nothing; lock-based Alpine provisioning (containers/`uv` on Alpine, superseding the pinned
pip-fallback installs); `ALPINE_HOST` option-injection validation in `ralpine`; a `module save`
write primitive for `ralpine`'s allowlist; and a provably-uniform constrained sampler for
`sample_cross_derangement` (its current repair-based sampler is measurably non-uniform over the
`diff_drug` constraint set — see `verification.md`'s derangement section).

One observation worth carrying to rung 1's design rather than fixing here: the design's Steps
line omits `restrict` and `split` although the run restricts (panel, drugs) and splits
(plates); their behavior is exercised inside the build/score controls and measured by the pool
description, but a wrong PubChem compound identifier (CID) list would pass every declared control — the declared-panel hashes
recorded at promotion are the guard.

## Drift audit

The audit stage's full record — the 68-clause first audit, the fix wave's dispositions, and the
re-audit verdict — lives in [`audit.md`](audit.md). Headline: 47 aligned, 3 recorded deviations,
18 drift, all drift documentary except the declared-vs-scored panel finding; every item fixed or
recorded by the fix wave, and the audit passed (re-audit + confirmation recorded in
`audit.md`). The measurement itself was independently re-derived and reproduced exactly.

## External review (CodeRabbit, PR 14)

18 findings triaged. Dispositions:

- **15 accepted and fixed**: `ralpine`'s `find`/`scontrol`/`file`/`nvidia-smi` allowlist escapes
  closed, with a new read-only `jobinfo` verb replacing the write-capable `scontrol` access and
  boundary regression tests; `00_target_cids.sbatch`'s empty-mapped-set guard; the Tahoe
  download's pinned `--revision` and always-resume behavior, with new behavior tests;
  `derangement_null.py`'s input validation (`n < 2` finite pairs, `n_perm < 2`) refused before
  the permutation loop; `promote_result.py`'s provenance-record immutability refusal; the
  rule-4 controls scan tightened to the `positive:`/`negative:` colon form; three loader
  edge-case controls added to `test_rung0_controls.py`; the Alpine sbatch jobs' pip-fallback
  installs pinned to the `pyproject.toml` floors with upper bounds; the stratified (within-drug,
  cross-constrained) derangement nulls measured on the real Tahoe pool (job 31770850, recorded
  in `verification.md`); the seven documentation corrections above (a–g, `verification.md`,
  `summary.md`, `audit.md`, this document, `docs/environment.md`, `design.md`); and the labeled
  re-promotion below.
- **1 dismissed with evidence**: a finding that the shard manifest was missing from the
  repository — already committed at
  `data/tranches/tahoe100m-pseudobulk-de.v1.manifest.txt` since the tranche-registration commit
  (`25cec05`).
- **Deferred follow-ups** (recorded, not blocking; also carried in Standing follow-ups above):
  lock-based Alpine provisioning; `ALPINE_HOST` option-injection validation; a `module save`
  write primitive for `ralpine`'s allowlist; and a provably-uniform constrained sampler for
  `sample_cross_derangement`.
