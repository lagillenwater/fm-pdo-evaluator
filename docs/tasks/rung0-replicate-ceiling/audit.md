# Rung 0 — drift audit record

**As of** 2026-08-28. The audit stage's artifact (PROCESS §1): the design read as numbered,
checkable claims against the landed tree, every departure classified, the fix wave's
dispositions, and the re-audit that decides whether the audit passed. An independent reader
performed the first audit at `15462bc`; the fix wave landed as `174ce62`, `e9f5c67`, `4bf24c3`.
This task's audit ran after promotion; it is the case that moved the stage before promotion for
every later task (PROCESS Changes, 2026-08-28).

## First audit (at `15462bc`)

Method: five passes over `design.md`, SPEC's rung-0 section and rules, and the plan's Global
Constraints, each clause verified against the tree with named evidence (recomputed hashes,
recounts of the committed CSVs, diffs against the archive branch, a full-suite run at 50
passed / 0 skipped). Verdicts: ALIGNED, DEVIATION-RECORDED (a dated entry exists), DRIFT
(departed, unrecorded).

**Counts — 68 clauses: ALIGNED 47, DEVIATION-RECORDED 3, DRIFT 18** (the first auditor's own enumeration; the condensed tables below merge some clause ranges in transcription and expand to 69/49/2/18 — the DRIFT set of 18 reconciles exactly either way, and it is the set that matters).

Auditor's judgment, verbatim: the scientific artifact is faithful and independently checkable —
the tranche content hash recomputed from the committed manifest, both promotion input hashes,
the byte-identity of the task-side and promoted CSVs, the pool arithmetic (1,650 − 50 Ribociclib
= 1,600 = n_pairs), and the CSV-to-log-to-record agreement all reproduced exactly. The drift is
almost entirely in the documents rather than the measurement, clustered in execution-time
rulings captured in prose but never as dated entries, plus one substantive finding: the ceiling
was measured over 31 of the 32 declared compounds plus a duplicate Trametinib solvate name, with
Ribociclib silently contributing nothing — and the 32 × 50 = 1,600 coincidence made the
discrepancy invisible in every document.

### Clause verdicts — design.md

| # | Clause | Verdict |
|---|---|---|
| D1 | Status OPEN, closes at merge | ALIGNED |
| D2 | Steps line: build, score, null, promote, document | DRIFT — run also restricts (panels) and splits (plates); rule-4 scan never demanded their controls |
| D3 | Delta = per-gene log2FC treated vs DMSO from the Tahoe DE table | ALIGNED |
| D4 | Plates split hash(plate)%2, halves averaged, correlated over panel | ALIGNED |
| D5 | One correlation per (line, drug) pair | ALIGNED |
| D6 | Mean-expression reproducibility not measured | ALIGNED |
| D7 | Per-gene diagnostic: CSV **and figure**, referenced from verification.md | DRIFT — no per-gene figure existed; CSV never discussed |
| D8 | Diagnostic not promoted | ALIGNED |
| D9 | Declared statistic: per-pair Pearson, mean over pairs, Spearman-Brown | ALIGNED |
| D10 | Declaration lives in SPEC rungs 0/1 measure lines | ALIGNED |
| D11 | Tranche record fields and content hash over sorted manifest; refuses overwrite | ALIGNED (hash recomputed: matches) |
| D12 | data_commit = tranche content_hash; args name the tranche id | ALIGNED |
| D13 | Drug panel = 32 CIDs in the CID file | ALIGNED (promoted copy = 32 lines, hash matches record) |
| D14 | "32 × 50 = 1,600 candidate pairs" | DRIFT — 33 Tahoe names, 1,650 candidates; Ribociclib unsplittable (0 pairs); 1,600 = 1,650 − 50 |
| D15 | CID file cited bare | DRIFT (minor) — Alpine-only file, needs the "on Alpine" form |
| D16 | Pool description measured: lines, drugs, pairs, plates per half, doses | ALIGNED (every count reproduced) |
| D17 | Gene panel pinned by path + sha256 at promotion, not committed | ALIGNED |
| D18 | Replicate unit = plate | ALIGNED |
| D19 | "Every build script above is on this branch" | DEVIATION-RECORDED (gene-panel derivation and 00's input stated as archived/Alpine-only) |
| D20 | DE statistics used exactly as published | ALIGNED |
| D21–D24 | build/score/null/tercile controls declared and implemented | ALIGNED (auditor reran the planted-0.8 recovery) |
| D25 | MDE at α=0.05, power=0.80 beside every promoted comparison | ALIGNED |
| D26 | "Plants sit at roughly 2× the MDE" | DRIFT — true of the statistics plant (2.03×, verified); measurement-core plant is 54.9×; no test relates plant to MDE |
| D27 | Trivially powered at n≈1,600 | ALIGNED |
| D28 | statistics.py ships two named functions | DRIFT — a third, `spearman_brown`, shipped unnamed in design/plan |
| D29 | Three null strata as described | ALIGNED (with the D14 caveat: the solvate name is a distinct "drug" in the strata) |
| D30 | Run sequence, partition, CPU-only | ALIGNED (wall ~40 min vs ~2h budgeted) |
| D31 | Outputs to task folder, log beside the promoted result | ALIGNED |
| D32 | Promotion fields complete; checksums | ALIGNED (both copies re-hashed: byte-identical) |
| D33 | STATE + README move in the promotion change | ALIGNED (one commit before, per the plan's own clean-tree ordering) |
| D34 | Expected result = archived pilot numbers | ALIGNED (pilot CSV re-read from the archive branch) |
| D35 | Pilot at "commit ff88bba" | DRIFT (minor) — pilot's own record says 4c23f609; ff88bba is the branch tip |
| D36 | Current result, job 31758395 | ALIGNED |
| D37–D42 | Ported-apparatus rows | ALIGNED, with reverse-direction items below (help-verb rewrite; tranche integrity checks unlisted) |
| D43 | Out-of-scope items untouched | ALIGNED |

### Clause verdicts — SPEC rung 0 and rules

| # | Clause | Verdict |
|---|---|---|
| S1–S5 | Question / Measure / Passing / task indexed / rule 1 | ALIGNED |
| S6 | Rule 2: reversals dated at design/plan feet | DRIFT — six execution rulings recorded only in review/verification prose |
| S7 | Rule 3: README in step | ALIGNED |
| S8 | Rule 4: controls for each measurement step touched | DRIFT (= D2) |
| S9 | Rule 4: MDE beside every promoted comparison | ALIGNED |
| S10 | Rule 4: controls placed relative to the MDE | DRIFT (= D26) |

### Clause verdicts — plan Global Constraints and text

| # | Clause | Verdict |
|---|---|---|
| P1–P4 | Statistic, controls, suite green, CI gates | ALIGNED (P4 DEVIATION-RECORDED: gates added mid-run, recorded) |
| P5 | Landed docs reference only landed work; path forms | DRIFT — verification.md cited the git-ignored, deleted execution ledger twice; one absolute site path in a decision entry |
| P6–P9 | Explicit staging; commit discipline; ralpine-only; ported-via-git-show | ALIGNED (33/33 commits carry the trailer) |
| P10 | Plan shows `replace=avail.size < n_perm` null sampler | DRIFT — shipped `replace=False` (ruled, unrecorded in plan) |
| P11 | Plan's score_split_half intersects index only | DRIFT — shipped column intersection (ruled, unrecorded) |
| P12 | Plan's clean_tree uses bare `--porcelain` | DRIFT — shipped `-uno`, captured before writes (ruled, unrecorded) |
| P13 | Plan's `--input PATH` | DRIFT — shipped `LABEL=PATH` (ruled, unrecorded) |
| P14 | Plan's register() checks etags only | DRIFT — shipped unverified-shard refusal (ruled, unrecorded) |
| P15 | Plan's five-package dependency list | DRIFT (minor) — pyarrow shipped unrecorded |
| P16 | Plan checkboxes track execution | DRIFT (minor) — 0 of 55 ticked |

### Additional findings

- verification.md's "commands as run" contained one command never run, against a path that does
  not exist (the shard-listing form); the actually-run find-count (→ 1,026) was absent.
- A decision entry cited a section title later renamed ("accepted upstream" vs "as published").
- summary.md's control table quoted a recovered range (0.796–0.800) the shipped fixture does not
  produce (seeds 0/1/2 → 0.809/0.800/0.806).
- The post-promotion prose rewrite of the measurement script (`c6f1baf`) was unreconciled
  against the record's code_commit, and its docstring example cited a nonexistent file.
- Reverse-direction (landed, undescribed): the ralpine help-verb rewrite; the tranche
  registration's etag cross-check and unverified-shard refusal.
- Decision-history verification: every dated entry held at HEAD except the renamed-section
  citation; the 2×-MDE claim held only for the statistics layer; the panel declaration was
  departed from in execution (D14).

## Fix wave (commits `174ce62`, `e9f5c67`, `4bf24c3`)

Gates green at every commit (ruff check/format, strict pyright, 52 passed / 0 skipped — two new
restrict-control tests). The promoted record and CSV untouched, verified by diff.

1. D14 (declared vs scored panel) — the measured reconciliation stated in design.md, DATA.md,
   and summary.md; the solvate-in-the-null placement stated as conservative at the design
   level; dated design entry. FIXED.
2. D26 (plant-vs-MDE scope) — **ruled after the wave** (below). RECORDED.
3. S6/P10–P15 (unrecorded rulings) — dated blocks at design.md's and plan.md's feet covering
   all six rulings plus pyarrow. RECORDED.
4. D2/S8 (Steps line) — Steps now build, restrict, split, score, null, promote, document;
   restrict and split control entries added; two new tests exercise the restrict controls
   against the real scoring path. FIXED.
5. D7 (per-gene figure) — `write_per_gene_figure` added to the script for future runs; the
   current figure generated from the committed CSV; verification.md reads the diagnostic with
   computed numbers. FIXED.
6. P5 (unreachable ledger citations) — replaced with self-contained, recomputable statements.
   FIXED.
7. D35, renamed-section citation, path forms — corrected. FIXED.
8. Summary control range corrected to the shipped fixture's values; the post-promotion prose
   rewrite reconciled in verification.md; the phantom filename removed from the docstring
   example. FIXED.
9. The non-independence assumption restated with its quantitative exposure in summary.md and
   verification.md, per PROCESS §3's new principle; the derangement permutation check declared
   there runs as this task's final verification step. FIXED (statement) / RUNNING (check).
10. The audit record itself — this document; review.md points here. FIXED.

**Ruling on finding 2 (D26/S10)**: the "~2× the MDE" placement is the discipline for the
statistics-layer plant, where the closed form makes the placement verifiable; the
measurement-core fixtures plant closed-form-verifiable values (0.8) chosen for exact
recoverability, and serve as recovery controls, not power-calibration controls. The design's
Power bullet is narrowed accordingly with a dated entry rather than forcing fixture plants near
their MDEs, which would trade an exact known answer for a fragile one.

## Re-audit

Pending: a fresh reader re-checks the 18 drifted clauses and the reverse-direction items against
the fixed tree; the audit passes only when every item verdicts FIXED or RECORDED. The verdict is
appended here.
## Re-audit (2026-08-28, fresh reader, at `5d0c7c3`)

Every item verified by reading shipped code or recomputing the number, never by trusting a
document: all 17 remaining drifted clauses, both reverse-direction items, and all four
additional findings verdicted FIXED or RECORDED; the promoted record confirmed unchanged since
`3b674de`; the suite green at 52 passed / 0 skipped; four independent recomputations (the pool
arithmetic, the per-gene diagnostic's every stated number, the planted-0.8 recovery at three
seeds, the 2.031× plant-to-MDE ratio) reproduced the documents' values exactly.

**Verdict: NOT PASSED on the first pass** — one open item and four one-line issues:

1. P5 residual: `verification.md` still cited the untracked execution-ledger file once more, in
   the transient-SSH note. Removed.
2. This document's headline counts did not reconcile with its own condensed tables (transcription
   merged clause ranges). Annotated above rather than silently edited; the 18-clause DRIFT set
   reconciles exactly either way.
3. Stale "As of" headers on `design.md` and `docs/DATA.md` (2026-08-27 under 2026-08-28 content).
   Updated.
4. The fix-wave disposition overstated D14 ("all three" documents carry the null-placement
   clause; it is a design-level statement). Disposition wording corrected above.
5. Pre-existing, first audit missed it: a broken em-dash ("__ see note below") in `design.md`'s
   Tahoe bullet, from before the audit. Repaired to name the actual section.

All five corrections applied in the commit carrying this entry; a confirmation pass by a fresh
reader checked exactly these lines. Its verdict:
