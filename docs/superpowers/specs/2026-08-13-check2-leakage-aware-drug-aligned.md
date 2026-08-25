# Check 2, leakage-aware, for the drug-aligned Stack checkpoint — handoff

**Status:** not started. This document exists so a fresh session (with working transcript
persistence) can pick this up without re-deriving the last two days of Alpine debugging and
Check-1 results. Start a new session by reading this file, then invoke
`superpowers:brainstorming` to resolve the open design question below before writing a plan.

## Where this picks up from

Branch `worktree-modular-harness-core`, pushed to `origin` (the `lagillenwater` fork), **not**
merged to `main`, **not** opened as a PR (explicitly left "keep as-is" — that decision is still
yours to make). HEAD at the time this doc was written: `5d6d45563466e11424ee49b8b6b2ec3bd20b362d`.

This branch's prior work (already committed, already pushed):
- `docs/superpowers/specs/2026-08-11-stack-drug-alignment-and-check1-design.md` and
  `docs/superpowers/plans/2026-08-11-stack-drug-alignment-and-check1.md` — the design/plan that
  fixed Stack's sci-Plex drug-alignment fine-tune pipeline and built a registry-driven Check-1
  driver (`scripts/check1_registry_driver.py`).
- A real drug-aligned Stack checkpoint now exists and has been scored on Check 1 (below).

## What Check 1 found (already done, real data, not synthetic)

Reproduced `docs/tahoe_generation_results.md`'s published cytokine-aligned table exactly (see
"Data provenance gotcha" below for why that took real debugging), then scored the new
drug-aligned checkpoint:

| source | cytokine-aligned (published, reproduced exactly) | drug-aligned (unfiltered) | drug-aligned (leak-excluded) |
|---|---|---|---|
| additive | 0.225 | 0.225 | 0.224 |
| knn | 0.178 | 0.178 | 0.177 |
| pca | 0.207 | 0.207 | 0.206 |
| nmf | 0.221 | 0.221 | 0.221 |
| **stack** | **0.012** | **0.021** | **0.021** |

Ceiling: 0.30 raw / 0.46 Spearman-Brown (delta reproducibility, Tahoe plate split-half).

Drug alignment roughly doubles Stack's Check-1 correlation vs. the cytokine-aligned checkpoint,
but both are still deeply null relative to the additive floor and the ceiling. Stable under
leakage filtering (doubly_exposed_frac=0.003, ~0.3%) — not an artifact of the ~5 doubly-exposed
A549/drug pairs.

**Open item, not yet decided:** this table has not been written into
`docs/tahoe_generation_results.md` alongside the existing cytokine-aligned numbers. Decide
whether/how to do that (this session ran out before Lucas answered).

## The Check-2 gap this document is about

Check 2 (end-to-end GDSC2 AUC prediction: fixed signature readouts + a representation-controlled
penalized-regression grid, gap@k, MOA stratification — see `docs/tahoe_generation_results.md`'s
own Check-2 section for the existing cytokine-aligned numbers) was explicitly out of scope for
the plan that just finished. Extending it to the drug-aligned checkpoint surfaces a real tooling
gap between the two existing scripts:

- `scripts/score_generation_eval.py` already has full Check-2 machinery (GDSC2 AUC grid,
  `build_sample_design`, `grouped_cv_predict`, `score_predictions` in
  `src/fmharness/evaluation.py`) and can already take a generated-delta source via
  `--generated-dir`/`--query-baseline`/`--pert-map` — but it has **no leakage filtering at all**.
  Running it against the drug-aligned checkpoint would score Check 2 blind to the measured
  pretraining overlap below.
- `scripts/check1_registry_driver.py` (this session's registry-driven driver) **does**
  leakage-filter via `fmharness.leakage.filter_leakage` and the `PregeneratedStackGenerator`'s
  `LeakageQueryable` declaration — but it only implements Check 1. No Check-2 machinery exists
  in it at all.

**The open design question for brainstorming:** build leakage-aware Check-2 support into the
registry-driven driver (more work, consistent with the harness-core philosophy this whole
project has been moving toward — registries drive every check, not just Check 1), or accept a
leakage-blind Check-2 run through `score_generation_eval.py` directly (fast, but the ~0.3%
doubly-exposed overlap would be silently included, and Check 2's stakes — "does this actually
predict drug response" — are higher than Check 1's, so a silent leakage blind spot matters more
here than it did for the reproduction-check debugging above). Do not resolve this by picking one
unilaterally — that's exactly what brainstorming is for.

## The measured pretraining-overlap corpus (already derived, reuse these values)

Deriving this required mapping Tahoe's DepMap `ACH-XXXXXX` line IDs and PubChem CID drug IDs
against sci-Plex's plain cell-line names and free-text drug names — non-trivial, already done:

- **Line:** Tahoe's A549 is `ACH-000681` (via `data/raw/gdsc2_sarcoma/depmap/Model.csv`'s
  `ModelID`/`StrippedCellLineName` columns). sci-Plex 3's other two lines (K562, MCF7) are not in
  Tahoe's 50-line panel at all — only A549 overlaps.
- **Drugs (5, via `context_by_drug/pert_to_cid.tsv`'s pert_id -> CID mapping, matched to
  sci-Plex's free-text drug names by stripping parenthetical synonyms):**

  | Tahoe pert_id | CID | sci-Plex name it matched |
  |---|---|---|
  | Temsirolimus | 6918289 | Temsirolimus (CCI-779, NSC 683864) |
  | crizotinib | 11626560 | Crizotinib (PF-02341066) |
  | Fulvestrant | 104741 | Fulvestrant |
  | Trametinib | 11707110 | Trametinib (GSK1120212) |
  | 5-Fluorouracil | 3385 | Fluorouracil (5-Fluoracil, 5-FU) |

- Check-1 CLI equivalent (already verified working):
  `--corpus-lines ACH-000681 --corpus-drugs 6918289,11626560,104741,11707110,3385`
- `task_signal_in_pretrain="adjacent"` is the correct tier here (the checkpoint was fine-tuned on
  sci-Plex's raw line/drug identities, not on actual Tahoe/GDSC2 response labels) --
  `check1_registry_driver.py`'s `main()` already sets this automatically when corpus flags are
  given; a Check-2 equivalent would need the same tiering (`filter_leakage`'s tiered rule: drop
  doubly-exposed always, drop single-axis-exposed only when `task_signal_in_pretrain="direct"`).

## Data already pulled locally (this worktree, gitignored/untracked -- won't survive a fresh
## clone or a different worktree, but should still be here if you reopen this same folder)

- `tahoe_context.h5ad` (2.6GB) -- do NOT use this for ground truth; see the gotcha below.
- `tahoe_deltas/` (`real_delta.parquet`, `real_key.parquet`, `base.parquet`) -- the correct
  ground-truth bundle, use `--deltas-bundle tahoe_deltas`.
- `tahoe_query.h5ad`, `context_by_drug/pert_to_cid.tsv` -- small, needed by both checks.
- `generated/` (gitignored, 33 files) -- cytokine-aligned Stack generation output.
- `generated_sciplex/` (33 files) -- drug-aligned Stack generation output, from checkpoint
  `finetuned-epoch=4-val_loss=5.0847.ckpt` (lowest val_loss during the 10-epoch fine-tune).

> **Status note, 2026-08-25.** The claim above does not match what was run. Alpine job logs
> record the resolved checkpoint for every generation run (`04`'s "Resolved: CKPT=" line), and
> both published drug-aligned arms used **`finetuned-epoch=5-val_loss=6.1078.ckpt`** --
> `OUTDIR=generated_drug_aligned` (33 tasks, Tahoe) and `OUTDIR=generated_soragni_drug_aligned`
> (53 tasks, Path B). `finetuned-epoch=4-val_loss=5.0847.ckpt` was used only for
> `OUTDIR=generated_sciplex`. So the two arms ARE consistent with each other, and
> `11_soragni_generate.sbatch`'s comment is right, but this spec and
> `smoke_test_env.sbatch`'s `CKPT_SCIPLEX` default are not.
>
> It also was not the lowest-val_loss checkpoint. Available on Alpine: 5.0847, 5.9110, 6.1078,
> 7.2349, 10.6114, 13.0135. The published drug-aligned numbers come from the third-best by
> val_loss, and no record anywhere says why. Whether that was deliberate is not established.
> Consequence: every published drug-aligned result is a negative result obtained on a
> checkpoint that is not the best available, which weakens it as evidence about drug alignment.

- `sciplex_lines.txt`, `sciplex_drugs.txt` -- sci-Plex's distinct cell_line/pert_id values,
  extracted via a small script run on Alpine (see `git log` on this branch around
  2026-08-12/13 for the exact extraction one-liner if these are gone and need re-deriving).
- GDSC2 AUC labels for Check 2: check `src/fmharness/data/loaders/` and
  `scripts/score_generation_eval.py`'s own `--auc-tranche` default (`gdscv2`) for where this
  already loads from -- not newly pulled this session, should already be wired into the existing
  Check-2 path in `score_generation_eval.py`.

If any of these are missing, re-pull via `scripts/alpine/ralpine pull <remote-path> <local-path>`
-- remote root is `/projects/lgillenwater@xsede.org/repositories/fm-pdo-evaluator`. Watch for two
gotchas hit repeatedly this session: (1) `ralpine pull` on a directory needs the remote path
form rsync won't nest -- verify with `ls <local>` after pulling, a single-entry directory means
it nested one level deep and needs flattening; (2) the SSH ControlMaster connection drops
periodically mid-transfer (`rsync error: io_read_buf`, exit 255) -- `ralpine status` confirms if
it's down, and re-establishing it needs a human (`ssh alpine` interactively), `ralpine` cannot
do this itself.

## Data provenance gotcha (already resolved, but the failure mode will recur if forgotten)

`tahoe_context.h5ad` (live single-cell context, whatever currently sits on Alpine's repo root)
and `tahoe_deltas/` (a precomputed pseudobulk parquet bundle) are **not interchangeable** despite
identical downstream code -- confirmed by running the exact same scoring code against both and
getting baselines ~2.4x higher from `--context` than from `--deltas-bundle`. `tahoe_deltas/` (Jul
24) is what the published table was actually computed from; `tahoe_context.h5ad` (Jul 30) is a
later, different rebuild. Always use `--deltas-bundle tahoe_deltas`, not `--context`, for
anything that needs to compare against the published cytokine-aligned numbers. If Check 2 gets
its own registry-driven driver, it needs the same `--deltas-bundle`/`--context` choice
`check1_registry_driver.py` now has (see `ground_truth_source_declared_ambiguously` in that
file) -- don't let a Check-2 driver default back to a live `--context` rebuild.

## Alpine mechanics reminder

`ralpine` (`scripts/alpine/ralpine`) is deliberately read-only -- no `sbatch`/`scancel`. Claude
authors/commits `.sbatch` scripts, Lucas submits and pastes back errors/status, Claude
polls/pulls via `ralpine`. Alpine's partition/QOS names were found stale this session (`amilan`
retired -> `acpu`; `mem-normal` QOS has a real `MinTRES mem=240G` floor; `--export=ALL` breaks
`module load` by dragging in the submitting shell's `$MODULEPATH` -- use `--export=VAR=val,...`
without `ALL`) -- all already fixed in the committed `.sbatch` scripts on this branch, shouldn't
need rediscovering, but worth rereading `scripts/alpine/08_sciplex_prep.sbatch` and
`scripts/alpine/04_stack_generate.sbatch`'s comments if a *new* Alpine job needs writing for
whatever Check 2 ends up needing.
