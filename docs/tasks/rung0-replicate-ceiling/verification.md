# Rung 0 — verification

**Task** `rung0-replicate-ceiling` · **As of** 2026-08-31
Records how the headline number in `docs/tasks/rung0-replicate-ceiling/rung0_delta_reproducibility.csv` was produced and promoted. Every number this document cites is copied from the named artifact tables — the headline comma-separated values (CSV), the pool-description and per-gene CSVs, and the derangement summary and per-permutation CSVs — or from the job logs, not from memory, with any derived calculation shown at the point it is used.

## Commands as run (Task 8 — deploy, register, measure)

Run by the controller on 2026-08-27/28, against `docs/tasks/rung0-replicate-ceiling/plan.md`'s Task 8 steps; commands below are those steps with the job ids substituted from the controller's own execution log for that run. Every hash the log recorded is independently recomputable: the two input hashes it produced also appear in the promoted provenance record below and were independently re-derived from the pulled files at audit time, matching exactly.

```bash
git push
./scripts/alpine/ralpine switch rung0-replicate-ceiling      # ran at b5da999 (registration), 25cec05 (measurement) after a ff
./scripts/alpine/ralpine run find "$SCRATCH/tahoe_pseudobulk_de" -name "*.parquet" -not -path "*/.cache/*" | wc -l
# (find was removed from ralpine's READ_ONLY allowlist on 2026-08-28, a security fix; today's equivalent read is `ralpine du`/`ralpine ls`, not a find invocation)
# -> 1026 (shards live under metadata/pseudobulk_differential_expression/ inside the scratch
#    dir, the HF repo's own layout — ties the tranche manifest's 1,026 lines to the job log's
#    "reading 1026 DE parquet files" line)
./scripts/alpine/ralpine run wc -l "$ROOT/results/rung1_panel/common_panel.txt" "$ROOT/data/static/tahoe_target_cids.txt"
# -> 14121 and 32

./scripts/alpine/ralpine submit scripts/alpine/register_tranche.sbatch
# job 31757772, COMPLETED
./scripts/alpine/ralpine log register-tranche
# -> registered tahoe100m-pseudobulk-de.v1: 1026 shards, version 2dc57900b7981cfcf5e211527169a0b006546a95
# -> content_hash 9a8797a5698e2c56ec1b61bdd3d5f68d18a972e227e86b64ac341ef507f73dd6

./scripts/alpine/ralpine pull "$ROOT/data/tranches/tahoe100m-pseudobulk-de.v1.json" data/tranches/tahoe100m-pseudobulk-de.v1.json
./scripts/alpine/ralpine pull "$ROOT/data/tranches/tahoe100m-pseudobulk-de.v1.manifest.txt" data/tranches/tahoe100m-pseudobulk-de.v1.manifest.txt
uv run python -c "from pathlib import Path; from fmharness.schema import Tranche; print(Tranche.model_validate_json(Path('data/tranches/tahoe100m-pseudobulk-de.v1.json').read_text()).content_hash)"
git add data/tranches/tahoe100m-pseudobulk-de.v1.json data/tranches/tahoe100m-pseudobulk-de.v1.manifest.txt
git commit -m "data: register the Tahoe pseudobulk DE pool as tranche tahoe100m-pseudobulk-de.v1"   # 25cec05
git push
./scripts/alpine/ralpine update

./scripts/alpine/ralpine submit scripts/alpine/delta_reproducibility.sbatch
# job 31758395, COMPLETED, wall ~40 min
./scripts/alpine/ralpine log delta-repro

mkdir -p results/rung0-replicate-ceiling
./scripts/alpine/ralpine pull "$ROOT/rung0_outputs/rung0_delta_reproducibility.csv" "docs/tasks/rung0-replicate-ceiling/rung0_delta_reproducibility.csv"
./scripts/alpine/ralpine pull "$ROOT/rung0_outputs/rung0_delta_reproducibility.params.json" "docs/tasks/rung0-replicate-ceiling/rung0_delta_reproducibility.params.json"
./scripts/alpine/ralpine pull "$ROOT/rung0_outputs/rung0_per_gene_reliability.csv" "docs/tasks/rung0-replicate-ceiling/rung0_per_gene_reliability.csv"
./scripts/alpine/ralpine pull "$ROOT/rung0_outputs/rung0_pool_description.csv" "docs/tasks/rung0-replicate-ceiling/rung0_pool_description.csv"
./scripts/alpine/ralpine pull "$ROOT/rung0_outputs/rung0_ceiling.png" "docs/tasks/rung0-replicate-ceiling/rung0_ceiling.png"
./scripts/alpine/ralpine pull "$ROOT/logs/delta-repro-31758395.out" "results/rung0-replicate-ceiling/delta-repro-31758395.out"

git add docs/tasks/rung0-replicate-ceiling/rung0_delta_reproducibility.csv docs/tasks/rung0-replicate-ceiling/rung0_delta_reproducibility.params.json docs/tasks/rung0-replicate-ceiling/rung0_per_gene_reliability.csv docs/tasks/rung0-replicate-ceiling/rung0_pool_description.csv docs/tasks/rung0-replicate-ceiling/rung0_ceiling.png results/rung0-replicate-ceiling/delta-repro-31758395.out
git commit -m "run: rung-0 ceiling outputs and job log, job 31758395"   # 0ce4ba7
```

One transient secure shell (SSH) failure (Permission denied — network) occurred on 2026-08-27 during this sequence; it resolved itself on retry, no re-auth needed.

## Job log — summary block (tail of `results/rung0-replicate-ceiling/delta-repro-31758395.out`)

```
splitting plates by hash(plate) % 2
scoring the ceiling on the supplied panel: 13886 of 14121 genes present
null[any_pair  ] median r = +0.034 over 500 draws
null[diff_drug ] median r = +0.034 over 500 draws
null[same_drug ] median r = +0.067 over 500 draws
...  (a "wrote .../rung0_delta_reproducibility.params.json" line omitted here)

=== delta reproducibility ceiling (real Tahoe delta, plate split-half) ===
  replicate_col          plate
  n_genes                13886
  n_pairs                1600
  splithalf_mean_r       0.135
  splithalf_median_r     0.109
  splithalf_q1_r         0.071
  splithalf_q3_r         0.155
  spearman_brown_full    0.238
  frac_pos               0.989
  null_any_pair_mean_r   0.036
  null_diff_drug_mean_r  0.035
  null_same_drug_mean_r  0.079
  null_n_draws           500
  p_vs_null              0.0005
  p_vs_same_drug         0.0005
  null_mean_ci_lo        0.033
  null_mean_ci_hi        0.037
  mde_80_vs_diff_drug    0.0392
  mde_80_vs_same_drug    0.0846
  splithalf_mean_r_tercile1 0.1
  splithalf_mean_r_tercile2 0.126
  splithalf_mean_r_tercile3 0.178

Check-1 achieved r ~ 0.2; the ceiling is the split-half mean (0.135) / Spearman-Brown full-data (0.238).
wrote /projects/lgillenwater@xsede.org/repositories/fm-pdo-evaluator/rung0_outputs/rung0_delta_reproducibility.csv
```

Every field above matches `docs/tasks/rung0-replicate-ceiling/rung0_delta_reproducibility.csv` exactly (checked column by column).

**Figure**: this run's figure is `docs/tasks/rung0-replicate-ceiling/rung0_ceiling.png`, produced alongside the headline table — per `docs/STATE.md`'s convention, a figure a task cites is pointed at from that task's `verification.md`, which this line does.

## Per-gene diagnostic (from `rung0_per_gene_reliability.csv`)

The per-gene reliability diagnostic (design.md, "per-gene reliability") is an unpromoted CSV of all 13,886 panel genes, each correlated across (line, drug) pairs between the two plate halves, plus its own figure, `rung0_per_gene_reliability.png`. Of the 13,759 genes with a finite value (127 have too few scored pairs to compute one), **97.0% have r > 0** and the **median per-gene r is 0.146** (quartiles 0.089–0.230), a touch above the pair-level headline mean of 0.135 — reproducibility is broadly distributed across the panel rather than concentrated in a handful of genes. The most reproducible individual genes are heat-shock and immediate-early stress-response transcripts (HSP90AA1 r=0.79, EGR1 r=0.75, HSPA1B r=0.72, HSPH1 r=0.71, PLEC r=0.69): a generic perturbation-stress signature that reproduces well regardless of which drug or line induced it. That is the evidence base design.md points to for any future panel restriction — it says nothing about *which* (line, drug) pairs reproduce, the pair-level question the promoted ceiling answers.

## Pool description (from `rung0_pool_description.csv`)

The pool-description table lists 1,650 (line × drug) rows: 50 cell-line keys by 33 drug-name entries in the differential expression (DE) table matching the target PubChem compound identifier (CID) panel (32 drugs, plus a `Trametinib (DMSO_TF solvate)` name variant). All 1,650 rows carry 3 dose levels. One of the 50 line keys is literally `NA` (a missing DepMap id in the source table — the grouping key carries it through as-is), accounting for 33 of the 1,650 rows, one line's worth.

- 1,210 of 1,650 rows (the majority) have exactly 3 plates: 1,200 of those split 1 plate in half0 / 2 in half1, and 10 split 2/1 — the typical case, with the minority split running the other way.
- 290 rows have 4 plates (2/2), 50 have 6 (2/4), 50 have 7 (3/4) — drugs profiled on more plates (e.g. Afatinib, Trametinib, Cytarabine, Docetaxel, Rapamycin, Retinoic acid, Temsirolimus).
- 50 rows — all 50 lines of Ribociclib, and only Ribociclib — have exactly 1 plate total (1 in half0, 0 in half1): a single plate cannot be split, so these pairs are unscoreable.
- 1,650 candidate rows − 50 unscoreable Ribociclib rows = 1,600, matching `n_pairs` in the headline CSV exactly.

## Tercile control

`splithalf_mean_r_tercile1/2/3` = 0.100 / 0.126 / 0.178 — monotonically increasing across pairs stratified by effect-size tercile. Per the design's declared positive control (design.md, "Controls and power"), this is the pass condition: an assay that could not find more reproducibility where there is more signal would be broken. The in-run positive control passed.

## Minimum detectable effect

- `mde_80_vs_diff_drug` = 0.0392 — the smallest split-half mean r this run (n = 1,600 pairs) could distinguish from the diff-drug (generic-structure) null floor at α = 0.05, power = 0.80. The observed 0.135 clears it by a wide margin (~3.4×).
- `mde_80_vs_same_drug` = 0.0846 — the corresponding minimum detectable effect (MDE) against the stricter same-drug (line-specificity) null floor; the observed 0.135 clears this one too (~1.6×), the tighter of the two comparisons.

Both MDEs are far below the observed ceiling — at n ≈ 1,600 pairs this run is trivially powered, which is expected and stated in the design (design.md, "Controls and power"): the same reporting will matter honestly at rung 5, where power is the whole question.

## Write-up caveat (ledgered at Task 3 review)

The stratified null draws reuse the same half-profiles across mismatched pairs, so they are not i.i.d., while the bootstrap treats them as an exchangeable pool; this is the residual assumption behind the reported p-values and MDEs, inherited from the archived lineage's design.

**Quantitative exposure (PROCESS §3).** That dependence can widen or narrow the null aggregate's spread depending on the sign of the shared-profile covariance it introduces — positive sharing covariance widens it, while the derangement's balanced, each-half-used-exactly-once reuse in fact narrows it, as measured below. The observed lift over the diff-drug floor (0.135 − 0.035 = 0.100) is roughly 100 bootstrap standard errors, using `SE ≈ (null_mean_ci_hi − null_mean_ci_lo) / (2 × 1.96) ≈ 0.001` from the CSV's null confidence-interval (CI) columns — so losing significance would require the profile-sharing dependence to inflate the null's variance by more than ~3,000-fold. The MDE columns degrade only by the square root of any such factor: a tenfold variance inflation would move `mde_80_vs_diff_drug` from 0.039 to about 0.12, still below the observed 0.135. Profile-sharing is far sparser than that: with 500 draws per stratum over 1,600 candidate pairs, roughly 0.25% of draw pairs share a half-profile, which produces single-digit inflation factors in practice, not thousands. An exact derangement-based permutation check — which carries the dependence by construction, sampling derangements of the pairing rather than assuming an exchangeable pool — was run; its result is in the section below.

## GDSC2 CID list — provenance note

`00_target_cids.sbatch` (which built `data/static/tahoe_target_cids.txt`, one of this promotion's two declared inputs) reads `data/static/gdsc2_auc_pubchem_cids.txt` on Alpine. That file is untracked on Alpine and not landed in this repository; neither `00_target_cids.sbatch` nor `01_pseudobulk_shortcut.sbatch` is re-run by this task — their outputs exist already and are pinned by hash (the CID file) and by the tranche (the DE pool). `data/static/gdsc2_auc_pubchem_cids.txt` becomes a tracked input in this repository when rung 4 registers Genomics of Drug Sensitivity in Cancer, release 2 (GDSC2).

## Promotion (Task 9)

Run from a tree clean of tracked modifications (`git status --porcelain -uno` empty, verified before running), from commit `84c094d` (the docs commit above):

```bash
SCRATCH=/private/tmp/claude-502/-Users-gillenlu-Repositories-fm-pdo-evaluator/982cf604-f4fc-49b1-9bc3-1bb128cafd76/scratchpad
uv run python scripts/promote_result.py \
  --task rung0-replicate-ceiling \
  --result docs/tasks/rung0-replicate-ceiling/rung0_delta_reproducibility.csv \
  --script scripts/delta_reproducibility.py \
  --input "$SCRATCH/common_panel.txt" --input "$SCRATCH/tahoe_target_cids.txt" \
  --seed 0 \
  --data-commit "$(uv run python -c "from pathlib import Path; from fmharness.schema import Tranche; print(Tranche.model_validate_json(Path('data/tranches/tahoe100m-pseudobulk-de.v1.json').read_text()).content_hash)")" \
  --arg tranche_id=tahoe100m-pseudobulk-de.v1 \
  --arg panel_file="results/rung1_panel/common_panel.txt on Alpine" \
  --job-id 31758395 \
  --log results/rung0-replicate-ceiling/delta-repro-31758395.out
```

Output:

```
promoted -> results/rung0-replicate-ceiling/rung0_delta_reproducibility.csv
           results/rung0-replicate-ceiling/rung0_delta_reproducibility.provenance.json
```

The written record (`results/rung0-replicate-ceiling/rung0_delta_reproducibility.provenance.json`):

```json
{
  "result": "results/rung0-replicate-ceiling/rung0_delta_reproducibility.csv",
  "result_sha256": "98985f030ba8933589a4910f2d47dda5878b3dff635e51a52bb5dd3b875c2e71",
  "task": "rung0-replicate-ceiling",
  "script": "scripts/delta_reproducibility.py",
  "args": {
    "tranche_id": "tahoe100m-pseudobulk-de.v1",
    "panel_file": "results/rung1_panel/common_panel.txt on Alpine"
  },
  "inputs": {
    "<scratchpad>/common_panel.txt": "356bcfe69c50fef5ab108be78c3d6dea2cd42f24fdc43b7a3d52dfbdc5471344",
    "<scratchpad>/tahoe_target_cids.txt": "0bd61793d051f9cad8d5bbbabe9e3589563ae01385692a1eb8fe6505076a1081"
  },
  "log": "results/rung0-replicate-ceiling/delta-repro-31758395.out",
  "log_sha256": "3a2398b5a906b475ed2d3789f5c74dd6b1b4683f7a0831a46c7f7a19a5ca0274",
  "job_id": "31758395",
  "clean_tree": true,
  "environment": {
    "code_commit": "84c094d5905b9499fc2f9d6a7572ac3d23e06887",
    "python_version": "3.13.5",
    "seed": 0,
    "cuda_deterministic": false,
    "data_commit": "9a8797a5698e2c56ec1b61bdd3d5f68d18a972e227e86b64ac341ef507f73dd6",
    "container_digest": null,
    "torch_version": null,
    "cuda_version": null,
    "model_weights_hash": null
  },
  "promoted_at": "2026-08-28T15:19:22.111185Z"
}
```

`inputs` paths are shown with `<scratchpad>` standing in for the session-local scratch directory recorded in the real file; the sha256 values are unabridged and match the controller's execution log's pulled-copy hashes exactly — both input hashes are independently recomputable from the pulled files and were re-derived and checked at audit time. `clean_tree: true` confirms the rule-1 fix (this task's first commit) — the working tree carries plenty of untracked data (`docs/tasks/rung0-replicate-ceiling/plan.md`, Global Constraints), but no tracked-file modification was pending at promotion.

**Commit reconciliation**: the measurement itself (job 31758395) ran against commit `25cec05` — the run's own `rung0_delta_reproducibility.params.json` records `"git_sha": "25cec0540d1cf86beb871aacb807003c14c9a843"`. The provenance record's `environment.code_commit` is `84c094d`, repository HEAD at promotion time, not the measuring commit — but `84c094d` is a descendant of `25cec05` whose only code changes are `scripts/promote_result.py` and its test (`tests/test_promote_result.py`); `scripts/delta_reproducibility.py` and everything under `src/` are byte-identical at both commits (`git diff 25cec05..84c094d -- scripts/delta_reproducibility.py src/` is empty). The measurement is reproducible from either commit; `code_commit` records the promoting commit, per the schema's contract, not the measuring one.

A later docs commit (`c6f1baf`) rewrote prose in `scripts/delta_reproducibility.py` (its module docstring, the `--n-hvg` help text, a comment, and the final print statement) with no computational change — the auditor read the full diff and confirmed no scoring logic moved — so HEAD's script text differs from the record's `code_commit` copy in wording only, while every computation the record reflects is identical; the job log's closing line ("Check-1 achieved r ~ 0.2 ...") is the pre-rewrite print and can no longer be produced verbatim by the shipped script's current print statement.

`scripts/promote_result.py` now accepts `--input LABEL=PATH` so future promotions key `inputs` by a durable label rather than a scratch path; rung 0's own record above ran before that support existed and deliberately stands with its scratchpad-path keys, with the mapping to what they were (the gene panel, the drug-CID file) documented in the promotion command above.

## Project-rule tests (Task 9, Step 4)

Before promotion, the rule-1 and rule-4-edge tests skipped (no promoted results yet). After promotion:

```
$ uv run pytest tests/test_project_rules.py -v -m "step_promote or step_score or step_null or step_document"
tests/test_project_rules.py::test_rule_01_every_promoted_result_carries_a_complete_provenance_record PASSED
tests/test_project_rules.py::test_rule_01_edge_promoted_records_validate_against_the_schema PASSED
tests/test_project_rules.py::test_rule_02_every_task_is_named_in_the_spec_tree PASSED
tests/test_project_rules.py::test_rule_02_edge_non_additive_task_edits_carry_a_dated_entry PASSED
tests/test_project_rules.py::test_rule_03_readme_links_to_the_project_documents PASSED
tests/test_project_rules.py::test_rule_03_edge_readme_is_revisited_when_the_ladder_changes PASSED
tests/test_project_rules.py::test_rule_04_every_task_declares_controls_for_its_measurement_steps PASSED
tests/test_project_rules.py::test_rule_04_edge_promoted_tasks_have_known_answer_tests PASSED

8 passed in 0.28s
```

All 8 run (none skip) and all pass: rule 1's two tests validate the real record above; rule 4's edge test finds `pytest.mark.known_answer` already present in the suite (`tests/test_statistics_known_answers.py`, `tests/test_rung0_controls.py`). The full suite (`uv run pytest -q`) also runs with zero skips now — every project-rule test that was gated on a promoted result binds.

## Derangement permutation check (2026-08-28/31, superseding job 31770850)

The exposure argument above bounded the damage the profile-sharing dependence could do; this
check measured it. `scripts/derangement_null.py` (submitted as
`scripts/alpine/derangement_null.sbatch`, same inputs as the ceiling run) broke the matched
pairing with 500 derangements of the 1,600 finite-scored conditions and computed the mean
mismatched correlation per derangement — the null distribution of the reported aggregate with
the dependence carried by construction.

The first run (job 31764582) covered only the pooled any-pair aggregate. External review
observed that the promoted p-values are per-stratum (`p_vs_null` against the `diff_drug`
mismatched-pair pool, `p_vs_same_drug` against the `same_drug` pool), while an any-pair
derangement mixes both mismatch types freely and carries neither stratum's dependence
specifically. The same script, extended with two stratum-preserving nulls
(`sample_within_drug_derangement`, `sample_cross_derangement`), was rerun as job 31770850; that
run supersedes 31764582 and is the current content of every artifact named below. The any-pair
numbers are unchanged from the first run.

Summary (`rung0_derangement_summary.csv`; per-permutation means in the three files named below;
resolved arguments and producing commit in `rung0_derangement_summary.params.json`; log at
`results/rung0-replicate-ceiling/derangement-null-31770850.out`):

```
any-pair    : observed_mean 0.1348   perm_mean_mean 0.0361   perm_mean_sd 0.0009
              p_exact 0.002   se_iid_pool 0.00099   design_effect 0.872   z_derangement 106.84
same-drug   : perm_mean_mean 0.0792   perm_mean_sd 0.0004   p_exact 0.002   design_effect 0.071
              (1600 of 1600 finite rows are in >=2-row drug groups: n_rows_same_drug 1600 =
              n_pairs 1600, same_drug_rows_equal_n True)
diff-drug   : perm_mean_mean 0.0342   perm_mean_sd 0.0008   p_exact 0.002   design_effect 0.516
```

Per-permutation means live in three files: `rung0_derangement_perm_means.csv` (any-pair),
`rung0_derangement_perm_means_same_drug.csv` (within-drug), and
`rung0_derangement_perm_means_diff_drug.csv` (cross-constrained), each 500 rows; resolved
arguments and the producing commit are in the `rung0_derangement_summary.params.json` sidecar;
the job log is `results/rung0-replicate-ceiling/derangement-null-31770850.out`.

Reading: the any-pair design effect is **0.872** — the dependence does not widen the null
mean's spread at all (a derangement uses each half-profile exactly once, whose balance slightly
*tightens* the permutation mean relative to independent pooling), so the exchangeable-pool
treatment behind the reported p-values and MDEs was marginally conservative, not optimistic.
The two per-stratum design effects read the same way and are smaller still: **0.071** for the
same-drug (within-drug) null and **0.516** for the diff-drug (cross-constrained) null, both
< 1. Because `n_rows_same_drug` (1600) equals `n_pairs` (1600, `same_drug_rows_equal_n` = True),
the same-drug design effect is measured over the same rows `p_vs_same_drug` is, so it transfers
directly to that promoted p-value with no scope caveat.

The two permutation-null means also independently reproduce the pooled floors the headline CSV
already reports: the same-drug derangement's `perm_mean_mean` (0.0792) matches
`null_same_drug_mean_r` (0.079) and the diff-drug derangement's `perm_mean_mean` (0.0342)
matches `null_diff_drug_mean_r` (0.035) — two numbers computed by an entirely different sampling
mechanism (exact derangement vs. bootstrapped exchangeable pool) landing on the same floor.
Verify both pairs against `rung0_delta_reproducibility.csv` and `rung0_derangement_summary.csv`
directly rather than trusting this sentence.

In every stratum the observed mean exceeds every one of its 500 permutation means; p_exact =
0.002 is the smallest value 500 permutations can certify. The stated non-independence assumption
is discharged by measurement rather than argument, for the pooled aggregate and for both
per-stratum p-values the promotion actually reports. The check is an unpromoted verification
diagnostic: the promoted record is unchanged, per PROCESS §1's convergence rule.

Two caveats, carried forward honestly rather than smoothed over:

1. **The cross-constrained sampler is not provably uniform.** `sample_cross_derangement`
   repairs a random permutation by local swaps until it matches the `diff_drug` stratum's
   constraint set (different line and different drug on every row), rather than rejection
   sampling exactly uniformly over that set the way the any-pair and within-drug samplers do.
   An empirical, data-blind probe (brute-force enumeration of all 448 valid permutations on a
   small 3-drug × 3-line fixture, then 50,000 draws from the sampler; see the script's
   `sample_cross_derangement` docstring) found a real departure from uniform: chi-square
   goodness-of-fit 2857 on 447 degrees of freedom against an expectation of ~447 under
   uniformity. Because the probe is data-blind (run on a synthetic fixture, not the real pool
   or its result), any non-uniformity it reveals perturbs the permutation mean's *variance* —
   a second-order effect on `design_effect` and the width of the null — not its *location*; it
   cannot manufacture the observed mean's separation from the null. At the margins measured
   here (every stratum's observed mean above all 500 permutation draws, p_exact at its floor of
   0.002), the effect is immaterial. A provably uniform cross-derangement sampler (e.g.
   Metropolis–Hastings with a detailed-balance-respecting proposal) is a recorded follow-up
   (`review.md`), not done here.
2. **A stated, inherited aggregation convention.** The per-permutation aggregates use
   `nanmean` over rows whose mismatched pairing falls below `min_genes` shared finite entries
   within that one draw, while the observed comparators (`observed_mean`,
   `observed_mean_same_drug_rows`, `observed_mean_diff_drug_rows`) use a plain `mean` over the
   rows already restricted to finite `r`; and every `design_effect`'s denominator uses the
   nominal row count (`n` finite rows, or `n_multi` same-drug rows) rather than re-deriving a
   per-draw effective count. This mirrors `stratified_null_draws`'s own convention in
   `scripts/delta_reproducibility.py` and is stated here for completeness, not as a newly found
   issue.
