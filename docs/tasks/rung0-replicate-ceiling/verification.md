# Rung 0 — verification

**Task** `rung0-replicate-ceiling` · **As of** 2026-08-28
Records how the headline number in `docs/tasks/rung0-replicate-ceiling/rung0_delta_reproducibility.csv` was produced and promoted, and every number this document cites is copied from that CSV or from the job log, not from memory.

## Commands as run (Task 8 — deploy, register, measure)

Run by the controller on 2026-08-27/28, against `docs/tasks/rung0-replicate-ceiling/plan.md`'s Task 8 steps; commands below are those steps with the job ids substituted from `.superpowers/sdd/plan/task-8-facts.md`.

```bash
git push
./scripts/alpine/ralpine switch rung0-replicate-ceiling      # ran at b5da999 (registration), 25cec05 (measurement) after a ff
./scripts/alpine/ralpine run ls "$SCRATCH/tahoe_pseudobulk_de/pseudobulk_differential_expression" | head -3
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

One transient SSH failure (Permission denied — network) occurred on 2026-08-27 during this sequence; it resolved itself on retry, no re-auth needed (`task-8-facts.md`).

## Job log — summary block (tail of `results/rung0-replicate-ceiling/delta-repro-31758395.out`)

```
splitting plates by hash(plate) % 2
scoring the ceiling on the supplied panel: 13886 of 14121 genes present
null[any_pair  ] median r = +0.034 over 500 draws
null[diff_drug ] median r = +0.034 over 500 draws
null[same_drug ] median r = +0.067 over 500 draws

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

## Pool description (from `rung0_pool_description.csv`)

The pool-description table lists 1,650 (line × drug) rows: 50 cell lines by 33 drug-name entries in the DE table matching the target-CID panel (32 drugs, plus a `Trametinib (DMSO_TF solvate)` name variant). All 1,650 rows carry 3 dose levels.

- 1,210 of 1,650 rows (the majority) have exactly 3 plates, split 1 plate in half0 / 2 in half1 — the typical case.
- 290 rows have 4 plates (2/2), 50 have 6 (2/4), 50 have 7 (3/4) — drugs profiled on more plates (e.g. Afatinib, Trametinib, Cytarabine, Docetaxel, Rapamycin, Retinoic acid, Temsirolimus).
- 50 rows — all 50 lines of Ribociclib, and only Ribociclib — have exactly 1 plate total (1 in half0, 0 in half1): a single plate cannot be split, so these pairs are unscoreable.
- 1,650 candidate rows − 50 unscoreable Ribociclib rows = 1,600, matching `n_pairs` in the headline CSV exactly.

## Tercile control

`splithalf_mean_r_tercile1/2/3` = 0.100 / 0.126 / 0.178 — monotonically increasing across pairs stratified by effect-size tercile. Per the design's declared positive control (design.md, "Controls and power"), this is the pass condition: an assay that could not find more reproducibility where there is more signal would be broken. The in-run positive control passed.

## Minimum detectable effect

- `mde_80_vs_diff_drug` = 0.0392 — the smallest split-half mean r this run (n = 1,600 pairs) could distinguish from the diff-drug (generic-structure) null floor at α = 0.05, power = 0.80. The observed 0.135 clears it by a wide margin (~3.4×).
- `mde_80_vs_same_drug` = 0.0846 — the corresponding MDE against the stricter same-drug (line-specificity) null floor; the observed 0.135 clears this one too (~1.6×), the tighter of the two comparisons.

Both MDEs are far below the observed ceiling — at n ≈ 1,600 pairs this run is trivially powered, which is expected and stated in the design (design.md, "Controls and power"): the same reporting will matter honestly at rung 5, where power is the whole question.

## Write-up caveat (ledgered at Task 3 review)

The stratified null draws reuse the same half-profiles across mismatched pairs, so they are not i.i.d., while the bootstrap treats them as an exchangeable pool; this is the residual assumption behind the reported p-values and MDEs, inherited from the archived lineage's design and stated rather than solved at rung 0.

## GDSC2 CID list — provenance note

`00_target_cids.sbatch` (which built `data/static/tahoe_target_cids.txt`, one of this promotion's two declared inputs) reads `data/static/gdsc2_auc_pubchem_cids.txt` on Alpine. That file is untracked on Alpine and not landed in this repository; neither `00_target_cids.sbatch` nor `01_pseudobulk_shortcut.sbatch` is re-run by this task — their outputs exist already and are pinned by hash (the CID file) and by the tranche (the DE pool). `data/static/gdsc2_auc_pubchem_cids.txt` becomes a tracked input in this repository when rung 4 registers GDSC2.
