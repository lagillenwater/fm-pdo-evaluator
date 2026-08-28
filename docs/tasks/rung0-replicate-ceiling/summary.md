# Rung 0, in plain language

**As of** 2026-08-28. The technical record is [`design.md`](design.md) (what and why),
[`verification.md`](verification.md) (the run and its evidence), and [`review.md`](review.md)
(what review found). This page says what was asked, what was found, and what it means.

## The hypothesis

When a cancer cell line is treated with a drug, its gene expression shifts. Rung 0 asks whether
that measured shift is reproducible at all — whether the Tahoe-100M screen's response profiles
contain real signal above assay noise, and how much. Formally: split each (cell line, drug)
condition's replicate plates into two halves, and the two halves' expression-response profiles
should agree with each other more than profiles from mismatched conditions do. If they do not,
nothing upstream of the assay could ever be predicted, and every later rung of the evaluation
would be chasing noise.

The number this produces is a **ceiling**: the best agreement any prediction of these responses
could achieve, because a prediction cannot agree with the measurement more than the measurement
agrees with itself. Every later rung is read as a fraction of this ceiling rather than of a
hypothetical 1.0.

## What was measured

For each of 1,600 (cell line, drug) conditions — 50 lines by 32 drugs, the drugs shared with
the GDSC2 viability screen — the replicate plates were split deterministically in half. Each
half's per-gene log2 fold change (treated vs plate-matched DMSO) was averaged, and the two
half-profiles were correlated across the declared 14,121-gene panel (13,886 present in the
data). The headline is the mean of those per-condition correlations, lifted by Spearman-Brown
to estimate full-data reliability.

## The evidence

Headline (promoted: [`results/rung0-replicate-ceiling/`](../../../results/rung0-replicate-ceiling/rung0_delta_reproducibility.csv), with its provenance record beside it):

| Quantity | Value |
|---|---|
| Conditions scored (cell line × drug) | 1,600 |
| Panel genes present | 13,886 of 14,121 |
| Split-half reliability, mean over conditions | **0.135** (median 0.109; quartiles 0.071–0.155) |
| Spearman-Brown full-data ceiling | **0.238** |
| Conditions with positive reliability | 98.9% |
| Mismatched-condition floor (different line and drug) | 0.035 |
| Same-drug floor (same drug, different line) | 0.079 |
| Significance vs both floors | p = 0.0005 |
| Smallest detectable effect at 80% power (vs each floor) | 0.039 / 0.085 |

The observed 0.135 sits roughly 3.4 times above what the experiment could have detected, so the
result is not a power artifact — and the same two power columns will report honestly at the
organoid rung, where small cohorts make power the whole question.

Controls (every one passed; details in `tests/` and `verification.md`):

| Control | Sign | Result |
|---|---|---|
| Synthetic replicate pool with a planted reliability of 0.8, through the real code | positive | recovered (0.796–0.800 across runs) |
| Synthetic pool with no planted signal | negative | correctly null (p > 0.05) |
| Planted drug-shared plus line-specific structure | positive | recovers the expected ordering: matched > same-drug floor > mismatched floor |
| Pure-noise pool | negative | all floors at zero |
| Real data, conditions split by effect size | positive | reliability rises with effect size: 0.100 / 0.126 / 0.178 across terciles |
| Power calculation vs the exact normal-theory answer | known answer | matches within 0.005 |
| Data integrity: all 1,026 downloaded data shards re-hashed | integrity | every hash matches the value recorded at download time |

Figure: [`rung0_ceiling.png`](rung0_ceiling.png) — the distribution of per-condition split-half
correlations against both mismatched-condition floors, with the headline mean marked.

## Conclusions

1. **The target is reproducible, but modestly.** About a quarter of the full-data response
   profile (Spearman-Brown 0.238) is stable signal on this gene panel and drug set; the rest is
   assay and biological noise. The signal is real — thirty-plus standard errors above the
   mismatched floor — but the ceiling is low.
2. **Every later score must be read against 0.238, not 1.0.** A model that correlates 0.2 with
   these responses has captured most of what is capturable; reported alone, the same 0.2 would
   look like failure.
3. **Reliability concentrates where responses are large.** The strongest third of conditions is
   nearly twice as reliable as the weakest third, so later rungs' successes and failures should
   be examined for effect-size dependence before other explanations.
4. **The measurement is worth trusting mechanically.** The number carries a provenance record
   whose every hash can be re-derived from the committed artifacts, the input data is pinned by
   a content-hashed registration, and each step of the computation recovers planted answers.

One stated assumption: the mismatched-condition floors reuse the same half-profiles across
comparisons, so their draws are not fully independent; the p-values treat them as an
exchangeable pool. This is inherited from the measurement design and stated rather than solved
at rung 0.

## Scripts this task touched

Measurement and statistics: `scripts/delta_reproducibility.py` (the split-half measurement),
`src/fmharness/statistics.py` (shared significance, power, and Spearman-Brown helpers).
Provenance machinery: `scripts/register_tranche.py` (content-hashed data registration),
`scripts/promote_result.py` (promotion with a schema-validated record).
Data acquisition: `scripts/download_tahoe_pseudobulk_de.py`, `scripts/alpine/00_target_cids.sbatch`,
`scripts/alpine/01_pseudobulk_shortcut.sbatch`.
Cluster jobs and access: `scripts/alpine/delta_reproducibility.sbatch`,
`scripts/alpine/register_tranche.sbatch`, `scripts/alpine/ralpine`.
Tests: `tests/test_statistics_known_answers.py`, `tests/test_rung0_controls.py`,
`tests/test_promote_result.py`, `tests/test_register_tranche.py`, and additions to
`tests/test_project_rules.py`.
