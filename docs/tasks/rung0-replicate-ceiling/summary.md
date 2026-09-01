# Rung 0, in plain language

**As of** 2026-08-31. The technical record is [`design.md`](design.md) (what and why),
[`verification.md`](verification.md) (the run and its evidence), [`review.md`](review.md)
(what review found), and [`decisions.md`](decisions.md) (the dated decision lineage). This page
says what was asked, what was found, and what it means.

**Check it yourself before reading further.** Every number on this page recomputes from files
committed in this repository — no cluster access needed, and no trust in this write-up: run
`uv run python scripts/verify_rung0.py` (48 checks, about a minute, PASS/FAIL per claim), or
open [`verify.ipynb`](verify.ipynb) for the same checks with a plain-language explanation of
what each one proves. How this task was checked, in one line each: the *code* was reviewed per
task (`review.md`), the *run* was evidenced with commands and output (`verification.md`), the
*claims* were audited clause-by-clause against what actually landed (`audit.md`), and the
*numbers* recompute mechanically (the battery above, run continuously by the test suite).

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

For each of 1,600 (cell line, drug) conditions scored — 50 lines by 31 of the 32 declared drugs
(shared with the GDSC2 drug-sensitivity screen — Genomics of Drug Sensitivity in Cancer, release 2) plus one drug's alternate name, a solvate variant of
Trametinib — the replicate plates were split deterministically in half. The 32nd declared drug,
Ribociclib, has only a single plate throughout the pool and cannot be split, so it scores zero
conditions and contributes nothing to the 1,600. Each
half's per-gene log2 fold change (treated vs plate-matched DMSO — dimethyl sulfoxide, the drug-free solvent control) was averaged, and the two
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

The observed 0.135 is roughly 3.4 times the mismatched-condition detection threshold (0.039)
and roughly 1.6 times the stricter same-drug detection threshold (0.085) — the experiment was
not underpowered to find either floor, so the result is not a power artifact — and the same two
power columns will report honestly at the organoid rung, where small cohorts make power the
whole question.

Controls (every one passed; details in `tests/` and `verification.md`):

| Control | Sign | Result |
|---|---|---|
| Synthetic replicate pool with a planted reliability of 0.8, through the real code | positive | recovered (0.800-0.809 across seeds; tolerance plus/minus 0.05) |
| Synthetic pool with no planted signal | negative | correctly null (p > 0.05) |
| Planted drug-shared plus line-specific structure | positive | recovers the expected ordering: matched > same-drug floor > mismatched floor |
| Pure-noise pool | negative | all floors at zero |
| Real data, conditions split by effect size | positive | reliability rises with effect size: 0.100 / 0.126 / 0.178 across terciles |
| Power calculation vs the exact normal-theory answer | known answer | matches within 0.005 |
| Data integrity: all 1,026 downloaded data shards re-hashed | integrity | every hash matches the value recorded at download time |

Figure: [`rung0_ceiling.png`](rung0_ceiling.png) — the distribution of per-condition split-half
correlations against both mismatched-condition floors, with the headline mean marked. The
underlying per-condition values are committed as `rung0_per_pair_r.csv` (one row per condition:
cell line, drug, genes scored, effect size, correlation), so every statistic in the table above
recomputes from the raw numbers — [`verify.ipynb`](verify.ipynb) draws that distribution and
recomputes them in front of the reader.

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

One caveat, checked and cleared. The floors above come from deliberately mismatched
comparisons — one condition's first half correlated against a different condition's second
half. With 1,600 conditions there are only so many halves to go around, so the same halves are
reused across many of those mismatched comparisons, which makes the comparisons partly
dependent on one another; the significance calculations initially treated each one as a fresh,
independent draw. Could that shortcut have flattered the result? Arithmetic already said no:
the observed agreement sits about a hundred times further above the floor than the floor's own
uncertainty, so the reuse would have had to distort that uncertainty thousands-fold to change
the conclusion, and reuse this sparse cannot come close. Rather than leave it argued, we
measured it: shuffle which first half is paired with which second half — all 1,600 at once, no
condition keeping its true partner — recompute the average agreement, and repeat 500 times.
That rebuilds the floor's uncertainty with the reuse fully included. The measured distortion
factor came out at 0.87: the reuse does not widen the floor's uncertainty at all, and if
anything the shortcut had slightly overstated it, making the reported significance cautious
rather than generous. The real, correctly paired agreement exceeded all 500 shuffled versions
(p = 0.002, the strongest claim 500 shuffles can support). The caveat is settled by
measurement, not argument.

The same shuffle check was then repeated separately within each comparison type the promoted
numbers actually use, not just on the original all-conditions version — once shuffling lines
only within the same drug, and once shuffling only across different drugs and lines. In every
version the distortion factor stayed at or below one (0.87 for the original all-conditions
check, 0.52 for the across-drugs-and-lines version, and 0.07 for the within-drug version), with
the true pairing beating all 500 shuffles every time. So the shortcut was cautious in every
comparison the promoted numbers make, not just the one first checked.

## Scripts this task touched

Measurement and statistics: `scripts/delta_reproducibility.py` (the split-half measurement),
`src/fmharness/statistics.py` (shared significance, power, and Spearman-Brown helpers).
Verification: `scripts/verify_rung0.py` and [`verify.ipynb`](verify.ipynb) (the executable
claim-recomputation battery), `tests/test_verify_rung0.py` (the same battery in continuous
integration).
Provenance machinery: `scripts/register_tranche.py` (content-hashed data registration),
`scripts/promote_result.py` (promotion with a schema-validated record).
Data acquisition: `scripts/download_tahoe_pseudobulk_de.py`, `scripts/alpine/00_target_cids.sbatch`,
`scripts/alpine/01_pseudobulk_shortcut.sbatch`.
Cluster jobs and access: `scripts/alpine/delta_reproducibility.sbatch`,
`scripts/alpine/register_tranche.sbatch`, `scripts/alpine/ralpine`.
Tests: `tests/test_statistics_known_answers.py`, `tests/test_rung0_controls.py`,
`tests/test_promote_result.py`, `tests/test_register_tranche.py`, and additions to
`tests/test_project_rules.py`.
