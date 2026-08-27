# Transfer ladder protocol

How far a prediction survives as the test set moves away from the training distribution, one
shift at a time. Each rung adds exactly one source of distribution shift, so a collapse can be
attributed to that shift rather than to the accumulated difference between cell lines and
patient organoids.

The design exists because of a measured result: L1000 and Tahoe deltas for the SAME
(cell line, drug) agree at Spearman **0.041** against a split-half ceiling of **0.572** -- 7% of
achievable. That 0.041 is real, not noise: corrected for an aggregate-vs-per-item p-value bug
found 2026-08-26, it clears its mismatched-pair null (p = 0.0005) and so does every
normalisation tried, though none closes the gap to ceiling [job 31676845,
`docs/results/l1000_tahoe_transform_sweep.csv`]. Sign concordance sits at 51.3%, not
distinguishable from the 50% chance rate (p = 0.0515). If two cell-line drug-perturbation
datasets collapse to 7% of their own reproducibility on the SAME perturbations, a single Path B
number built by transferring across that gap has no interpretable scale. The ladder supplies
the scale.

## The rungs

| rung | test cohort | modality | target | shift added | ceiling |
|---|---|---|---|---|---|
| 0 | Tahoe, same lines | cell line | delta | — (replicate split) | is the ceiling |
| 1 | Tahoe, held-out line | cell line | delta | new cell line | rung 0 |
| 2 | Tahoe, map fit on L1000 | cell line | delta | new platform | rung 0 |
| 3 | GDSC2 | cell line | viability AUC | new readout + screen cohort | screen agreement |
| 4 | sarcoma organoids (embargoed) | organoid | viability | new modality + tissue | **none — see below** |

Rung 1 is Check 1/1b done on the common panel with both checkpoints; rung 3 is Check 2 done the
same way. Neither is a new experiment. Rungs 0 and 2 are diagnostics. Rung 4 is the only new
build.

**Rungs 0–2 score deltas; rungs 3–4 score viability.** That discontinuity is structural, not an
oversight: Tahoe has no viability labels and the organoids have no treated RNA-seq. The two
halves answer different questions — 0–2 diagnose *where* representation transfer breaks, 3–4
reach the goal — and the claim joining them, that a representation surviving rung 2 should
survive rung 3, is itself testable and currently untested.

## Invariants

Every rung holds these identical, or its number cannot be placed on the same axis as another's.
This is not hypothetical: the existing 0.46 delta ceiling is Spearman-Brown-corrected over
top-HVG genes while the 0.572 split-half is uncorrected over 978 landmarks, and putting them
side by side would be meaningless.

1. **Gene panel** — the common panel (14,121 at Check 1; 12,368 with GDSC2), built by
   `common_gene_panel()` and verified by `assert_common_genes()`.
2. **Metric — CORRECTED 2026-08-26, was wrong.** This line previously claimed universal
   Spearman. That was never true and was never the intent: the two specs that predate this one
   and describe the same delta-fidelity code (`2026-08-07-modular-harness-core-design.md`,
   `2026-08-18-stack-faithful-generation-and-de-metrics-design.md`) both correctly and
   consistently say Pearson for rungs 0/1 (matching `evaluation.py`'s `_row_corr` and
   `delta_reproducibility.py`'s `np.corrcoef`, which is what actually produced every promoted
   number at those rungs). Rung 2 alone is Spearman (`rung2_score_one.py`,
   `stats.spearmanr` — a deliberate choice given L1000/Tahoe's heavy-tailed cross-platform
   noise). The viability rungs (3/4) use the two-way-demeaned Spearman interaction rho via
   `evaluation.py`'s `score_predictions`. **This means a rung-0-vs-rung-1 ratio is fine (both
   Pearson) but a rung-0-vs-rung-2 or rung-1-vs-rung-2 ratio is not yet valid** (see
   `docs/PROJECT_STATE.md`, rung 0) — the metric mismatch, not just the reliability-
   correction one below, blocks it. Fix by explicitly stating Pearson-for-delta-rungs /
   Spearman-for-viability-rungs as the real rule (this note), or by switching rung 0/1 to
   Spearman to match rung 2 — not decided; either closes the gap, leaving it open is what does
   not.
3. **Unit** — one value per (line, drug), then **the SAME aggregation statistic** over pairs —
   currently violated: rung 0's headline (`splithalf_median_r`, `spearman_brown_full`) is the
   MEDIAN over pairs, while rungs 1/2/3 report the MEAN
   (`f["r"].mean()`/`r.mean()`/`score_predictions`'s per-pair average). `splithalf_mean_r` is
   already computed and sitting in `docs/results/rung0_delta_reproducibility.csv`, unused as the
   headline — switching to it (or switching the other rungs to a median) is the fix. Never a
   pooled correlation, regardless.
4. **Reliability correction** — Spearman-Brown everywhere or nowhere, stated per rung.
5. **CV scheme** — **5-fold over cell lines everywhere**, from the single shared partition in
   `fmharness.deltas.fold_assignment`. This was missing from the invariants and the rungs had
   drifted: rung 1 and rung 2's in-platform arm ran leave-one-line-out while rung 3 ran 5-fold,
   so a cross-rung ratio mixed a bias/variance difference into what was meant to be a transfer
   effect. The folds must be the same PARTITION, not merely the same count — two rungs each
   writing `i % n_folds` agree only until one iterates its lines in a different order, so the
   helper sorts first. LOO is the eventual target and needs no second code path: `n_folds >=`
   the line count degenerates to it.
6. **Null** — mismatched-pair permutation, recomputed *within* each rung and each transform.
7. **Reporting** — fraction of that rung's own ceiling. Raw numbers only alongside the fraction.

## Baselines, models and controls per rung

Shared vocabulary, from `data/model_matrix.yaml`:

| id | kind | what |
|---|---|---|
| `prior` | control | one constant feature — per-drug intercept only. The true line-independent floor; scores interaction exactly 0.000 |
| `knn` | baseline | mean delta of the k most similar other lines by baseline expression |
| `pca` | baseline | ridge on PCA components of the baseline, predicting the delta residual |
| `nmf` | baseline | as `pca`, non-negative factorisation |
| `expr` | representation | untreated baseline expression. No drug information |
| `base` | model | per-line embedding from the UNALIGNED `bc_large` encoder |
| `aligned` | model | per-line embedding from the cytokine-aligned encoder |
| `stack_cytokine` | model | generated delta, cytokine-aligned checkpoint |
| `stack_drug_aligned` | model | generated delta, sci-Plex drug-aligned fine-tune |
| `observed_delta` | baseline | each drug's mean measured delta over the OTHER lines — the drug-level average of observed deltas. Formerly `additive`, a name that described neither what it computes nor what it is for. Standardised, it is `measured_delta` sign-flipped (corr −1.000000), so the two are reported together and neither is read as an independent floor |
| `measured_delta` | reference | the real delta as its own prediction — best-case input, not a positive control |
| `planted` | control | a known interaction planted at controlled effect size. Must be recovered |
| `*_random` | control | per-representation noise control, one per representation |

**Rung 0** — no predictors. It is a property of the data: plates split by `hash(plate) % 2`,
each half aggregated to a delta, halves correlated, Spearman-Brown to full data.

**Rung 1** (delta, held-out Tahoe line, LOO/5-fold over 50 lines)
- Floor: `prior`
- Baselines: `knn`, `pca`, `nmf`
- Models: `stack_cytokine`, `stack_drug_aligned`
- Reference: `measured_delta` · Controls: `planted`, `*_random`
- Nulls: `within_drug` (the test of line specificity) and `shuffle_all` (reported, but every
  source clears it at the p floor, so it is uninformative — [job 31660552])

**Rung 2** (delta, Tahoe test set, maps fit on L1000)
- Baselines refit on L1000: `knn`, `pca`, `nmf`
- The number that carries the rung is **rung-2 minus rung-1 per baseline** — the transfer
  penalty — not the rung-2 score alone.
- **Stack does not face this rung's shift, and the table must say so.** The shift is "your
  training data came from a different platform", which only applies to a predictor that gets
  fitted. Stack takes a query baseline plus a drug and generates from frozen pretrained
  weights: same input, same weights, same output at rung 1 and rung 2. Its rung-2 score is
  therefore *numerically identical* to its rung-1 score. If Stack is tabulated beside baselines
  that just paid a transfer penalty and appears to win, that is a category error, not a result.
  Report it as a fixed reference line labelled "unchanged from rung 1 by construction".
- **Stack's actual rung-2 arm, per decision D3 (`docs/decisions/2026-08-25-ladder-round.md`):**
  rebuild Stack's CONTEXT — its in-context learning corpus — from L1000 instead of Tahoe, query
  baseline held at Tahoe throughout. D3 explicitly rejected swapping the query baseline (below)
  as "wrong in a subtler way: it shifts what the model is conditioned on, while the baselines'
  arm shifts where their map was learned." Built by `scripts/alpine/31_l1000_context_tahoe.sbatch`
  + `32_shard_l1000_context.sbatch` (`l1000_context_by_drug`), scored SEPARATELY from the fitted
  baselines' grid — it answers "does Stack's generation change when its examples come from a
  different platform", which is the platform-shift analogue for a model that is never fitted.
  Known limitation: only 15 of Tahoe's 33 drugs matched L1000 by PubChem CID, so this is a
  14-drug comparison (one CID collision) against the baselines' wider one.
  **Scored** [job 31678008, `docs/results/rung2_l1000_context_generation.csv`, via
  `scripts/score_l1000_context_generation.py`]: 618 shared (line, drug) pairs, both checkpoints
  null (`stack_cytokine` r=-0.001, `stack_drug_aligned` r=0.011), matching rung 1's Tahoe-context
  result (r=0.018/0.040) within noise. Stack's generation is null regardless of which platform
  its in-context examples come from — the rung-1 failure is not explained by a context/platform
  mismatch.
- A DIFFERENT, narrower measurement — give Stack the L1000 DMSO profile of line X as its query
  baseline instead of Tahoe's untreated profile, still scored against the Tahoe delta — shifts
  Stack's INPUT rather than its context, and answers a different question again. D3 considered
  and rejected this as Stack's PRIMARY rung-2 arm; if run at all it must be labelled separately
  from both the context-swap arm above and the baselines' penalty, never placed in the same
  column as either. The inputs exist: L1000 has DMSO wells for all 7 shared lines.
- Worth stating in the writeup: needing no refit for a new platform is a real deployment
  advantage of a pretrained model. It is simply not a score on the same task.

**Rung 3** (viability, GDSC2 AUC, 5-fold over lines)
- Floor: `prior` · Representations: `expr`, `base`, `aligned`
- Baselines: `knn`, `pca`, `nmf`
- Models: `stack_cytokine`, `stack_drug_aligned`
- Reference: `measured_delta` · Controls: `planted`, `*_random`
- Model class: penalized regression (L1 / L2 / elastic net) per the representation-controlled
  grid, alpha path `np.logspace(-2, 8, 24)` with `alpha_is_interior` checked — 77.3% of fits
  previously sat at the old path ceiling

**Rung 4** (viability, organoids, embargoed)
- Same list as rung 3 **minus `measured_delta`** — no treated-organoid RNA-seq exists, so the
  best-case-input row is unavailable. `planted` remains, being synthetic.
- Baseline maps fit on cell lines and applied to organoid baselines; Stack generates from the
  organoid baseline.

## Rung 4 feasibility — measured [job 31663218]

| question | answer |
|---|---|
| organoids drug-screened | 94 |
| organoids with expression | 17 |
| **usable n (both)** | **17** |
| drugs per usable organoid | min 5, **median 17** of 34 |
| replicate/dose granularity anywhere in the tranche | **none** |
| organoid drugs covered by L1000 | 25 / 34 |
| organoid drugs covered by GDSC2 | 21 / 34 |
| covered by both / either | 19 / 27 |

Two consequences.

**Rung 4 has no ceiling, and this is now verified rather than assumed.** All seven tables were
scanned; none carries a dose or replicate column, and `drug_screen.parquet` holds exactly one
row per (organoid, drug) across 1,350 rows. There is no test-retest reliability to compute. So
rung 3 is not merely the previous rung — it is rung 4's *only* reference frame, and rungs 3 and
4 must share readout, metric, predictor family and CV scheme exactly. **No rung-4 result may be
stated as an absolute value**; the honest form is always "X% of what the same method achieves on
cell lines".

**The design is unbalanced.** 17 organoids at a median of 17 drugs is roughly 290 (organoid,
drug) pairs, in the same range as the corrected Check 2 grid, so the interaction axis is
powered comparably — but the drug axis must be reported per organoid, not assumed full.

## Decision: GDSC2 supplies rung 4's drug axis

L1000 covers more organoid drugs (25 vs 21), and GDSC2 is still the right choice, because rung
3 is rung 4's only reference frame. GDSC2 already supplies rung 3's labels, so taking rung 4's
drug axis from GDSC2 keeps both rungs on **the same drugs**, and a shared drug axis is worth
more than four extra drugs when the entire interpretation of rung 4 is a ratio against rung 3.
Choosing L1000 would buy coverage and give up the comparison the rung exists to support.

The 19 drugs covered by *both* remain available as a controlled sub-analysis, letting rung 4 be
scored under either representation source on an identical drug set.

## Ceilings and their provenance

| ceiling | value | status |
|---|---|---|
| rung 0 — delta reproducibility | 0.109 split-half median / 0.197 Spearman-Brown | **promoted** [job 31676846], on the SAME 14,121-gene panel as rung 1 (`docs/results/rung0_delta_reproducibility.csv`). Supersedes the earlier 0.30/0.46 figure, which was computed on an unpinned top-2000-HVG panel and cannot denominate rung 1. |
| rung 3 — screen agreement | 0.47 (GDSC2↔CTRP), 0.31 (GDSC2↔PRISM) | **promoted** [job 31663627] (`docs/results/rung3_label_ceiling.csv`) |
| rung 4 | none exists | verified [job 31663218] |

**Open issue, not yet reconciled:** rung 1's baselines (knn/pca/nmf, mean_rho 0.28-0.32) now
numerically EXCEED the correctly-panelled rung-0 ceiling (0.109-0.197). This is not evidence a
baseline beats reproducibility itself -- it is the still-unresolved invariant-2/invariant-3
mismatch: rung 0 aggregates by MEDIAN and scores Pearson, rung 1 aggregates by MEAN and also
scores Pearson (rung 2 alone scores Spearman). A right-skewed per-pair distribution has a mean
well above its median, and that alone could close most of the gap. **Do not report a rung-1
"fraction of ceiling" until this is fixed** — either rung 0's headline switches to
`splithalf_mean_r` (already computed, just not the reported statistic) or rung 1 switches to a
median, and the two must agree before division means anything.

## Artifacts

Every rung emits a result table plus a `.provenance.json` carrying the git sha, job id, resolved
arguments, hashed inputs and the run log's hash, via `scripts/promote_result.py`. Embargoed
rows stay out: `organoid_cohort.csv` is deliberately unpromoted because its `SARC####`
identifiers are patient-derived and absent from the public cell-line registry, which
`check_release.py` verifies per value rather than per declaration.
