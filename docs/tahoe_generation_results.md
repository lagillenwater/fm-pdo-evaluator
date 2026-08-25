# Tahoe generation eval — results

**Question.** Does the Stack single-cell foundation model, run in **generation** mode on
Tahoe-100M drug perturbations, reproduce real drug-induced transcriptional changes — and
does that translate into predicting measured drug response (GDSC2 AUC) better than simple
baselines?

**Design.** Per (cell line, drug), the real Tahoe pseudobulk delta (treated − DMSO) and a
per-line control baseline. Every delta source is judged on equal footing:
- **Check 1 (generation quality, label-free):** per-(line, drug) delta-Pearson vs the real
  Tahoe delta, plus an off-diagonal correlation and a specificity rank (catch a source that
  is merely smooth). Ceiling = delta reproducibility.
- **Check 1b (2026-08-19, DE-restricted, label-free):** the same idea restricted to genes real
  single-cell Wilcoxon testing calls significantly changed, instead of the full dense profile —
  Spearman on the real-significant genes' LFC, PR-AUC, top-N overlap/Jaccard. Permutation-null
  significance (shuffle the pair label a predicted delta claims) in place of an off-diagonal
  correlation.
- **Gate:** the real delta scored through Hallmark vs random gene sets — is the readout even
  powered on this data?
- **Check 2 (end-to-end vs GDSC2 AUC, grouped 5-fold by cell line):** fixed signature readouts, and a
  representation-controlled penalized grid (expr + every delta source × L1/L2/EN). Ceiling =
  label reproducibility (independent viability screens).

Delta sources: `additive` (drug-mean, line-independent floor), `knn`, `pca`, `nmf` (baselines,
rebuilt leave-one-line-out), and `stack` (Stack-Large generated delta), evaluated on **three**
checkpoint variants: cytokine-aligned, drug-aligned-unfiltered, and drug-aligned-leak-excluded.

---

## Check 1 — generation quality (delta-Pearson vs real Tahoe delta)

| source | r | off-diag r | rank | pairs |
|---|---|---|---|---|
| additive | **0.225** | 0.095 | 0.885 | 1600 |
| nmf | 0.221 | 0.088 | 0.912 | 1600 |
| pca | 0.207 | 0.083 | 0.896 | 1600 |
| knn | 0.178 | 0.067 | 0.904 | 1600 |
| **stack (gen, cytokine-aligned)** | **0.012** | −0.002 | 0.644 | 1568 |
| stack (gen, drug-aligned, unfiltered) | 0.021 | 0.006 | 0.665 | 1568 |
| stack (gen, drug-aligned, leak-excluded) | 0.021 | 0.006 | 0.665 | 1563 |
| stack (gen, cytokine-aligned, faithful `--mode mdm`, 2026-08-19) | −0.001 | −0.005 | 0.538 | 1568 |
| stack (gen, drug-aligned, faithful `--mode mdm`, 2026-08-19) | 0.006 | −0.002 | 0.593 | 1508 |

**Pairs are not all from the same run.** Baseline rows (additive/knn/pca/nmf) are scored on the
unfiltered 1,600-pair design; the stack rows use Stack's own coverage (1,568), and the
leak-excluded stack row further drops the 5 doubly-exposed A549/drug pairs (1,563). None of this
changes any conclusion (~5/1300–1600 pairs is tiny) — noted here since it isn't visible from the
table alone.

**2026-08-19 — faithful generation procedure (Change 1), re-run.** The rows above (dated
2026-08-13 and earlier) used `--mode vanilla` with `prompt+context = 0.901` — a fixed-ratio
workaround adopted to dodge an `IndexError` at the old 50-row pseudobulk query baseline, not
Stack's own generative procedure. The new rows use the CLI's actual default, `--mode mdm`:
`context_ratio` scheduled over `linspace(0.2, 0.4, 5)`, confidence-guided selective unmasking
carried between steps, `n_test_cells` ranging 179–281 per step — fed by a genuinely larger query
pool (400 real single control cells, 8/line, replacing the old pseudobulk row) so the schedule's
padding never truncates. Per-query-cell replicates are then confidence-filtered
(`gen_logit < 0` — calibrated empirically against this Check-1 Pearson-Delta itself; see
Reproducibility) and averaged to one row per line before scoring, matching Stack's own generation
confidence classifier rather than a fixed literature threshold. **Both checkpoints still land in
the same null band as the vanilla-mode rows** (r = −0.001 cytokine-aligned, 0.006 drug-aligned) —
the faithful procedure does not recover the signal the workaround was suspected of suppressing.
See "Check 1b" below, though, for a metric where the two procedures disagree.

Ceiling (delta reproducibility, Tahoe plate split-half): **0.30 raw / 0.46 Spearman-Brown.**

**Stack generation is null.** r = 0.012 is essentially orthogonal to the real change
(off-diagonal ≈ 0, rank 0.64 ≈ random specificity) — far below even the line-independent
additive floor (0.225) and the 0.46 ceiling. The FM's generated delta carries no real signal
about the perturbation. (Baselines top out ~0.18–0.22; the *line-specific* part of the delta
is not recovered by any of them either — `additive`, which ignores the line, is the best.)
Drug alignment (fine-tuning the generation head on sci-Plex instead of cytokines) roughly
doubles Stack's Check-1 correlation vs. the cytokine-aligned checkpoint (0.012 → 0.021), but
both stay far below the additive floor (0.225) and the 0.46 ceiling, and leakage filtering does
not change the drug-aligned number (0.021 unfiltered and leak-excluded alike;
doubly_exposed_frac=0.003 on the ~5 doubly-exposed A549/drug pairs).

## Check 1b — DE-based metrics (2026-08-19, faithful generation only)

Pearson-Delta scores the full ~15,012-gene continuous profile, most of which is non-DE noise for
any given (line, drug) pair — a model could nail the genes that actually moved and still post a
near-zero `r` if it gets the other 14,900+ near-constant genes' fine structure wrong. `de_fidelity`
(`fmharness.evaluation`) asks a narrower, sparser question instead: restricted to genes
Wilcoxon-called significant in the real Tahoe single cells (`build_tahoe_de_calls`, LFC ≥ 0.25 /
FDR ≤ 0.05, Stack paper Methods 4.8's cell-eval threshold), does the predicted delta rank/flag the
right genes? Four metrics: `de_spearman_lfc` (rank correlation on the real-significant genes),
`pr_auc` (average precision of |predicted delta| against the true significant/non-significant
label), `de_overlap_accuracy` / `jaccard` (top-N-by-|predicted delta| vs. the true significant set,
N = that pair's true significant-gene count).

| source | de_spearman_lfc | pr_auc | overlap_accuracy | jaccard | pairs |
|---|---|---|---|---|---|
| additive | 0.389 | 0.012 | 0.017 | 0.009 | 1568 |
| knn | 0.382 | 0.010 | 0.009 | 0.005 | 1568 |
| pca | 0.414 | 0.034 | 0.023 | 0.014 | 1568 |
| nmf | 0.414 | 0.041 | 0.029 | 0.019 | 1568 |
| stack (gen, cytokine-aligned, mdm) | 0.357 | 0.030 | **0.049** | **0.026** | 1600 |
| **stack (gen, drug-aligned, mdm)** | **0.466** | **0.075** | 0.076 | 0.047 | 1518 |

**Unlike Pearson-Delta, both Stack checkpoints beat every baseline here** — cytokine-aligned
already leads on overlap_accuracy/jaccard; drug-aligned leads on all four, by close to 2x the best
baseline on `pr_auc`. Point estimates alone, though, do not distinguish real (patient, drug)-
specific signal from a generically-plausible predictor that would score above these baselines on
*any* pairing — the same failure mode Pearson-Delta's `r_offdiag`/`rank` columns exist to catch.
No `r_offdiag` analogue existed for the DE metrics, so before trusting this table a permutation
null was run: shuffle which (patient, drug) label each predicted-delta row claims (`pred_key`'s
row order only — the delta content is untouched), recompute the four metrics' means, 200 shuffles
per checkpoint, one-sided p = fraction of shuffles reaching the real (correctly-paired) value.

| checkpoint | metric | observed | null mean | null std | **specific lift** | p |
|---|---|---|---|---|---|---|
| cytokine-aligned | de_spearman_lfc | 0.357 | 0.129 | 0.0065 | **+0.228** | <0.005 |
| cytokine-aligned | pr_auc | 0.030 | 0.023 | 0.0006 | **+0.0072** | <0.005 |
| cytokine-aligned | overlap_accuracy | 0.049 | 0.035 | 0.0009 | **+0.0137** | <0.005 |
| cytokine-aligned | jaccard | 0.026 | 0.019 | 0.0006 | **+0.0076** | <0.005 |
| drug-aligned | de_spearman_lfc | 0.466 | 0.193 | 0.0061 | **+0.273** | <0.005 |
| drug-aligned | pr_auc | 0.075 | 0.063 | 0.0008 | **+0.0119** | <0.005 |
| drug-aligned | overlap_accuracy | 0.076 | 0.064 | 0.0012 | **+0.0123** | <0.005 |
| drug-aligned | jaccard | 0.047 | 0.039 | 0.0010 | **+0.0075** | <0.005 |

All 8 rows: 0/200 shuffles reached the observed value (true p is almost certainly far below the
1/201 resolution floor here — every lift is 7–45 null standard deviations out, and the null
distributions are tight, ~1500-pair means). **Both checkpoints carry real, statistically robust
(patient, drug)-specific DE signal — this itself contradicts Pearson-Delta's null verdict.** But
the null mean is not near zero for either checkpoint (de_spearman_lfc null ≈ 0.13–0.19) — a real
generic/non-specific correlation floor, the DE-metric analogue of a smooth predictor's nonzero
`r_offdiag`. Comparing checkpoints on the *specific lift* rather than the raw point estimate
narrows drug-alignment's apparent edge and reverses it on two of four metrics: drug-aligned wins
clearly on the ranking-type metrics (de_spearman_lfc +20% relative, pr_auc +65% relative) but is
a wash on the set-identification metrics (overlap_accuracy: cytokine slightly ahead;
jaccard: tied). Drug-alignment also raises its own generic floor across all four metrics — worth a
follow-up look (are its deltas more broadly perturbation-plausible in a non-specific way, or is
this sci-Plex-domain structure that partially transfers to Tahoe drugs regardless of identity?),
not concerning on its own since the specific-signal finding stands either way.

**Read this section as: the generation-quality question has two different answers depending on
which axis you score.** Pearson-Delta (dense, whole-transcriptome) says both checkpoints are null.
DE-based metrics (sparse, restricted to genes that actually moved) say both checkpoints carry real
per-pair signal beyond baselines, with drug-alignment providing a real but modest edge on half the
metrics. Neither reading is wrong; they are answering different questions about the same
generated delta, and a dense correlation can be swamped by thousands of non-DE genes even when the
sparse, biologically-relevant signal is real.

## Gate — Hallmark readout on the real delta vs random gene sets

| Hallmark set | interaction | global | p vs random | clears gate? |
|---|---|---|---|---|
| P53 pathway | 0.009 | −0.091 | 0.645 | no |
| Apoptosis | 0.004 | 0.003 | 0.720 | no |
| E2F targets | 0.032 | 0.049 | 0.025 | **yes** |
| G2M checkpoint | 0.047 | 0.117 | 0.000 | **yes** |

Only the two proliferation sets beat random; the cell-death sets are indistinguishable from
random on Tahoe (so a death-signature readout is underpowered here).

## Check 2 — end-to-end vs GDSC2 AUC (grouped 5-fold by cell line, CV-tuned)

Trained penalized regression. global = overall potency, interaction = cell-line-specific
response, per-drug = within-drug line ranking, p_label = label-permutation p on interaction.

> **Two caveats, both added 2026-08-24, that bound how far every number below can be read.**
>
> **1. The generated deltas come from the prompting configuration the preprint itself reports
> as insufficient for this task.** Stack v2's Section 2.6 tests donor-specific (individual-
> specific) perturbation response — structurally the same axis as `interaction` here — and
> finds that it requires a synthetic prompt with blending, stating that "Stack with synthetic
> prompts outperforms alternative baselines **and Stack with original prompts** in capturing
> donor-specific effects". Our generation (`04_stack_generate.sbatch`, and Path B's
> `11_soragni_generate.sbatch`) uses ordinary drug-context prompts on the default `--mode mdm`
> 5-step schedule — the "original prompts" arm. So the near-zero `interaction` for every
> **generated delta** row is what v2 predicts for this configuration, and is not yet evidence
> about Stack's ceiling on personalization. Re-running under the synthetic-prompt construction
> (v2 Methods 4.10, including its 1-step rather than 5-step generation) is the outstanding test.
> This caveat does **not** touch the `expr`/`pca`/`nmf`/`additive`/`knn` rows or the two
> embedding rows, none of which involve generation.
>
> **2. No row here has faced a same-width random-feature control.** The 2026-08-22 controls run
> (Slurm 31564601) scored `expr`, `stack` and `oracle` against matched i.i.d. Gaussian features
> through this identical pipeline, and **0 of 9 real rows beat their own noise control on
> `global`**, on both checkpoints — including `oracle`, the real measured Tahoe delta. That
> exposes `global` here as largely a drug-mean artifact: a CV-tuned model on pure noise shrinks
> to the per-drug training mean, which already ranks drug potency about as well as anything.
> Critically, **`base (embed)` was not in that run**, and it is the one representation whose
> headline finding depends on the outcome — its `global` of 0.644 sits inside the ~0.62 band
> every random control produced. Until it is scored against its own random control, treat the
> base-embedding result as provisional. The same run's planted positive control recovers
> cleanly (interaction 0.66–0.68, p_label 0.000), so the pipeline does have power — but note it
> is planted and scored in a 5-dimensional PCA subspace, whereas every real row above is fed at
> 2,000 HVGs; `check2.py` records that a signal planted in raw gene space "cannot recover ANY
> signal at any effect size" at this n. Power at k=5 is not power at p=2000.

**Representations tested.** Check 2 runs two separate analyses: (a) fixed-signature Hallmark
readouts, no model fit, applied only to the 6 delta sources; (b) the penalized-regression grid
below, applied to all 10 representations (source of the ladder table). Every representation in
(b) is scored through the *same* model class per Kurilov (2020) — RidgeCV/LassoCV/ElasticNetCV,
alpha tuned per representation via inner 3-fold CV — under leave-cell-line-out grouped 5-fold CV,
so a difference across rows is the representation, not the model.

| representation | kind | mechanism | key params |
|---|---|---|---|
| expr | raw baseline expression | the untreated (DMSO) cell state itself — no delta, no drug information in the representation | top 2,000 HVGs |
| additive | baseline delta | each drug's mean real Tahoe delta across every *other* line, broadcast flat to the held-out line — ignores the line entirely; the drug-main-effect floor | leave-one-line-out |
| knn | baseline delta | mean real delta of the *k* other lines whose baseline expression is most similar (cosine, standardized/L2-normalized), among lines treated with that drug — sees the query baseline like Stack does, but averages real neighbors instead of generating | k=10 |
| pca | learned baseline→delta map | ridge-regress each drug's delta *residual* (delta − drug mean) on PCA components of the per-line baseline, predict, add the drug mean back — an organoid-specific correction on top of the additive floor | 20 components, ridge α=1.0, Hallmark-restricted gene panel |
| nmf | learned baseline→delta map | same as pca, non-negative matrix factorization instead of PCA | same params |
| stack (gen, cytokine-aligned) | FM-generated delta | Stack-Large in-context-generated post-drug state minus the per-line control baseline, cytokine-aligned checkpoint | see Check 1 methodology |
| stack (gen, drug-aligned, unfilt./leak-excl.) | FM-generated delta | same, sci-Plex drug-aligned checkpoint; leak-excluded drops the ~5 doubly-exposed A549/drug pairs | see Check 1 methodology |
| base (embed) | learned representation, not a delta | the per-line Stack embedding from the *unaligned* `bc_large.ckpt` encoder, fed directly into the regression as a feature vector — no generation head, no drug info | encoder only |
| aligned (embed) | learned representation, not a delta | same, from the cytokine-aligned checkpoint's encoder | encoder-stripped from `bc_large_aligned.ckpt` |

**Splitting — corrected.** The delta *sources* are rebuilt genuinely leave-one-line-out
(`_loo_baseline_source`), but the penalized fit is **grouped 5-fold**, not leave-one-out:
`--folds` defaults to 5 (`score_generation_eval.py:271`) and neither `05_stack_score.sbatch` nor
`07_stack_emb_score.sbatch` passes it. Still leakage-free — `fold_of` groups by cell line, so no
line is in both train and test, and the interaction/p_label conclusions stand — but true LOO
requires re-running with `--folds 999`.

**Selection gap@k**: rank drugs by predicted response for each cell line, take the top k; the
shortfall from the best *actual* drug in that shortlist to the line's true best, **divided by that
line's observed AUC range** (`regret_norm_at_k`, `evaluation.py:224`) — so it is unitless on
[0, 1], not AUC units. Lowest across the L1/L2/EN sweep (each k minimized independently).
Because GDSC2 panels are right-skewed, a **random** ranking scores ≈0.70 here, not 0.50.

**Baseline, generated-delta, and embedding ladder** (base = unaligned `bc_large` embedding;
aligned = cytokine-aligned checkpoint embedding, encoder-stripped; baseline/embedding rows are
scored on the unfiltered ~1,313-pair design, the leak-excluded stack row on ~1,308):

| representation | L2 global / int | L1 global / int | EN global / int | sel. gap@1 | sel. gap@3 |
|---|---|---|---|---|---|
| expr | 0.475 / −0.037 | 0.599 / −0.108 | 0.604 / −0.123 | 0.354 | 0.119 |
| additive | 0.628 / −0.095 | 0.603 / −0.159 | 0.601 / −0.155 | 0.264 | 0.091 |
| knn | 0.547 / −0.068 | 0.617 / −0.171 | 0.618 / −0.169 | 0.250 | 0.101 |
| pca | 0.585 / +0.007 | 0.634 / −0.108 | 0.634 / −0.103 | 0.219 | 0.102 |
| nmf | 0.550 / +0.007 | 0.610 / −0.198 | 0.614 / −0.178 | 0.251 | 0.082 |
| stack (gen, cytokine-aligned) | 0.539 / −0.003 | 0.568 / −0.182 | 0.574 / −0.164 | 0.320 | 0.144 |
| stack (gen, drug-aligned, unfiltered) | 0.561 / −0.082 | 0.575 / −0.145 | 0.570 / −0.147 | 0.343 | 0.133 |
| stack (gen, drug-aligned, leak-excluded) | 0.561 / −0.079 | 0.572 / −0.134 | 0.562 / −0.123 | 0.324 | 0.122 |
| **base (embed)** | **0.644 / +0.119** | 0.612 / −0.166 | 0.613 / −0.170 | 0.273 | 0.102 |
| aligned (embed) | 0.618 / +0.045 | 0.623 / −0.097 | 0.625 / −0.103 | 0.240 | 0.096 |

Leakage filtering changes essentially nothing here either: the drug-aligned row moves by
≤0.02 on every ladder metric after excluding the doubly-exposed A549/drug pairs. (Its own
leakage-filter print reports `doubly_exposed_frac=0.000`, not the ~0.003 Check 1 reports on the
same declared corpus — not a discrepancy: `filter_leakage` runs on the *full* 141,103-row GDSC2
`design` frame before it is restricted to the ~1,313 Tahoe-drug-overlapping pairs the table
scores, so the same 5 doubly-exposed pairs Check 1 finds land on a ~90x larger denominator and
round to 0.000 at 3 decimals.)

**Fixed-signature readouts** (Hallmark gene sets scored directly on each delta source — no
training — full per-source table, all three Stack checkpoint variants; baseline rows are scored
on the unfiltered ~1,313-pair design, the leak-excluded stack rows on ~1,308):

| source | method | global | interaction | per-drug | sel. gap@1 | sel. gap@3 | p_label |
|---|---|---|---|---|---|---|---|
| additive | hallmark | 0.088 | −0.063 | −0.033 | 0.855 | 0.579 | 0.979 |
| additive | proliferation | 0.097 | −0.016 | −0.035 | 0.585 | 0.547 | 0.884 |
| knn | hallmark | 0.080 | 0.022 | −0.008 | 0.788 | 0.587 | 0.220 |
| knn | proliferation | 0.092 | 0.015 | 0.063 | 0.628 | 0.415 | 0.360 |
| pca | hallmark | 0.089 | −0.075 | 0.053 | 0.855 | 0.579 | 0.985 |
| pca | proliferation | 0.123 | −0.010 | 0.049 | 0.585 | 0.535 | 0.848 |
| nmf | hallmark | 0.077 | −0.068 | −0.072 | 0.855 | 0.579 | 0.981 |
| nmf | proliferation | 0.094 | −0.016 | −0.023 | 0.585 | 0.549 | 0.882 |
| stack (gen, cytokine-aligned) | hallmark | −0.128 | −0.030 | −0.090 | 0.733 | 0.581 | 0.809 |
| stack (gen, cytokine-aligned) | proliferation | −0.150 | −0.014 | −0.090 | 0.849 | 0.674 | 0.560 |
| stack (gen, drug-aligned, unfiltered) | hallmark | −0.077 | 0.019 | −0.084 | 0.687 | 0.532 | 0.322 |
| stack (gen, drug-aligned, unfiltered) | proliferation | −0.038 | 0.081 | −0.057 | 0.317 | 0.209 | 0.009 |
| stack (gen, drug-aligned, leak-excluded) | hallmark | −0.074 | 0.021 | −0.079 | 0.687 | 0.532 | 0.296 |
| stack (gen, drug-aligned, leak-excluded) | proliferation | −0.035 | 0.082 | −0.057 | 0.317 | 0.208 | 0.004 |

Through the fixed readouts the cytokine-aligned Stack delta is *negative* on both methods
(hallmark −0.128, proliferation −0.150) — consistent with its Check-1 null. The drug-aligned
delta's global scores stay negative but smaller in magnitude, and — unlike the ladder's trained
fits — its *interaction* term is slightly positive (proliferation +0.08, p_label 0.004–0.009);
leakage filtering moves it only marginally (interaction 0.019→0.021 hallmark, 0.081→0.082
proliferation), the same near-independence Check 1 found. The other (non-Stack) deltas sit at
global ~0.08–0.12 regardless of checkpoint.

**This does not overturn "only the base embedding" (Findings below).** The same drug-aligned,
leak-excluded delta, scored through the **trained** ridge fit instead of this fixed readout,
gives interaction **−0.079** with p_label **0.987** (see the ladder table above) — opposite sign,
non-significant. The fixed and trained readouts disagree on this representation. And p_label
0.004 is one of 28 fixed-readout comparisons in this table (7 sources × 2 methods × the
checkpoint variants); it does not survive even a Bonferroni correction (0.05/28 ≈ 0.0018). Read
this row as readout-fragile, not as evidence against the base-embedding-only finding.

Per-drug ranking and label-permutation significance — the two columns the ladder omits, for the
ridge (L2) fits that carry the embedding signal:

| representation | per-drug | p_label |
|---|---|---|
| **base (L2)** | **+0.200** | **0.001** |
| aligned (L2) | +0.059 | 0.175 (n.s.) |

**The base Stack embedding is the only representation in the whole eval that captures
cell-line-specific drug response.** Under ridge it reaches interaction +0.119 (per-drug +0.200,
p_label 0.001) — where expression, PCA, NMF, and every generated delta sit at ≈0 and
non-significant. Two features sharpen the reading:
- **Dense, not sparse.** The signal is spread across many correlated embedding dimensions:
  L1/EN sparsify it away (interaction −0.17). The opposite of raw expression, where L1 *helps*
  (global 0.475 → 0.599) by selecting informative genes.
- **Alignment does not transfer.** base > aligned on every interaction metric (aligned L2 int
  0.045, n.s. p = 0.175). Cytokine-domain alignment adds nothing to — and slightly dilutes —
  the drug-response signal.

The interaction is modest against the 0.31–0.47 screen-agreement ceiling (~25–40% of it), and
on the top-1 selection gap the representations are close (base 0.27 vs pca 0.22, aligned 0.24),
converging further at top-3 (all ~0.08–0.14), so the edge is in the interaction *correlation*,
not yet the single best drug pick.

**Do not read the gap@1 column as a ranking.** pca 0.219 / aligned 0.240 / knn 0.250 /
nmf 0.251 / additive 0.264 / base 0.273 span 0.054, against a minimum detectable difference of
0.047–0.106 at n = 50 lines — and ≈0.028 of that span is the min-over-{L1,L2,EN} operator, which
buys that much pure selection optimism even on zero-signal models. Those six are mutually
indistinguishable. Only expr (0.354) and the stack variants (0.320–0.343) plausibly separate,
and all are borderline once the 0.028 is subtracted. Note also that the reported gap@1 and gap@3 for a row
generally come from *different* penalties, so no single deployable model achieves both.

**Overall potency is solved (~0.6) by everything; personalization is captured only by the base
Stack embedding.** Drug main effect (global) reaches ~0.5–0.64 for every representation, but the
cell-line-specific interaction is ≈0 and non-significant for all except base-embedding ridge.

> **Proposed — MOA-level interpretation of the shortlists.** Selection gap is drug-level and
> mechanism-blind: representations can post the same ΔAUC while shortlisting mechanistically
> different drugs, and the top-3 convergence is likely because the true-best drug for most of the
> 50 lines is one of a few broadly-potent compounds (so every representation picks the same
> pan-active MOAs — a saturated, low-power discriminator). Two mechanism-aware readouts to add:
> (1) **MOA hit-rate@k** — does the top-k contain a drug sharing the true-best's pathway/target?
> Me-too compounds collapse, it is the clinical question (right pathway, not right compound), and
> the base embedding's positive interaction predicts it should hit more often than PCA at equal
> ΔAUC. (2) **Stratify the interaction by MOA class** — targeted agents (MEK/PI3K/RTK/CDK…) are
> line-specific by biology, broad cytotoxics are not, so base's edge should concentrate in
> targeted MOAs and vanish in cytotoxics; if it does, that is mechanistic corroboration of the
> 0.119 and ties to the driver-matching story, if not the signal is likely non-biological.
> Caveats: 50 lines → per-MOA is illustrative not powered; control hit-rate against the pan-active
> base rate (shuffled shortlist) so saturation does not read as skill. Needs GDSC2
> `PATHWAY_NAME`/`PUTATIVE_TARGET` joined to the drug table (not in the current context map).

> **Answered (2026-08-19) — are all the models just picking the same few toxic drugs?** Filled in
> by replicating `check2_registry_driver.run_check2`'s representation construction but keeping
> `penalized_preds`'s per-pair predictions instead of letting `score_check2` discard them (no
> library code changed — the function already returns exactly this), picking each
> representation's best-gap@1 penalty (matching how the published SEL_GAP table's own "best of
> L1/L2/EN" was built — this reconstruction's own gap@1 values reproduce SEL_GAP to 3 decimals
> for every non-stack representation, confirming it's a faithful replay), then scoring the actual
> #1 picks (n=44 of 50 lines scored) against the same broadly-active definition
> `scripts/pick_concentration_reference.py` used for the truth/prior rows:
>
> | | distinct drugs ever picked #1 (of 26) | most-picked drug's share of lines | share of #1 picks that are broadly active |
> |---|---|---|---|
> | observed best (the truth) | 6.7 (95% 2–13) | 89% (95% 49–100%) | *reference* |
> | potency prior (ignores the cell line) | 1 by construction | 100% | 100% |
> | additive | 3 | 86.4% | 100% |
> | knn | 3 | 86.4% | 100% |
> | nmf | 3 | 86.4% | 100% |
> | pca | 4 | 86.4% | 97.7% |
> | base (embed) | 4 | 81.8% | 97.7% |
> | aligned (embed) | 4 | 88.6% | 97.7% |
> | expr | 7 | 65.9% | 88.6% |
> | stack (gen, cytokine-aligned) | 5 | 54.5% | 88.6% |
> | stack (gen, drug-aligned, unfiltered) | 7 | 70.5% | 93.2% |
>
> "Broadly active" = a drug whose AUC is below the line's own median in most lines — i.e. the
> compounds that work on nearly everything. Read the table like this: **if a representation picks
> only 1–2 distinct drugs across all 50 lines, it is ranking toxicity and nothing else.** If its
> pick distribution resembles the observed one, it is doing something cell-line-specific.
>
> **Result: the regression baselines (additive/knn/nmf/pca) ARE just ranking toxicity** — 3–4
> distinct picks, 82–86% modal share, both far more concentrated than the truth's own 6.7/89%
> reference and close to the 1-drug/100% potency-prior floor. **expr and both stack rows are
> the least concentrated** (5–7 distinct drugs, 55–71% modal share) — closer to, though still
> more concentrated than, the observed truth. This does not contradict finding 4 (only the base
> embedding shows a significant trained-ridge *interaction* term): a representation can shortlist
> more diverse drugs without those picks being more *correct* — gap@1 for stack and expr is
> actually the *worst* in the table (0.32–0.36 vs. 0.22–0.26 for the tightly-concentrated
> baselines), so their extra diversity isn't (yet) buying better selections, just less pure
> potency-chasing. Caveat: the stack rows pair the Aug-12 vanilla-mode generated deltas (whatever
> produced the currently-published CHECK2_RIDGE/SEL_GAP numbers) with the newer
> `tahoe_query_baseline.h5ad`, since the original 50-row pseudobulk query baseline wasn't
> preserved — an approximation on baseline *magnitude*, not on per-line *ranking*, which is what
> top-1-pick concentration actually depends on.
>
> **Still open:** the potency-prior-vs-gap@1 comparison below (a genuinely separate check — the
> prior needs each model's coefficients zeroed, not a different data source) has not been
> measured; the ≈0.06–0.11 estimate remains a prediction, not a result.
>
> **The truth is itself concentrated, which is the trap.** Across 955 GDSC2 lines the observed
> best drug is one of only 13 distinct compounds, and Staurosporine alone is best for 69% of
> lines. So "picks toxic drugs" is partly *correct behaviour*, and a good gap@k does not by
> itself indicate personalization. The question is whether the models are **more** concentrated
> than the truth. On a 50-line panel the observed reference is ~6 distinct drugs (95% band 4–8)
> with a modal share of ~0.69 (band 0.58–0.80).
>
> **The one control that settles it.** Rank drugs purely by their mean AUC over the training
> lines, ignoring the cell line entirely — the "potency prior". This is not an external baseline:
> `fmharness.check2.penalized_preds` fits per drug with `StandardScaler` on the training lines
> and `fit_intercept=True`, so the intercept already *is* that
> training mean, and the prior is the same fitted model with its coefficients zeroed. Score it
> with the same gap@k on the same folds. **If the models do not beat it, their shortlists carry
> no cell-line information at all.** Two independent reconstructions put the prior at gap@1
> ≈ 0.06–0.11 against 0.22–0.36 for every representation, so the expected result is that the
> prior *wins* — a stronger statement than "confounded by toxicity".
>
> **The `y_prior` fix is still the right next step for the prior comparison.**
> `fmharness.check2.penalized_preds` builds the per-(line, drug) prediction frame and returns it
> unchanged — `score_check2` is what discards it, which is why the table above only needed a new
> script, not a library change. But `y_prior` (the same fitted model with its coefficients
> zeroed) isn't in that returned frame at all, so the potency-prior-vs-gap@1 comparison still
> needs the fold loop itself changed (~10 lines, per the original estimate) before it can be
> measured rather than predicted.
>
> **If the answer comes back "yes, concentrated"**, the follow-up is to stop scoring selection in
> raw AUC and score it in percentile-within-drug instead (each drug's out-of-fold rank among
> training lines). That makes every drug's marginal uniform, so a pan-cytotoxic compound carries
> zero advantage, random is exactly 1/(k+1), and the potency prior lands there by construction —
> which is the property that deleting the toxic drugs from the panel was trying to buy, except
> that deletion also collapses gap@k's per-line normalizer (the prior moves 0.061 → 0.508 with no
> change in model quality) and conditions on the outcome, since the observed best drug is itself
> a broadly-active one for 83–93% of lines.
>
> This closes the loop on the MOA note above: its guess that the top-3 convergence comes from a
> few broadly-potent compounds is exactly what the first table tests.

## Ceilings — the most any predictor can score

| comparison | pairs | overall (global) | cell-line-specific (interaction) | per-drug |
|---|---|---|---|---|
| GDSC2 ↔ CTRP | 25,028 | 0.69 | **0.47** | 0.44 |
| GDSC2 ↔ PRISM | 27,443 | 0.59 | **0.31** | 0.31 |

Even independent gold-standard screens agree only 0.31–0.47 on the cell-line-specific axis, so
the interaction target is *real but weak*. The near-zero interaction above is therefore a
genuine prediction gap, not label noise.

---

## Findings

1. **Stack generation does not reproduce drug-induced changes on Tahoe** (Check 1 r = 0.012).
   Consistent with Ahlmann-Eltze (FM-as-generator ≤ "no change") and the earlier L1000 Path-B
   null.
2. **The generator *is* the alignment.** The base `bc_large.ckpt` has no generation head
   (missing `query_pos_embedding`, `cls`); it's added by the alignment step. There is no
   "unaligned Stack generation" — removing the cytokine/PBMC domain bias requires
   **re-alignment on drug data**, not a checkpoint swap.
3. **The baselines solve the easy axis, not the hard one.** Drug potency (global) ~0.6, near
   the 0.59–0.69 screen-agreement level; cell-line-specific response ~0 for every *generated
   delta* and for raw expression/PCA/NMF, against a 0.31–0.47 ceiling.
4. **Generation and embedding dissociate: the signal is in the embedding.** Used as a cell-line
   representation, the *unaligned* base Stack embedding is the only predictor with significant
   cell-line-specific drug response (interaction 0.119, per-drug 0.200, p = 0.001, ridge).
   Cytokine-alignment does not help (base > aligned). So Stack carries drug-response-relevant
   structure — in the cell-state **embedding**, not the generative **delta**.
5. **The Pearson-Delta null survives switching to Stack's faithful generation procedure, but a
   DE-restricted metric tells a different story (Check 1b).** Re-running Check 1 under
   `--mode mdm` (the CLI's own confidence-guided schedule, not the `vanilla` workaround) leaves
   both checkpoints in the same null band (r = −0.001 / 0.006) — finding 1 was not an artifact of
   the workaround. But scored on DE-restricted metrics instead of the full dense profile, both
   checkpoints show real, permutation-significant (patient, drug)-specific signal beyond every
   baseline (p < 0.005, 7–45 null-SDs out on every metric) — Pearson-Delta's whole-transcriptome
   average was swamping a real sparse signal in ~15,000 mostly-non-DE genes. Drug-alignment gives
   a real, permutation-confirmed edge on the ranking-type DE metrics (de_spearman_lfc, pr_auc) but
   not on the set-identification ones (overlap_accuracy, jaccard) once each checkpoint's own
   generic-correlation floor is subtracted out — a real, modest, partial confirmation of the
   drug-alignment hypothesis, not the clean 2x win the raw point estimates alone suggested.

## Relation to the Stack preprint

**Which version.** The preprint has two, and they differ in ways that matter here. **v1**
(2026-01-09) is what PMC12803207 mirrors and labels "[Version 1]". **v2** (2026-06-08)
exists only on bioRxiv (`10.64898/2026.01.09.698608v2`); PMC, Semantic Scholar and the
HuggingFace `arcinstitute/Stack-Large` card all still point at v1, and v2 does not surface
in web search, so it has to be read from the bioRxiv PDF directly (that host also rate-limits
automated fetches — download and `pdftotext -layout`). v2 grows Perturb Sapiens from 201 to
892 perturbations, adds the **DiseasePert-3M** dataset (3.3M T/NK cells, 40 donors, 32 patients
across 14 diseases plus 8 healthy controls, 11 cytokines), and adds **Section 2.6, "Stack
exhibits donor-specific cytokine response prioritization ability"**, which has no v1
counterpart. Everything below is checked against v2.

Broadly, these results are **consistent with the preprint's scope** on generation and
**provisionally consistent with its central claim** on embeddings. That is weaker than
replication, and the first bullet is now a scoped non-replication rather than untouched
territory.

- **Generation null on Tahoe drugs — partly out of domain, partly a non-replication.** The
  paper post-trains the in-context generation head only on **cytokine/PBMC** perturbations
  (CELLxGENE + the Parse PBMC 10M dataset: 12 donors, 90 cytokine perturbations), a corpus
  "enriched for primary cells profiled from human tissues and blood, with a particular
  emphasis on immune cells" — **no drugs, no cancer lines**. That holds in both versions. But
  it does *not* follow that drugs are untested: the OpenProblems **drug perturbation** dataset
  is one of its generation benchmarks in both versions, on drug conditions the paper states
  were "unseen during model pre-training or post-training", and v2's abstract foregrounds
  "892 drug, cytokine, and genetic perturbations". So the generative head was never *trained*
  on drugs but was explicitly *tested* on them, and the paper claims generalization there.
  What remains genuinely out of domain is **cell context** (its drug generation is primary
  immune cells, largely T-cell lineage; ours is 50 cancer lines) and **Tahoe specifically** —
  Tahoe stays an embedding / perturbation-classification benchmark in v2 (Fig. 2E linear
  probing), never a generation benchmark, and generation is never benchmarked on cancer cell
  lines anywhere in either version. Read our r ≈ 0 as a failure to reproduce a claimed
  drug-generation generalization *across cell context*, scored on the paper's own headline
  metric (Pearson Delta) — not as a result in territory the paper never entered. The
  *magnitude* (below the additive / no-change floor) still matches the general FM-as-generator
  critique (Ahlmann-Eltze) rather than being a Stack-specific failure.
- **Embedding carries drug-response signal — the preprint's actual claim.** Stack's validated
  strength *is* the embedding, and Tahoe is one of its embedding benchmarks. That the Tahoe
  cell-line embedding predicts GDSC2 response is a downstream confirmation of that claim —
  **provisionally**, see the random-feature caveat under Check 2: `base (embed)` has not yet
  been scored against a same-width random control, and it is the one representation whose
  survival that control would decide.
- **base > aligned for drugs — expected.** Alignment is cytokine-domain fine-tuning with drugs
  held out; using it out-of-domain should not help and can distort the representation. It does
  exactly that here.

**v2's Section 2.6 is the same question our Check 2 asks**, and it cuts against reading our
interaction null as a statement about Stack's ceiling. 2.6 tests donor-specific (individual-
specific) perturbation response — structurally the interaction axis — and reports that it
required a bespoke **synthetic prompt with blending** (Methods 4.10: add the log-normalized
healthy-donor perturbed-minus-control difference onto the patient control query, clip at zero,
project back to count space, generate with **1-step rather than the default 5-step** schedule,
then average the prediction with the healthy-donor perturbed profile). Decisively: "Stack with
synthetic prompts outperforms alternative baselines **and Stack with original prompts** in
capturing donor-specific effects." Plain prompting does not do this task, by the authors' own
evaluation — and plain prompting is exactly our configuration (see the caveat under Check 2).
Two of 2.6's own concessions also line up with our results: "absolute scores remain low", and
"minimal differences observed across methods under standard Cell-Eval applied to all DEGs" —
which is our Check 1 dense-metric null, stated by the authors. Note also that 2.6 isolates the
donor-specific component by subtracting the healthy-donor perturbation DEGs from the predicted
DEGs, the same shared-effect removal `interaction_rho` performs by row-centering: independent
convergence on our decomposition.

Two things go **beyond** what the preprint reports: (i) the advantage is resolved specifically
to the cell-line × drug **interaction** term (leave-line-out GDSC2), a finer decomposition than
any paper benchmark; and (ii) the separation is unusually clean — only the embedding clears
significance while PCA/expression sit at null, a sharper margin than the incremental FM-vs-PCA
gaps typical of the paper's embedding tables. Both are consistent in *direction* with the
preprint; neither is directly stated by it. Both are also contingent on the random-feature
control below.

## Completed since the previous checkpoint

- **sci-Plex drug-aligned generation — done, answered.** Fine-tuned the base on sci-Plex 3
  (GSE139944) single-cell drug perturbations (disjoint from Tahoe), then generated + scored —
  `08_sciplex_prep.sbatch` → `09_stack_finetune.sbatch` → `04` (CKPT override) → score. This
  tested whether the Check-1 null is specifically the cytokine alignment (fixable by
  drug-aligning the gen head) or intrinsic to generation mode. **Answer: intrinsic to generation
  mode.** Drug alignment moves Check-1 r from 0.012 to 0.021 (see the Check 1 table above) —
  still null either way, both far below the additive floor (0.225) and the 0.46 ceiling.
  Consistent with finding 4: the embedding story holds regardless of checkpoint.
- **Faithful generation procedure + DE-based metrics — done, answered (2026-08-19).** Two
  open items: (a) the vanilla-mode workaround above was a materially simpler fixed-ratio
  procedure than Stack's own default (`--mode mdm`, confidence-guided scheduled unmasking) — was
  the null an artifact of that workaround? (b) Pearson-Delta scores the full dense profile —
  would a metric restricted to genes that actually moved tell a different story? Fixed the
  sci-Plex identity-missing-cell bug (54,100 → 17,578 correct controls), re-ran `08`→`09` on the
  corrected input, re-ran `03`→`04` under `--mode mdm` with a 400-real-cell query pool
  (replacing the 50-row pseudobulk baseline that forced the vanilla workaround), added a
  confidence-filtered replicate-aggregation step (`gen_logit`-based, calibrated empirically) and
  a ground-truth Wilcoxon DE-calls bundle. **Answers: (a) no — the faithful procedure lands in
  the same Pearson-Delta null band. (b) yes — DE-restricted metrics show real,
  permutation-significant per-pair signal in both checkpoints that Pearson-Delta was missing.**
  See Check 1b and finding 5.
- **"Are the top picks just the pan-toxic drugs?" per-representation table — done, answered
  (2026-08-19).** `fmharness.check2.penalized_preds` already returns full per-pair predictions;
  no library change was needed, just a script that keeps what `score_check2` was discarding.
  **Answer: yes for the regression baselines, less so for expr and stack.** additive/knn/nmf/pca
  pick only 3–4 distinct drugs at 82–86% modal share (close to the 1-drug/100% potency-prior
  floor); expr and both stack checkpoints pick 5–7 distinct drugs at 55–71% share (closer to the
  observed truth's 6.7/89%) but with *worse* gap@1 — more diverse shortlists, not yet more
  correct ones. See the Check-2 "Proposed" section (now answered) for the full table and caveats.

## Reproducibility

Branch `tahoe-generation-eval`. Pipeline: `00`–`05` (target CIDs → pseudobulk → context →
generate → score), `06`/`07` (embeddings), `08`/`09` (sci-Plex alignment). Getting generation
to run required `--split-column pert_id --split-values "$PERT"`, `--mode vanilla`, and
`prompt+context = 0.901` (so n_test = 50 = the query size, no padding into Stack's 512-cell set).
Key commits: 4720b63 (generation wiring), 6a3fa7a (scorer crash fix), 3ce6a02 (`--stack-emb`).

**2026-08-19 update — faithful procedure supersedes the vanilla workaround above for all rows
dated 2026-08-19.** `03_stack_context.sbatch` now writes a 400-real-cell query pool (8/line, real
single cells, not a pseudobulk row) large enough for `04_stack_generate.sbatch`'s
`--mode mdm --prompt-ratio 0.25 --context-ratio 0.4 --context-ratio-min 0.2` (Stack's own default
schedule) without the old workaround's `IndexError`. Per-query-cell replicates are reduced with
`fmharness.stack_aggregate.aggregate_generated_replicates` (keep `gen_logit < 0`, empirically
calibrated against Check-1 Pearson-Delta — see the threshold sweep in the implementation plan's
Task 7 Step 6) before scoring; `fmharness.stack_aggregate.collapse_query_baseline` reduces the
same query pool to a line-indexed baseline for `--query-baseline` (required — `04`'s query file is
cell-indexed, `build_generated_deltas` joins on the AnnData index directly). DE-calls bundle:
`scripts/build_tahoe_de_calls.py` (Wilcoxon per line, LFC ≥ 0.25 / FDR ≤ 0.05, Stack paper Methods
4.8's threshold) → `fmharness.evaluation.{de_fidelity,score_de_metrics}`. Both driven end-to-end
by `scripts/check1_registry_driver.py --deltas-bundle tahoe_deltas --query-baseline
tahoe_query_baseline.h5ad` (Check 1) and the one-off pattern in the implementation plan's Task 10
Step 2 (Check 1b). Permutation significance for Check 1b: shuffle `pred_key`'s row order (not
`pred_delta`'s), 200 shuffles/checkpoint, one-sided p = frac(null ≥ observed) — script not
committed (matches this project's own uncommitted one-off-analysis convention). Plan:
`docs/superpowers/plans/2026-08-18-stack-faithful-generation-and-de-metrics.md`.
