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

**Pairs are not all from the same run.** Baseline rows (additive/knn/pca/nmf) are scored on the
unfiltered 1,600-pair design; the stack rows use Stack's own coverage (1,568), and the
leak-excluded stack row further drops the 5 doubly-exposed A549/drug pairs (1,563). None of this
changes any conclusion (~5/1300–1600 pairs is tiny) — noted here since it isn't visible from the
table alone.

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

> **Proposed — are all the models just picking the same few toxic drugs?** The direct way to
> answer this is not another summary statistic; it is to look at the actual shortlists. For each
> representation, write down the drug it ranks #1 for each of the ~50 cell lines, and put that
> next to the drug that *actually* was best for that line:
>
> | | distinct drugs ever picked #1 | most-picked drug, and its share of lines | share of #1 picks that are broadly active |
> |---|---|---|---|
> | observed best (the truth) | *reference* | *reference* | *reference* |
> | potency prior (ignores the cell line) | 1 by construction | 100% | 100% |
> | expr / pca / nmf / additive / knn | ? | ? | ? |
> | stack (gen delta) | ? | ? | ? |
> | base (embed) | ? | ? | ? |
>
> "Broadly active" = a drug whose AUC is below the line's own median in most lines — i.e. the
> compounds that work on nearly everything. Read the table like this: **if a representation picks
> only 1–2 distinct drugs across all 50 lines, it is ranking toxicity and nothing else.** If its
> pick distribution resembles the observed one, it is doing something cell-line-specific.
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
> **Why we cannot answer this today, and the fix.** `fmharness.check2.penalized_preds` builds the
> per-(line, drug) prediction frame, scores it, and discards it; nothing in
> `results/` holds per-pair predictions, so the picks are unrecoverable. Emitting `y_prior` in the
> fold loop and dumping `(source, penalty, patient, drug, y_true, y_pred, y_prior)` to
> `results/check2_preds.parquet` is ~10 lines, after which the table above and the prior
> comparison are a groupby. `_personalization` (`per_patient_eval.py:444`) already computes the
> distinct-count and modal-share columns and already carries the observed row as its reference.
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

## Relation to the Stack preprint

Broadly, these results are **what the preprint's scope predicts.**

- **Generation null on Tahoe drugs — expected (out of domain).** The paper (PMC12803207) aligns
  the in-context generation head only on **cytokine/PBMC** perturbations (CELLxGENE + Parse), and
  benchmarks generation on **Parse cytokines + OpenProblems** — never on small-molecule drugs and
  never on Tahoe generation (Tahoe is used *only* as an **embedding** benchmark). A cancer-line
  drug-generation task is therefore a domain the generative head was neither trained nor tested
  on; a null there does not contradict the paper. The *magnitude* (r ≈ 0, below the additive /
  no-change floor) matches the general FM-as-generator critique (Ahlmann-Eltze) rather than being
  a Stack-specific failure.
- **Embedding carries drug-response signal — the preprint's actual claim.** Stack's validated
  strength *is* the embedding, and Tahoe is one of its embedding benchmarks. That the Tahoe
  cell-line embedding predicts GDSC2 response is a downstream confirmation of that claim.
- **base > aligned for drugs — expected.** Alignment is cytokine-domain fine-tuning with drugs
  held out; using it out-of-domain should not help and can distort the representation. It does
  exactly that here.

Two things go **beyond** what the preprint reports, in our favor: (i) the advantage is resolved
specifically to the cell-line × drug **interaction** term (leave-line-out GDSC2), a finer
decomposition than any paper benchmark; and (ii) the separation is unusually clean — only the
embedding clears significance while PCA/expression sit at null, a sharper margin than the
incremental FM-vs-PCA gaps typical of the paper's embedding tables. Both are consistent in
*direction* with the preprint; neither is directly stated by it.

## Completed since the previous checkpoint

- **sci-Plex drug-aligned generation — done, answered.** Fine-tuned the base on sci-Plex 3
  (GSE139944) single-cell drug perturbations (disjoint from Tahoe), then generated + scored —
  `08_sciplex_prep.sbatch` → `09_stack_finetune.sbatch` → `04` (CKPT override) → score. This
  tested whether the Check-1 null is specifically the cytokine alignment (fixable by
  drug-aligning the gen head) or intrinsic to generation mode. **Answer: intrinsic to generation
  mode.** Drug alignment moves Check-1 r from 0.012 to 0.021 (see the Check 1 table above) —
  still null either way, both far below the additive floor (0.225) and the 0.46 ceiling.
  Consistent with finding 4: the embedding story holds regardless of checkpoint.

## Reproducibility

Branch `tahoe-generation-eval`. Pipeline: `00`–`05` (target CIDs → pseudobulk → context →
generate → score), `06`/`07` (embeddings), `08`/`09` (sci-Plex alignment). Getting generation
to run required `--split-column pert_id --split-values "$PERT"`, `--mode vanilla`, and
`prompt+context = 0.901` (so n_test = 50 = the query size, no padding into Stack's 512-cell set).
Key commits: 4720b63 (generation wiring), 6a3fa7a (scorer crash fix), 3ce6a02 (`--stack-emb`).
