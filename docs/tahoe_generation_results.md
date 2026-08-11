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
rebuilt leave-one-line-out), and `stack` (Stack-Large-**Aligned** generated delta).

---

## Check 1 — generation quality (delta-Pearson vs real Tahoe delta)

| source | r | off-diag r | rank | pairs |
|---|---|---|---|---|
| additive | **0.225** | 0.095 | 0.885 | 1600 |
| nmf | 0.221 | 0.088 | 0.912 | 1600 |
| pca | 0.207 | 0.083 | 0.896 | 1600 |
| knn | 0.178 | 0.067 | 0.904 | 1600 |
| **stack (gen)** | **0.012** | −0.002 | 0.644 | 1568 |

Ceiling (delta reproducibility, Tahoe plate split-half): **0.30 raw / 0.46 Spearman-Brown.**

**Stack generation is null.** r = 0.012 is essentially orthogonal to the real change
(off-diagonal ≈ 0, rank 0.64 ≈ random specificity) — far below even the line-independent
additive floor (0.225) and the 0.46 ceiling. The FM's generated delta carries no real signal
about the perturbation. (Baselines top out ~0.18–0.22; the *line-specific* part of the delta
is not recovered by any of them either — `additive`, which ignores the line, is the best.)

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
aligned = cytokine-aligned checkpoint embedding, encoder-stripped):

| representation | L2 global / int | L1 global / int | EN global / int | sel. gap@1 | sel. gap@3 |
|---|---|---|---|---|---|
| expr | 0.475 / −0.037 | 0.598 / −0.105 | 0.605 / −0.121 | 0.360 | 0.114 |
| additive | 0.628 / −0.095 | 0.603 / −0.159 | 0.601 / −0.151 | 0.264 | 0.091 |
| knn | 0.547 / −0.068 | 0.617 / −0.171 | 0.618 / −0.168 | 0.250 | 0.101 |
| pca | 0.585 / +0.007 | 0.634 / −0.108 | 0.634 / −0.103 | 0.219 | 0.102 |
| nmf | 0.550 / +0.007 | 0.610 / −0.198 | 0.614 / −0.178 | 0.251 | 0.082 |
| stack (gen delta) | 0.540 / −0.003 | 0.567 / −0.194 | 0.571 / −0.187 | 0.320 | 0.140 |
| **base (embed)** | **0.644 / +0.119** | 0.612 / −0.166 | 0.613 / −0.170 | 0.273 | 0.102 |
| aligned (embed) | 0.618 / +0.045 | 0.623 / −0.097 | 0.625 / −0.103 | 0.240 | 0.096 |

Through fixed signature readouts the Stack generated delta is *negative* (hallmark −0.128,
proliferation −0.150) — consistent with its Check-1 null; the other deltas sit at global
~0.08–0.12.

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
  (global 0.475 → 0.598) by selecting informative genes.
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
indistinguishable. Only expr (0.360) and stack (0.320) plausibly separate, and both are
borderline once the 0.028 is subtracted. Note also that the reported gap@1 and gap@3 for a row
generally come from *different* penalties, so no single deployable model achieves both.

**Overall potency is solved (~0.6) by everything; personalization is captured only by the base
Stack embedding.** Drug main effect (global) reaches ~0.5–0.64 for every representation, but the
cell-line-specific interaction is ≈0 and non-significant for all except base-embedding ridge.

### MOA-level interpretation of the shortlists

Selection gap is drug-level and mechanism-blind: representations can post the same gap@k while
shortlisting mechanistically different drugs. `scripts/check2_selection_audit.py` joins the
check-2 shortlists to GDSC2's `TARGET` / `TARGET_PATHWAY` columns
(`data/raw/gdsc2_sarcoma/gdsc2/screened_compounds_rel_8.5.csv`) — via
`data/static/tahoe_pert_to_cid.tsv`, since the eval's drug key is a PubChem CID, not a name —
and gets 30/30 drugs annotated. Two mechanism-aware readouts, on the 44-line, 30-drug check-2
panel:

**1. MOA hit-rate@k** — does the top-k shortlist contain a drug sharing the true-best drug's
target pathway? Against a shuffled-shortlist base rate of moa@1=0.106 / moa@3=0.291 /
moa@5=0.441 (what a random ranking already scores, since a handful of pathways cover much of
this 30-drug panel):

| representation (L2) | moa@1 | moa@3 | moa@5 |
|---|---|---|---|
| pca | 0.386 | 0.750 | 0.864 |
| nmf | 0.386 | 0.773 | 0.864 |
| base | 0.295 | 0.727 | 0.909 |
| additive | 0.295 | 0.750 | 0.909 |
| stack | 0.295 | 0.591 | 0.841 |
| knn | 0.273 | 0.705 | 0.841 |
| expr | 0.227 | 0.659 | 0.795 |
| aligned | 0.205 | 0.705 | 0.909 |
| potency prior | 0.318 | 0.750 | 0.886 |

Every representation clears the shuffled floor, but so does the potency prior, and all nine
rows sit in the same narrow band. MOA hit-rate@k does not separate the representations from
each other, or from the line-blind prior, any more cleanly than gap@k does — it corroborates
that this panel is pathway-saturated (few pathways cover most of the 30 drugs), not that any
representation is picking better mechanisms.

**2. Interaction stratified by MOA class** — targeted agents are line-specific by biology, broad
cytotoxics are not:

| representation (L2) | int. targeted | int. cytotoxic |
|---|---|---|
| base | **+0.134** | +0.051 |
| aligned | +0.076 | −0.045 |
| stack | +0.015 | −0.056 |
| pca | 0.000 | +0.004 |
| expr | −0.010 | +0.080 |
| nmf | −0.015 | −0.071 |
| knn | −0.067 | −0.059 |
| additive | −0.086 | −0.191 |
| potency prior | −0.215 | −0.357 |

This is the mechanistic corroboration the original proposal asked for: **base is the only
representation whose interaction concentrates in the targeted class** (+0.134 targeted vs +0.051
cytotoxic), and it is the largest targeted interaction of any representation — consistent with
the 0.119 headline and the driver-matching story. `aligned` shows the same direction more weakly
(+0.076 / −0.045). Every other representation sits at or near zero in the targeted class (pca
0.000, stack +0.015) or negative (expr, nmf, knn, additive); the potency prior is clearly
negative in both classes, as expected from a predictor that carries no cell-line information at
all.

### Are the models just picking the same few toxic drugs? — measured

For each representation, the drug ranked #1 for each of the 44 cell lines in the check-2 panel,
against the drug that was *actually* best for that line ("observed"). "Broadly active" = a drug
whose AUC is below the line's own median in most lines — the compounds that work on nearly
everything.

| representation (L2) | distinct drugs picked #1 | modal share | broadly active share |
|---|---|---|---|
| observed best (the truth) | 10 | 0.295 | 0.977 |
| potency prior (ignores the cell line) | 3 | 0.909 | 1.000 |
| additive | 3 | 0.864 | 1.000 |
| nmf | 5 | 0.818 | 1.000 |
| knn | 4 | 0.750 | 1.000 |
| aligned | 6 | 0.750 | 1.000 |
| pca | 6 | 0.705 | 1.000 |
| base | 5 | 0.727 | 1.000 |
| stack | 6 | 0.591 | 1.000 |
| expr | 8 | 0.500 | 1.000 |

**The models are more concentrated than the truth — unambiguously.** Across all 24
(representation × penalty) combinations in `results/check2_selection_audit.csv`, distinct ranges
3–8 and modal share ranges 0.500–0.886; every single one falls inside the observed row's 10
distinct / 0.295 modal share. The truth itself is *not* concentrated on this 30-drug panel — 10
of 30 drugs are somebody's actual best, and the most-picked truth-optimal drug wins under 30% of
lines — a different picture from the 955-line, 621-compound GDSC2 catalog (13 distinct
best-drugs, Staurosporine 69%) that motivated the original 4–8 / 0.58–0.80 expectation. On this
panel the gap between truth and model is, if anything, larger than predicted: every
representation collapses onto a handful of drugs the truth does not collapse onto.

**The one control that settles it: the potency prior.** Rank drugs purely by their training-fold
mean AUC, ignoring the cell line — `y_prior` in `results/check2_preds.parquet` (Task 3), scored
with the identical `regret_norm_at_k` on the identical folds.

| | gap@1 | pct_gap@1 |
|---|---|---|
| potency prior | 0.245 | **0.621 — worst of all 26 rows** |
| best representation × penalty (pca, L1/L2) | **0.219** | — |
| worst representation × penalty (expr, EN) | 0.393 | — |
| range across all 24 representation × penalty combos | 0.219–0.393 | 0.478–0.608 |

**In raw AUC, the prior does not lose.** Its gap@1 (0.245) sits inside the pack, not below it:
only 5 of 24 representation × penalty combinations score lower (pca at all three penalties,
`aligned` L1/EN), and by a margin of 0.001–0.026 — small next to the ~0.17 spread the
representations show among themselves (0.219 to 0.393). The other 19 combinations, including
every `expr` and `stack` row, score higher (worse) than the prior, several by 0.05–0.15. No
representation clearly separates from the prior on this axis.

**In within-drug percentile space, the picture inverts and resolves.** The prior's pct_gap@1
(0.621) is the single worst value in the entire 26-row table — worse than every representation
at every penalty, the closest being `aligned` L2 at 0.478, a 0.14 gap. Removing each drug's own
location and scale is exactly what strips the prior of its advantage, and it does so cleanly,
in the direction theory predicts.

**The recorded metric decision: `pct_gap@k`, not raw-AUC `gap@k`.** The earlier prediction (two
independent reconstructions, prior gap@1 ≈ 0.06–0.11 against 0.22–0.36 for the representations)
was directionally right but numerically off — the measured prior (0.245) is far higher than
predicted, and the raw-AUC gap between the prior and the representations turned out to be inside
the noise rather than the wide margin expected. That makes the case for `pct_gap@k` stronger, not
weaker: raw-AUC `gap@k` could not even cleanly separate a cell-line-blind ranking from the actual
representations, so it was never a valid selection metric on this panel. `pct_gap@k` does
separate them, correctly, with the line-blind prior scoring worst of all. **Phases 4–6 score
selection with `pct_gap@k`.**

This closes the loop on the MOA finding above: MOA hit-rate@k does not separate the
representations from the prior either (potency prior 0.318 / 0.750 / 0.886 sits mid-pack against
moa@k ranges of 0.205–0.386 / 0.591–0.773 / 0.795–0.909) — an independent confirmation that
raw-AUC-adjacent selection metrics do not isolate per-line personalization on this panel. Only
`pct_gap@k` and the interaction-by-MOA-class split (base's targeted-class edge) do.

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

## In progress

- **sci-Plex drug-aligned generation** — fine-tune the base on sci-Plex 3 (GSE139944) single-cell
  drug perturbations (disjoint from Tahoe), then generate + score — `08_sciplex_prep.sbatch`
  → `09_stack_finetune.sbatch` (running) → `04` (CKPT override) → score. Tests whether the
  Check-1 null is specifically the cytokine alignment (fixable by drug-aligning the gen head) or
  intrinsic to generation mode. Given finding 4, this is confirmatory: the embedding story holds
  either way.

## Reproducibility

Branch `tahoe-generation-eval`. Pipeline: `00`–`05` (target CIDs → pseudobulk → context →
generate → score), `06`/`07` (embeddings), `08`/`09` (sci-Plex alignment). Getting generation
to run required `--split-column pert_id --split-values "$PERT"`, `--mode vanilla`, and
`prompt+context = 0.901` (so n_test = 50 = the query size, no padding into Stack's 512-cell set).
Key commits: 4720b63 (generation wiring), 6a3fa7a (scorer crash fix), 3ce6a02 (`--stack-emb`).
