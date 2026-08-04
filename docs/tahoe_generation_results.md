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
- **Check 2 (end-to-end vs GDSC2 AUC, leave-cell-line-out):** fixed signature readouts, and a
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

## Check 2 — end-to-end vs GDSC2 AUC (leave-one-cell-line-out, CV-tuned)

Trained penalized regression. global = overall potency, interaction = cell-line-specific
response, per-drug = within-drug line ranking, p_label = label-permutation p on interaction.
**Selection gap@k**: rank drugs by predicted response for each cell line, take the top k; the
ΔAUC from the best *actual* drug in that shortlist to the line's true best (AUC units, lower
better) — the potency you lose by trusting the model's top k. Lowest across the L1/L2/EN sweep
(each k minimized independently).

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

**Overall potency is solved (~0.6) by everything; personalization is captured only by the base
Stack embedding.** Drug main effect (global) reaches ~0.5–0.64 for every representation, but the
cell-line-specific interaction is ≈0 and non-significant for all except base-embedding ridge.

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
