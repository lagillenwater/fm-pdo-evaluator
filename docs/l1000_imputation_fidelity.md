# Do L1000's imputed genes carry real delta signal?

**Correction, 2026-08-26 [job 31676844/31676845, supersedes 31661570/31661918].** Every
`p_vs_null` below was computed by comparing a reported AGGREGATE (a mean over ~32 pairs)
against the spread of INDIVIDUAL mismatched-pair draws, instead of against the bootstrapped
sampling distribution of that aggregate at the same pair count -- the same class of bug fixed
for the rung-0 replicate ceiling (commit 6a7a7cf), independently reintroduced here. It
inflated every p in this document by one to two orders of magnitude. With the fix
(`fmharness.statistics.bootstrap_aggregate_pvalue`): **landmark genes DO clear their null**
(p = 0.0005, not 0.2438) and so do **bing** genes (p = 0.0035, not 0.3234); fully-imputed
**other** genes still do not (p = 0.1219). Every transform in the normalisation sweep now
clears its null too (p = 0.0005-0.0055, none at 0.13-0.28). The rest of this document below the
next line is the ORIGINAL, now-superseded writeup; read the correction above first.

**Result (corrected): real but weak.** Cross-platform L1000/Tahoe delta agreement on directly
measured genes (landmark) and the Broad's own best-inferred subset (bing) is statistically real
-- not a mismatched-pair artifact -- but small: mean rho 0.041, about **7% of the L1000
split-half noise ceiling (0.572)**. Fully-imputed genes outside the bing subset show no such
signal. Separately and unaffected by the bug (already a paired, like-for-like test): once
imputed genes are variance-matched to the landmarks, there is no significant fidelity GAP
between measured and imputed genes (Wilcoxon p = 0.27) -- so imputation itself does not look
like the source of the landmark/imputed difference seen in the raw, unmatched comparison; gene
selection (landmarks are chosen to be high-variance) explains most of it. Put together: the
platforms agree weakly but really on genes with real information content (measured + bing), and
imputed genes lose no clearly-measurable fidelity once that's accounted for -- but 7% of ceiling
is still a weak floor to build a common gene panel on. Evidence:
`docs/results/l1000_imputation_fidelity{,_paired,_per_pair}.csv` [job 31676844],
`docs/results/l1000_tahoe_transform_sweep.csv` [job 31676845].

---

## Why this was measured

The L1000 matrix this project uses is
`GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx`. The filename states its own
construction: `INF` for inferred, `mlr` for multiple linear regression, `12k` for the output
width. It carries **978 measured landmark genes** plus roughly **11,350 genes imputed from
them** by linear regression.

That matters because L1000 is the binding constraint on the gene panel. Measured intersections
[job 31660841]:

| panel | genes |
|---|---|
| Path B, Tahoe as delta-transfer corpus | 12,597 |
| Path B, L1000 as delta-transfer corpus | 8,600 |

Requiring L1000 costs **3,997 genes**. It is worth paying only if the genes bought are real
measurements rather than a linear shadow of 978 of them.

The published accuracy figures cannot settle this. L1000toRNAseq (Ma'ayan lab, *BMC
Bioinformatics* 2022) reports gene-wise Pearson **0.502** (SD 0.255) for a CycleGAN
style-transfer model against **0.442** (SD 0.236) for a linear baseline, on 2,929 paired
GTEx-LINCS samples over 11,780 genes. Two reasons that does not transfer to this harness:

1. **It scores absolute expression; this harness scores deltas.** A treated-minus-control delta
   is a small difference of two large numbers, so a fixed imputation error is proportionally
   far larger in the delta than in the level. Fidelity on deltas is a different quantity and
   was unmeasured.
2. **Its own stated limitation is this harness's failure mode.** The authors note the model
   "poorly predict[s] the expression of the targeted single genes from the shRNA and CRISPR
   knockdown profiles" -- it fails precisely where a perturbation acts specifically on a gene.
   A drug that specifically modulates a gene has the same structure, and that is the signal
   Check 1b exists to detect.

So the question was answered here, on this project's own data.

## Design

L1000 and Tahoe share **7 cell lines** and **14 drugs** by PubChem CID [job 31660822], so the
same (cell line, drug) perturbation exists on both platforms -- one imputing most of its
transcriptome, one measuring all of it. For each shared pair, the L1000 delta is correlated
(Spearman) against the Tahoe delta separately within three gene classes read from
`GSE92742_Broad_LINCS_gene_info.txt.gz`:

| class | definition | what it is |
|---|---|---|
| `landmark` | `pr_is_lm = 1` | directly measured on the L1000 platform |
| `bing` | `pr_is_bing = 1`, `pr_is_lm = 0` | the Broad's own "best inferred" subset |
| `other` | `pr_is_bing = 0` | the remaining inferred genes |

**Comparing classes within the same pair is what makes this work.** Platform chemistry, dose,
timepoint and cell-line identity are properties of the pair, so they act on every gene class in
that pair equally and cannot produce a gap between classes. The landmark class doubles as the
positive control: it is measured on both platforms, so it establishes what agreement looks like
when nothing is imputed.

## Controls

The naive landmark-vs-imputed comparison has two ways to be wrong, and both are addressed.

**Variance matching.** Landmark genes were not chosen at random -- they were selected to be
informative and highly expressed, so they vary more, and rank correlation rises with dynamic
range. A landmark-vs-imputed gap could therefore be an artifact of *which genes were chosen*
rather than of *how they were produced*. Imputed genes are resampled to match the landmarks'
Tahoe-side variance decile profile, at equal gene count. **The variance-matched comparison is
the one to trust**; the raw one is reported alongside it to show what the confound was worth.

**Permutation null.** Correlating deltas from two platforms has a nonzero floor: genes share
structure (expression level, pathway co-regulation) regardless of whether the perturbation
matches. Pairing each L1000 delta with a *different* pair's Tahoe delta measures that floor
per gene class, so an imputed-gene correlation can be read against what mismatched pairs
already produce rather than against zero.

## What a positive result would and would not mean

Imputed genes are a deterministic function of the landmarks. If the landmarks agree with Tahoe,
a linear function of them will agree partially too. **Correlation above null is therefore
expected, and is not evidence that imputed genes add information.** What this test measures is
how much fidelity survives imputation, not whether imputed genes contribute anything beyond
the 978.

The stronger question -- do imputed genes add signal beyond what the landmarks already carry --
needs a different test: predict Tahoe's delta for an imputed gene from (a) L1000's imputed
value versus (b) the best linear predictor built from L1000's landmarks alone. If (a) and (b)
perform equally, the imputed genes are redundant with the landmarks by construction. That test
is not run here and is the recommended follow-up.

## Reproducing

```
sbatch scripts/alpine/25_l1000_imputation_fidelity.sbatch
```

Serial deliberately, against the standing rule to parallelise on Alpine. The cost is one pass
over the `.gctx` restricted to the shared lines and drugs -- a few thousand wells out of 1.3M.
A per-pair split would re-open and re-seek the same `.gctx` per task, and the DMSO baseline is
shared across all pairs of a cell line, so a per-pair split would recompute it identically N
times. Everything after the read takes seconds.

## Results

**32 matched (line, drug) pairs**: 7 cell lines (A549, HS578T, HT29, LOVO, NCIH596, RKO,
SW480) x 19 shared drugs at 24 h, from 180 treated and 180 DMSO wells. Gene classes among the
12,328 L1000 genes: **978 landmark, 9,196 BING, 2,154 other**; after intersecting with Tahoe,
946 / 8,833 / 2,069 [job 31661570].

### Why the positive control fails: measured separately [job 31661769]

The fidelity test's landmark class is the positive control, and it failed. A follow-up job
measured why, over the same 32 pairs and 978 landmark genes
[`docs/results/l1000_tahoe_agreement_summary.csv`]:

| quantity | value (95% CI) |
|---|---|
| **L1000 split-half reproducibility (noise ceiling)** | **+0.5721 [+0.4885, +0.6557]** |
| cross-platform rho | +0.0410 [+0.0054, +0.0766] |
| sign concordance, all 978 landmarks | 0.5126 [0.4999, 0.5252] -- p = 0.0515 vs 0.5 |
| sign concordance, top-100 genes Tahoe moved | 0.5334 [0.4958, 0.5711] -- p = 0.0800 vs 0.5 |
| median abs delta, L1000 | 0.5753 [0.4866, 0.6640] |
| median abs delta, Tahoe | 0.1972 [0.1665, 0.2279] |

**The failure is not noise.** Splitting each pair's treated and DMSO wells into disjoint halves
and correlating the two independent deltas gives **0.572** -- L1000's own delta reproduces
itself well. So the achievable ceiling was 0.57 and the cross-platform result reached 0.041,
about 7% of it. The platforms genuinely disagree; L1000 is not simply measuring nothing.

**The perturbations are not in the same direction.** Sign concordance is 51.3%, against a 50%
chance rate (p = 0.0515). Restricting to the 100 genes Tahoe says moved most -- where signs
should be most reliable -- it is 53.3% (p = 0.0800). Neither is distinguishable from chance.

**Magnitudes are not comparable, and the ratio should not be read as effect size.** L1000's
median absolute delta is 3.3x Tahoe's, but the two are in different units: L1000 Level 3
normalized expression versus Tahoe logCPM log-fold-change. What the number rules out is the
possibility that L1000's deltas are flat or absent. It says nothing about which platform sees
a larger biological effect, and no unit conversion was performed.

### Can a transformation recover it? No [job 31661918]

Seven per-gene normalisations, each scored against its **own** mismatched-pair null
[`docs/results/l1000_tahoe_transform_sweep.csv`]:

| transform | mean rho | null | lift | p (corrected) |
|---|---|---|---|---|
| none | +0.0410 | +0.0039 | +0.0371 | 0.0005 |
| center_per_gene | +0.0499 | -0.0052 | +0.0551 | 0.0005 |
| zscore_per_gene | +0.0453 | +0.0039 | +0.0414 | 0.0005 |
| rank_per_gene | +0.0425 | -0.0020 | +0.0445 | 0.0005 |
| robust_scale_per_gene | +0.0436 | +0.0020 | +0.0416 | 0.0005 |
| drop_pc1 | +0.0358 | +0.0009 | +0.0348 | 0.0055 |
| zscore_then_drop_pc1 | +0.0336 | +0.0032 | +0.0304 | 0.0030 |

[job 31676845, p corrected -- see the banner above]. Everything lands between 0.034 and 0.050
against a reachable ceiling of **0.572**, and now every transform clears its own null. Even the
best, per-gene centering, only moves the raw 0.041 to 0.050 -- statistically real but still far
below the ceiling.

Only per-GENE transforms were tested, and that is not an omission: rank correlation is
invariant to any monotone transform applied within a profile, so per-profile rescaling cannot
change the number by construction. Each transform is scored against a null recomputed under
that same transform, because a transform that inflates every correlation -- including between
unrelated perturbations -- would otherwise read as an improvement.

**So the disagreement is not closed by a per-gene scale or offset transform.** [Corrected: it
IS real and clears its null in every transform tried, but no transform lifts it from ~7% of
ceiling to anywhere near 0.572.] A learned mapping such as L1000toRNAseq is a different and
untested object; what this rules out is the class of normalisations that could plausibly have
closed most of the gap.

**Agreement appears only where the effect is very large.** Bortezomib, a proteasome inhibitor
with a massive transcriptional response, reaches cross-platform rho 0.399 in HT29 and 0.218 in
A549, with top-gene sign concordance 0.84 and 0.71. It is the only drug that clearly separates
from the rest. That is consistent with agreement requiring an effect large relative to the
between-experiment differences, but it rests on two pairs and is an observation, not a result.

Most pairs have 3-8 treated wells against 60 DMSO wells, so the split-half is often 1-vs-2
treated wells and the 0.572 ceiling is if anything an underestimate of what more replication
would give.

### The positive control [corrected: it passes]

| gene class | genes | mean rho | null mean | lift | p vs null (corrected) |
|---|---|---|---|---|---|
| landmark | 946 | 0.0410 | 0.0018 | +0.0392 | 0.0005 |
| bing | 8,833 | 0.0193 | -0.0025 | +0.0218 | 0.0035 |
| other | 2,069 | -0.0003 | -0.0059 | +0.0057 | 0.1219 |

[job 31676844, p corrected -- see the banner above]. **Landmark genes are measured on both
platforms, and they DO clear their own null** (p = 0.0005), as do bing genes (p = 0.0035).
Absolute agreement between L1000 and Tahoe deltas on genes with real information content is
therefore established, if weak -- 0.041 against a 0.572 ceiling. Fully-imputed "other" genes do
not clear their null (p = 0.1219), consistent with them carrying markedly less signal.

Why this aggregation fails is worth recording: it compares each class's *mean* against a null
of mismatched pairs, and pair-level noise -- dose, replicate count, effect size -- varies
enormously between pairs and swamps it. None of that variation is about imputation.

### The within-pair comparison, which is what the design calls for

Each pair is its own control, so pair-level noise cancels [`l1000_imputation_fidelity_paired.csv`]:

| comparison | pairs favouring first | median drho | Wilcoxon p |
|---|---|---|---|
| landmark - bing | 22 / 32 | +0.0136 | **0.0034** |
| landmark - other | 21 / 32 | +0.0184 | **0.0156** |
| bing - other | 20 / 32 | +0.0183 | 0.0595 |
| **landmark - imputed, variance-matched** | **19 / 32** | +0.0183 | **0.2699** |

The raw ordering is monotone and significant, exactly as imputation would predict: measured >
best-inferred > other-inferred. **The last row is the one that matters.** Once imputed genes
are matched to the landmarks' variance profile at equal gene count (946 vs 946), the gap falls
to p = 0.2699 and is no longer significant.

### What that means

Landmark genes were selected to be informative and highly expressed, so they vary more, and
rank correlation rises with dynamic range. The variance-matched control exists to separate
*which genes were chosen* from *how they were produced* -- and it says the raw landmark
advantage is substantially attributable to selection rather than to imputation. **This test
does not establish that L1000's imputed genes are less faithful than its measured ones on
deltas.**

Note the median difference barely moves between the raw and matched comparisons (+0.0136 to
+0.0183); what changes is the significance, because matching costs gene count and adds sampling
noise. So this is an underpowered null, not a demonstration of equivalence. It does not show
the classes are the same either.

### The one suggestive signal

Degradation is clearest where the perturbation is strongest (per-pair artifact, hash-recorded;
see Provenance):

| pair | landmark | bing | other |
|---|---|---|---|
| HT29 / bortezomib | +0.399 | +0.260 | +0.063 |
| A549 / bortezomib | +0.218 | +0.139 | +0.053 |
| HS578T / trametinib | +0.206 | +0.132 | +0.025 |
| HT29 / azd8055 | +0.182 | +0.088 | +0.006 |

Only 4 of 32 pairs exceed rho 0.15 on landmarks; the median is +0.012. Bortezomib is a
proteasome inhibitor with a very large transcriptional effect, and it shows the ordering
cleanly. This is consistent with the test being effect-size limited rather than the classes
being equivalent -- but it is a post-hoc observation on 4 pairs and is **not** evidence.

## What this does and does not change

**It does not justify dropping L1000 on fidelity grounds** -- if anything less so now that
agreement on measured + bing genes is confirmed real, just weak. That argument is not supported
by this measurement.

**The a priori argument is untouched and does not depend on this test.** L1000's imputed genes
are a deterministic linear function of the 978 landmarks. Whatever their fidelity, they add no
independent measurement: a panel of 12,328 L1000 genes carries at most 978 genes of
information. That is a property of `INF_mlr12k` construction, not a hypothesis, and it is the
sound reason to prefer measured platforms -- not the fidelity claim this test failed to
establish.

**The panel decision therefore still rests on drug and cell-line coverage**, where L1000 wins
decisively: 25 of 34 organoid drugs and 76 cell lines, against Tahoe's 10 and 50 and PANACEA's
8 and 11 [job 31660822, 31660841].

## Recommended follow-ups

1. **Use the paired LINCS/GTEx samples (GSE92743) instead.** The confound here is that L1000
   and Tahoe are different experiments on different platforms, so even measured genes barely
   agree. GSE92743 has L1000 and RNA-seq on the *same* samples, removing the cross-experiment
   term entirely. It is the design that should have been run first.
2. **Test redundancy, not fidelity.** The sharper question is whether imputed genes add
   anything beyond the landmarks: predict Tahoe's delta for an imputed gene from (a) L1000's
   imputed value versus (b) the best linear predictor built from L1000 landmarks alone. If
   (a) ~ (b), the imputed genes are redundant by construction. This does not need cross-platform
   agreement to be high, so it survives the power problem that limited this test.
3. **Restrict to high-effect perturbations** with a pre-registered effect-size threshold, rather
   than the post-hoc look above.

## Provenance

| artifact | contents |
|---|---|
| `docs/results/l1000_imputation_fidelity.csv` | per-class summary with permutation null |
| `docs/results/l1000_imputation_fidelity_paired.csv` | paired within-pair signed-rank tests |
| `docs/results/l1000_imputation_fidelity_per_pair.provenance.json` | every (line, drug) pair, all classes -- **hash only, contents not committed** |

Each carries a `.provenance.json` with the git sha, job id, resolved arguments, hashed inputs
and the run log's hash.

The per-pair table is recorded by hash rather than committed. `scripts/check_release.py`
refuses it because it carries a `line` column, and the gate treats any sample-identifier column
as row-level data. Whether cell-line rows fall inside this project's embargo is a policy
question for the release manifest's owner, not something to settle by amending the gate --
`line` holding A549 is a public CCLE identity, while `line` holding SARC0065 is a
patient-derived organoid, and the column name cannot distinguish them. Until that is decided,
the contents stay out of the repo and the hash preserves the ability to verify a rerun. Jobs: 31661481 (first run), 31661545 (failed, non-existent output
directory), **31661570** (the promoted run).

**Disclosure.** The within-pair design was fixed in advance and is stated in the script
docstring. The choice of paired signed-rank *statistic* was made after seeing that the marginal
aggregation was uninformative, and the variance-matched paired test was added in the same edit.
The variance-matched arm is the one that reverses the conclusion, so this ordering is recorded
rather than smoothed over.
