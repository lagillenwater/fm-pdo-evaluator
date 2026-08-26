# Do L1000's imputed genes carry real delta signal?

**Result: not established.** The raw comparison shows imputed genes degrading monotonically
away from measured ones, but that gap does **not** survive the variance-matching control, so
this test does not demonstrate that L1000's imputed genes are less faithful on deltas.
Evidence: `docs/results/l1000_imputation_fidelity{,_paired,_per_pair}.csv` [job 31661570].

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

### The positive control fails

| gene class | genes | mean rho | null mean | lift | p vs null |
|---|---|---|---|---|---|
| landmark | 946 | 0.0410 | 0.0018 | +0.0392 | 0.2438 |
| bing | 8,833 | 0.0193 | -0.0025 | +0.0218 | 0.3234 |
| other | 2,069 | -0.0003 | -0.0059 | +0.0057 | 0.4378 |

**Landmark genes are measured on both platforms, and they do not clear their own null**
(p = 0.2438). Absolute agreement between L1000 and Tahoe deltas is therefore **not
established**, and no absolute claim about any gene class can rest on this comparison. Read the
0.041 as "indistinguishable from mismatched pairs", not as "weak agreement".

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

**It does not justify dropping L1000 on fidelity grounds.** That argument is not supported by
this measurement.

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
