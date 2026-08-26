# Do L1000's imputed genes carry real delta signal?

**Status:** results pending (job 31661349). Design and methods below are final; the Results
section is filled from the promoted artifact, not from the run log.

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

_Pending job 31661349._
