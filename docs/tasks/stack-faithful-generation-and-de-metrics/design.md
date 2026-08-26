# Faithful Stack generation procedure + DE-based Check-1 metrics — design

**Date:** 2026-08-18
**Status:** approved (adversarially reviewed), pending implementation plan

## Purpose

Three changes to the Tahoe generation-eval harness, following up on the completed leakage-aware
Check-2 plan and a full read of the Stack paper (Dong et al. 2026, "Stack: In-Context Learning
of Single-Cell Biology," PMC12803207, bioRxiv posted 2026-06-08). The first two were the
original scope; the third was discovered while resolving one of this design's own open items
and gates the other two's real runs on the drug-aligned checkpoint:

0. **Change 0 — fix a confirmed sci-Plex ingestion bug.** `scripts/build_sciplex_finetune.py`
   silently mislabels 36,522 cells with no recoverable identity (failed hash-demultiplexing
   calls in the original sci-Plex 3 data) as vehicle controls, inflating the control pool from
   a true 17,578 to 54,100. The existing drug-aligned checkpoint — and every downstream number
   derived from it, including already-published Check-1/Check-2 results — was fine-tuned with
   this bug present.
1. **Change 1 — faithful generation procedure.** `scripts/alpine/04_stack_generate.sbatch`
   currently runs `--mode vanilla`, a workaround that skips the paper's actual described
   generative procedure (Methods 4.2.5: scheduled context ratio + confidence-guided selective
   unmasking). Switch to the CLI's true default (`--mode mdm`).
2. **Change 2 — faithful DE-based metrics.** Check 1 (`delta_fidelity`/`score_delta_sources`,
   `src/fmharness/evaluation.py`) currently scores only Pearson-Delta. The paper's own Check-1
   analogue (cell-eval, Methods 4.6.3) reports DE Spearman LFC, PR-AUC, DE Overlap Accuracy, and
   Jaccard similarity alongside Pearson Delta — and its own data shows Pearson Delta is the
   metric on which Stack's advantage is weakest or absent (Fig. S9: -8.0%, -2.1% in two of its
   own ablation panels; Parse's own headline number is -0.7% on Pearson Delta specifically,
   Methods 4.7/Fig. 3F). Check 1 has, by construction, been scoring the one axis the paper's own
   data says is least informative about this model's generative advantage.

Changes 1 and 2 were adversarially reviewed against the real `ArcInstitute/stack` source (not
paper prose alone) before this design was finalized — see "Design history" below for what that
review overturned. Change 0 was found and verified via a full crosstab against both the live
scPerturb-hosted file and the raw NCBI GEO deposit — see "Change 0" below.

## Why now, not deferred

This project's Check-1 finding (Stack generation is null on Tahoe, r=0.012-0.021, both
checkpoints) already has one confirmed, sufficient explanation: the paper never benchmarked
generation on Tahoe at all (Table 3, Methods 4.6.4 enumerate every generation-eval dataset used;
Tahoe appears only once in the whole paper, as an *embedding* classification benchmark, Fig. 2E
— a different capability). But two further factors could independently be inflating the null,
and both are now confirmed real, not hypothetical:

- The current `--mode vanilla` invocation never exercises the model's confidence-guided
  unmasking at all (see below) — a materially simpler procedure than what the paper describes
  and reports results for.
- Check 1 has never scored the DE-based metrics on which the paper's own data shows Stack's
  advantage concentrates.

Fixing both closes the gap between "what we measured" and "what the paper's own methodology and
results actually support," so the eventual Check-1 number (whatever it turns out to be) is
comparable to the paper's claims on its own terms.

## Design history — what adversarial review overturned

An earlier draft of this design (mid-conversation, not committed) proposed a stratified,
duplication-avoiding sampling scheme for the *base/prompt* cell pool, based on a
misidentification of where duplication risk lives in the CLI's sampling code. A dedicated
adversarial-review pass (four independent reviewers, each trying to refute one claim, against
the real `ArcInstitute/stack` source fetched fresh via `gh api`) found:

- **No duplication risk in the base/prompt pool, at any drug, ever** (survived challenge, three
  independent proofs: DMSO cannot leak into a drug's split; within-chunk base draws are
  mathematically duplicate-free by construction, since `n_base_cells <= 333` is drawn from a
  pool of `>= 9,198` real cells per drug; and even a hypothetical cross-chunk overlap would be
  architecturally inert, since PyTorch's batched attention never mixes batch indices). The
  stratified-pool proposal solved a problem that does not exist at this pipeline's actual scale
  — dropped entirely.
- **The real failure mode on the query side is a crash, not silent duplication** (this is the
  one place the original framing was substantively wrong, not just imprecise). Padded/looped
  query cells are computed by the model but their predictions are discarded before the returned
  array is built (`is_test_cell_mask` is captured *before* padding is appended); the actual
  bug is that `pad = test_indices[:need]` truncates silently if `need > len(test_indices)`,
  shrinking the batch below what the downstream boolean-index step expects, which is what raises
  the `IndexError` `--mode vanilla` was adopted to route around. The correct fix target is
  therefore "pool size avoids the crash" (`pool >= max(n_test_cells)`), not "pool size avoids any
  padding at any step" (provably impossible — the five schedule steps' thresholds are pairwise
  near-coprime).
- **The aggregation step (multiple generated replicate predictions per line -> one row per line
  for scoring) was genuinely underspecified, not just missing detail.** Naive averaging is a
  bias problem, not a variance problem: a low-confidence replicate is not a noisy-but-unbiased
  draw, it is a prediction the model itself judged as still resembling the unresolved/masked
  population, which is mechanistically pulled toward the control baseline under weak context
  support. This is exactly the paper's own stated reason for filtering before aggregating in its
  structurally identical Perturb Sapiens construction (Methods 4.7: "We removed cells with
  classifier logits above 2.5," and Results 2.4's own ablation: "Evaluations without confidence
  filtering results in reduced cell-type specificity, confirming the essence of the procedure").

The corrected design below reflects all three findings.

## Change 0 — fix a confirmed sci-Plex ingestion bug (prerequisite for Change 1's drug-aligned run)

Discovered while resolving Change 1's "real per-cell-line control-pool sizes" open item
(2026-08-18 smoke test). Not part of the original two changes this design set out to make, but
gates Change 1's drug-aligned generation run: that run needs a correctly-fine-tuned checkpoint,
and the existing one was fine-tuned with this bug present.

### What the smoke test found, and what it turned out to mean

The real per-cell-line control-pool query returned:

```
cell_line
A549     5,857
K562     3,935
MCF7     7,786
nan     36,522
```

The `nan` bucket — 67.5% of all 54,100 cells `build_sciplex_finetune.py` currently calls
"control" — was initially read as a large, alarming data-quality gap in the published sci-Plex 3
release. It is not. Independent verification against both the live scPerturb-hosted file
(Zenodo record 13350497, `SrivatsanTrapnell2020_sciplex3.h5ad`, read directly via HTTP range
reads) and the raw NCBI GEO deposit (`GSM4150378_sciPlex3_pData.txt.gz`) it derives from, via a
full crosstab over all 799,317 rows (not a sample), found:

- The true rate of missing `cell_line` in the published file is **4.57%** (36,522 of 799,317),
  not 67.5% of anything.
- Those 36,522 rows are cells whose nuclear hash-oligo call was too ambiguous for the *original
  sci-Plex authors' own* well/plate identity lookup to resolve (spot-checked top-to-second-best
  hash purity ratios ~1.0-2.0 on these rows, vs. 5-275+ on confidently-assigned cells) — when
  that happens, the authors' own pipeline blanks `cell_line`, `perturbation`, `dose`, `treatment`,
  `well_oligo`, and `plate_oligo` together as one bundle, not as an independent per-column
  dropout. This is expected behavior in a pooled multi-line hashing experiment, already handled
  as "ambiguous, drop" by the original authors — not a scPerturb reprocessing defect and not
  something this project's ingestion introduced.
- Crucially, **the crosstab confirms 0 of the 36,522 are real controls and 0 are real treated
  cells** — they carry no usable identity for either category. The true control population is
  exactly 17,578 cells (5,857 + 3,935 + 7,786 — matches the per-line breakdown above exactly),
  and the true treated population is 745,217, with **zero missingness among treated cells**.

### The actual bug, confirmed in `scripts/build_sciplex_finetune.py`

```python
VEHICLE_NAMES = {"control", "vehicle", "dmso", "none", "nan"}
...
pert = a.obs[pert_col].astype(str).to_numpy()   # a real NaN silently becomes the STRING "nan"
...
is_ctl = np.array([p.strip().lower() in VEHICLE_NAMES for p in pert])
```

The raw scPerturb file's real control indicator lives as a value (`'control'`) inside the
`perturbation` categorical column itself, not as a separately-named flag column, so
`CTRL_FLAG_CANDIDATES = ["control", "is_control", "vehicle"]` never matches anything and the
script falls into this string-matching fallback. `VEHICLE_NAMES` including the literal string
`"nan"` means any cell whose real perturbation is missing (and therefore stringifies to `"nan"`)
gets swept into `is_ctl=True` alongside genuine vehicle controls. The arithmetic confirms this
exactly: 17,578 (real controls) + 36,522 (unassignable) = **54,100** — precisely the
`"controls 54100/799317"` the original `08_sciplex_prep.sbatch` run logged.

**Consequence for the already-completed fine-tune:** `pert_id` is fine-tuning's grouping axis,
and `"control"` is one of the 189 groups; whenever a 512-cell training block is drawn from that
group, roughly two-thirds of its cells are these unassignable cells rather than real untreated
biology — the model was trained on a "control" reconstruction target contaminated by ambiguous
hash calls for ~67.5% of that group's own samples. (Whether this also contaminates the
per-cell-line control-*replacement* candidate pools used for every *other* drug group's training
samples is reasoned-but-not-independently-verified: `identity_col=cell_line`, and these
unassignable cells carry `cell_line="nan"`, a fourth identity value distinct from
A549/K562/MCF7 — plausibly excluding them naturally from real-identity replacement lookups — but
this hasn't been traced against the actual `ArcInstitute/stack` fine-tuning source the way
everything else in this design has been, so it is stated as a hypothesis, not a fact.)

### The fix

Two parts, both in `scripts/build_sciplex_finetune.py` / `src/fmharness/sciplex_prep.py`:

1. **New filter, applied before any `is_control` detection**, on the *raw*, pre-stringified
   columns (checking `.isna()` before `.astype(str)` runs — stringifying first is exactly what
   turns real missingness into the deceptive `"nan"` string this bug exploited):

   ```python
   def identity_missing_mask(pert: pd.Series, cell_line: pd.Series) -> np.ndarray:
       """True for cells whose raw perturbation and/or cell-line identity is missing.

       sci-Plex 3's own nuclear-hash demultiplexing blanks perturbation/cell_line/dose/well/
       plate together when a hash call is too ambiguous to resolve (a real, ~4.6% fraction of
       the published release, concentrated in no single column) -- these cells have no usable
       identity as either "control" or "treated" and must be dropped before any downstream
       is_control detection, not swept in via a stringified-NaN string match against a value
       like "control"/"vehicle"/"nan".
       """
       return pert.isna().to_numpy() | cell_line.isna().to_numpy()
   ```

   `build_sciplex_finetune.py`'s `main()` calls this immediately after reading `pert_col`/
   `line_col` (before the existing `pert = a.obs[pert_col].astype(str).to_numpy()` line),
   drops the matched rows, and prints the count dropped — matching the existing
   `--min-cells-per-cond` filter's own established convention of a visible, counted drop rather
   than a silent one.
2. **Remove `"nan"` from `VEHICLE_NAMES`.** Once the explicit filter above runs first, no
   genuinely-null cell should ever reach this fallback; leaving `"nan"` in the set is a
   redundant landmine for any future sci-Plex-shaped file where the explicit filter isn't
   applied first, or is applied in the wrong order.

### Required re-run, and its effect on already-published numbers

`08_sciplex_prep.sbatch` (produces `sciplex_finetune.h5ad`) and `09_stack_finetune.sbatch`
(produces the drug-aligned checkpoint) both need re-running with the fix before Change 1's real
generation run on the drug-aligned checkpoint. **This means the drug-aligned Check-1/Check-2
numbers already written into `docs/tahoe_generation_results.md` and the harness-overview deck**
(from the just-completed leakage-aware Check-2 SDD plan — r=0.021 both variants, plus the
Check-2 ladder/fixed-signature rows) **were derived from the contaminated checkpoint and will
need re-deriving once the corrected one exists.** Folded into this design's scope per explicit
direction (not deferred to a separate plan) — the implementation plan's real-run task should
re-run `08`/`09` first, then feed the corrected checkpoint into Change 1's generation work and
re-derive the downstream Check-1/Check-2 rows, rather than treating the prior numbers as still
valid.

Group-eligibility mechanics measured in "Batch construction" below (all 189 groups clear both
the 128- and 512-cell floors) are very likely unaffected by this fix in *outcome* — the true
control count (17,578) still clears 512 by a wide margin, same as every other group — but this
must be re-measured after the re-run, not assumed to carry over unchanged, since removing
36,522 rows (4.57% of the total dataset) changes the underlying counts even if it probably
doesn't flip which groups qualify.

## Change 1 — faithful generation procedure

### What the CLI actually does (verified against `ArcInstitute/stack@main` source, not paper prose)

`get_incontext_generation` (`src/stack/models/core/inference.py:841`) runs `num_steps=5` (the
CLI's own default) regardless of `--mode`. The two modes differ in what happens *within* each
step:

- `mode='vanilla'` (current): fixed `context_ratio` every step, decreasing `mask_rate` schedule
  (`alpha=1-1/t`) — but does **not** carry `is_masked`/`test_logit` between steps, so the
  model's confidence-guided selective-unmasking classifier (`self.cls`, trained specifically for
  this purpose) is never consulted. A materially simpler baseline procedure the CLI happens to
  also expose, not "one-shot generation" as informally described earlier in this project's
  history.
- any other `--mode` value (CLI default: `"mdm"`) hits the branch that matches Methods 4.2.5
  exactly: `context_ratio` scheduled via `cr_list = linspace(context_ratio_min=0.2,
  context_ratio=0.4, 5)`, and `is_masked`/`test_logit` threaded through every step so the
  classifier's confidence score drives which query cells get finalized each round.

### The root cause of the current 50-cell-query crash, and the corrected fix

`n_test_cells = max(1, int(self.n_cells * (1 - prompt_ratio - context_ratio)))`, recomputed
per-step under the scheduled `context_ratio`. At `self.n_cells=512` (confirmed for both
checkpoints this project uses — see "n_cells verification" below) and the CLI's own
`prompt_ratio=0.25` default:

| step | context_ratio | n_test_cells | n_base_cells |
|---|---|---|---|
| 1 | 0.20 | 281 | 231 |
| 2 | 0.25 | 256 | 256 |
| 3 | 0.30 | 230 | 282 |
| 4 | 0.35 | 204 | 308 |
| 5 | 0.40 | 179 | 333 |

The current `tahoe_query.h5ad` has exactly 50 rows (one pseudobulk-mean row per DepMap line,
`03_stack_context.sbatch`'s `_group_mean` collapse of the 200 real control cells/line that
`tahoe_context.h5ad` already has). At every one of the 5 steps, `need = n_test_cells - (50 mod
n_test_cells)` exceeds the 50-cell pool, so the silent-truncating pad crashes downstream with an
`IndexError` — this, not row duplication, is what `--mode vanilla`'s fixed-ratio, exact-50-fit
sizing (`--prompt-ratio 0.501 --context-ratio 0.4`) was built to avoid.

**Fix: replace the pseudobulk collapse with 8 real control cells/line (400 total)**, comfortably
inside the confirmed 200-real-cells/line ceiling (no thinness — Tahoe's real single-cell DMSO_TF
control counts were already confirmed uniform, exactly 200/line, 50/50 lines, in prior work on
this branch). Verified against all 5 real schedule steps that 400 never crashes:

| step | N | 400 mod N | need | need > 400? |
|---|---|---|---|---|
| 1 | 281 | 119 | 162 | no |
| 2 | 256 | 144 | 112 | no |
| 3 | 230 | 170 | 60 | no |
| 4 | 204 | 196 | 8 | no |
| 5 | 179 | 42 | 137 | no |

400 clears the sufficient bound (`pool >= max(n_test_cells) = 281`) with ~40% headroom, keeping
worst-case wasted compute (padded-but-discarded cells, not output duplication) moderate across
all 5 steps.

### `03_stack_context.sbatch` changes

- Replace the `_group_mean` pseudobulk collapse (writes 50 rows, one mean profile per line) with
  a per-line sample of 8 real single control (`is_control=True`) cells, drawn without replacement
  (200 real cells/line always available — no replacement path needed or wanted here).
- Preserve a `cell_line_id` obs column per row (not just a collapsed `patient` index) — the
  aggregation step in Change 1's second half needs this as its groupby key.
- **Fix the stale `--partition=amilan` reference while this file is touched anyway.** Confirmed
  via `sinfo` that `amilan` no longer appears in Alpine's partition list at all (fully retired,
  not just renamed) — several other sbatch scripts in this repo (`01`, `02`, `05`, `07`, `00`,
  `delta_reproducibility`) still reference it too, but fixing those is out of scope here; this
  design only touches `03` because Change 1 already modifies it.

### `04_stack_generate.sbatch` changes

Drop `--mode vanilla --prompt-ratio 0.501` entirely. Use the CLI's true defaults, made explicit
rather than implicit (matching this repo's existing "log resolved CKPT/GENELIST/OUTDIR"
reproducibility discipline, and guarding against a future package upgrade silently changing a
default):

```
--mode mdm --prompt-ratio 0.25 --context-ratio 0.4 --context-ratio-min 0.2
```

**The mode and prompt-ratio changes are not independent — both must move together.** If
`--prompt-ratio 0.501` were left in place while only dropping `--mode vanilla`, the resulting
schedule (`ratio = 0.501 + cr`) would range 0.701-0.901, giving `n_test_cells = {153, 127, 101,
76, 50}` — a different, still partly-broken range under which even the corrected 400-cell query
pool would still be sized against the wrong target (each step now needs far fewer query cells
than provisioned, which is harmless — the pool being *oversized* relative to a step's need never
crashes — but reintroduces the original vanilla-specific reverse-engineering this change exists
to remove). Both flags change together, sized against the new 400-cell pool.

### Aggregation step (new)

Once `mode != 'vanilla'`, each generated query cell carries a real confidence logit, already
surfaced by the CLI for free: `pred_adata.obs["gen_logit"]` (`src/stack/cli/generation.py`,
whenever `test_logit is not None`). Sign convention, confirmed against both the source and the
paper's Methods 4.2.5 verbatim: **positive logit = still looks like an unresolved query cell;
negative/low logit = looks like confidently-resolved real data.**

New step, inserted between Stack's raw per-drug `.h5ad` output and
`build_generated_deltas`/`delta_fidelity` (which require exactly one row per (line, drug)):

1. **Filter, then average — not average alone.**
   `df.groupby("cell_line_id").apply(lambda g: g[g.gen_logit < threshold].X.mean(axis=0))`.
2. **Do not hardcode the paper's `2.5`.** That value is calibrated to Perturb Sapiens' own
   checkpoint/logit scale, not to this project's further sci-Plex-finetuned classifier head,
   which can shift the scale. Calibrate on this checkpoint instead: compare Check-1 Pearson-Delta
   with vs. without filtering at a few candidate thresholds and confirm filtering improves
   (not degrades) it before committing to a value — the same ablation the paper itself ran
   (Fig. S15) to justify the procedure, not assumed to transfer.
3. **Explicit missing-value handling.** If a (line, drug) group has zero surviving replicates
   after filtering, mark it missing. Do not silently fall back to the unfiltered mean — that
   would reintroduce exactly the bias the filter exists to remove.
4. This step only exists once `mode != vanilla` (`gen_logit` is `None` under vanilla) — the mode
   switch above is a hard prerequisite for it, not merely a paper-fidelity nicety.

### `n_cells` verification — confirmed `512` for both checkpoints

`self.n_cells=512` was first confirmed in source for both checkpoints this project can point
`CKPT=` at (the cytokine-aligned `bc_large_aligned.ckpt`, built from `ArcInstitute/stack`'s own
`configs/finetuning/ft_parsecg.yaml` which pins `sample_size: 512`; and this project's own
sci-Plex-finetuned checkpoint, via `--sample_size 512` in `09_stack_finetune.sbatch`, which
`override_model_config_n_cells` bakes into the checkpoint's own `hyper_parameters` and
`generation.py` reads with no override available at generation time) — but since `n_cells` is a
plain constructor kwarg with **zero effect on any tensor shape** (`TabularAttentionLayer`'s
parameters, and `query_pos_embedding`, are both cell-axis-independent), a wrong value would
never crash, only silently change every `n_test_cells`/`n_base_cells` calculation above, so this
needed a direct file-level check, not just source/config inference. **Now confirmed directly**
(2026-08-18 `ah200` smoke test, job 31418001) against both real checkpoint files:

```
cytokine-aligned: n_cells=512  (stack-aligned/bc_large_aligned.ckpt)
sci-Plex-drug-aligned: n_cells=512  (finetuned-epoch=4-val_loss=5.0847.ckpt)
```

The entire `n_test_cells`/`n_base_cells` schedule table above is now verified against real
checkpoint data, not source/config inference alone. (The sci-Plex-drug-aligned checkpoint here
is the pre-Change-0-fix one — `n_cells` is independent of the control-cell mislabeling bug and
this number is not expected to change once `09` is re-run, but the corrected checkpoint should
still be spot-checked the same way once it exists, as routine hygiene rather than because this
specific value is suspected to differ.)

## Change 2 — faithful DE-based Check-1 metrics

### Why a new precomputed bundle, not inline computation

The paper's DE calling (Wilcoxon rank-sum + Benjamini-Hochberg FDR<0.05, |log2FC| threshold,
Methods 4.6.3) needs per-cell data to produce a p-value. `build_tahoe_deltas`
(`src/fmharness/deltas.py:615`) collapses straight to pseudobulk means with no per-cell or
significance information retained, and this project's established `--deltas-bundle` shortcut
(preferred for reproducibility — see `check1_registry_driver.py`'s own documented
`--context`-vs-`--deltas-bundle` divergence caveat from 2026-08-12) is a cache of that
already-collapsed output; there is no path back to per-cell data from it. Faithful DE calling
needs the raw per-cell `tahoe_context.h5ad`, and the Wilcoxon test itself (~1,650 (line, drug)
treated-vs-control pairs × ~15,012 genes) is real, one-time compute — cache it, matching the
existing `tahoe_deltas/` bundle pattern, rather than repeating it on every Check-1 run.

### New build step: `build_tahoe_de_calls.py` (or similar; exact name at planning time)

- Input: `tahoe_context.h5ad` (raw per-cell, already in this worktree).
- For each of the ~1,650 (line, drug) pairs: Wilcoxon rank-sum test per gene (treated vs. that
  line's control cells), Benjamini-Hochberg FDR correction, `|log2FC| > threshold` (paper's own
  value; confirm the exact threshold from Methods 4.6.3/6.4 at planning time rather than
  guessing) to call significance.
- Output: a cached bundle (parquet, matching `tahoe_deltas/`'s existing `real_delta.parquet`/
  `real_key.parquet` shape) of per-(line, drug, gene) DE calls (log2FC, adjusted p-value,
  significant flag) — the ground-truth side only. The **predicted** side needs no test: ranking
  by `|log2FC|` alone is enough for DE Overlap Accuracy/PR-AUC/Jaccard, since only the ground
  truth needs a formal significance call.

### `delta_fidelity`/`score_delta_sources` extension

Add DE Spearman LFC, PR-AUC, DE Overlap Accuracy, and Jaccard similarity (paper's own cell-eval
definitions, Methods 4.6.3) alongside the existing Pearson-Delta scoring, computed against the
new DE-calls bundle. Exact function signatures and whether this becomes a new `score_de_metrics`
alongside `delta_fidelity` (matching this project's existing pattern of one function per metric
family, composed by a `score_*` driver — see `score_delta_sources`/`score_check2`) or an
extension of `delta_fidelity` itself is a planning-time decision, not fixed here.

## Sci-Plex fine-tuning — confirmed mechanics (real numbers, Alpine reconnected 2026-08-18)

Pulled directly from the already-completed `sciplex-prep` and `stack-finetune` Alpine job logs
(`ralpine cat` on the full log files — `ralpine log`'s default view truncates/tail-samples, do
not rely on it alone for anything requiring the complete log), plus the corrected control-cell
counts from "Change 0" above. All numbers below are measured, not inferred. **The counts in
this section describe the pre-fix run** (what actually happened on 2026-08-12) — they will
change once `08`/`09` are re-run with Change 0's fix applied; re-measure rather than assume
they carry over.

### Data scale

`build_sciplex_finetune.py`'s own summary line (pre-fix run): **799,317 cells x 110,983
features, 745,217 treated / 54,100 control, 189 perturbation groups, 4 cell identities.** Per
Change 0 above, the true composition is **745,217 treated (zero missingness) / 17,578 real
control / 36,522 unassignable (dropped by the fix, not real controls)** — the pre-fix "54,100
control" figure conflates the latter two.

**Gene-count anomaly, confirmed real, not yet checked anywhere in this pipeline.** The prep run's
own log shows `gene_symbol='var_names'` — none of `build_sciplex_finetune.py`'s candidate
symbol columns (`gene_symbol`/`symbol`/`gene_name`/`feature_name`/`gene_short_name`) existed in
this file's `.var`, so it fell back to the raw index — and anndata's own "Variable names are not
unique" warning fired during a downstream op. 110,983 is far beyond any real human transcriptome
gene count (~20-40K), consistent with duplicate/non-unique var_names (e.g. transcript-level
features, or anndata-auto-suffixed duplicates like `TP53-1`/`TP53-2`).
`check_gene_count`'s existing floor (`min_genes=5000`) only guards against *too few* genes — it
was built to catch a pre-subset HVG release, not this failure mode, and passes 110,983 through
trivially. Real, concrete consequence: `ArcInstitute/stack`'s own generation-time gene alignment
(`_align_genes_to_target_list`, uppercase-string set matching against the model's ~15K-gene
panel) would silently zero-pad any duplicate-suffixed gene that no longer matches its clean
symbol — real signal dropped with no error. **New check needed in `src/fmharness/sciplex_prep.py`**
(same module as the existing raw-counts/gene-count/perturbation-truncation checks): verify the
resolved gene-symbol column produces unique values before accepting it, following the same
"raise with a clear message, don't just warn" pattern `check_raw_counts`/`check_gene_count`
already use.

### Batch construction (verified against `ArcInstitute/stack`'s actual fine-tuning source, `src/stack/data/finetuning/datasets.py`, and cross-checked against the real log's exact numbers — not the paper's prose alone)

For `type='drug'` datasets, `DatasetConfig` swaps roles relative to the human/PBMC case:
`group_col = pert_id` (grouping is by drug condition, pooling across all cell lines and doses of
that drug) and `identity_col = cell_line` (cell line plays the role cell type plays for
donor-grouped data) — a real, code-level role-swap, not an artifact of this project's own
ingestion script.

**Group eligibility and split — measured exactly, log line `Found 189 groups with >= 128
cells`:** every one of the 189 total perturbation groups (188 distinct compounds + the literal
`control` label, matching `build_sciplex_finetune.py`'s own reported "189 perts") cleared the
128-cell eligibility floor — **zero groups excluded**. The 80/10/10 group-level split (log line
`Split groups: 153 train, 18 val, 18 test`, summing to 189) is genuine — group-level, not
cell-level, so a drug is entirely train, entirely val, or entirely test, never split within
itself.

**Usable-sample floor (the two-tier concern raised earlier in this design's own history) —
measured, and the gap does not materialize in this run.** The concern was that a group could
clear the 128-cell eligibility bar but still contribute zero actual training batches, since
batches are drawn as non-overlapping contiguous blocks of exactly `sample_size=512` cells. The
real run's per-split logs (`Skipped 0 combinations with < 512 cells`, all three of train/val/test)
confirm **every one of the 189 groups also had >= 512 cells** — the theoretical two-tier gap this
design flagged as a real risk turned out not to bite for this specific dataset, though the
mechanism (and the risk for a future, thinner dataset) remains real and worth stating in the
methods write-up, not silently dropped now that this run happened to clear it.

**A confusing pair of duplicate-looking log lines, resolved by reading the actual call site
(`create_train_val_test_datasets`, `src/stack/data/finetuning/datasets.py:2038`):** the log
shows two different group/sample counts under the same "train mode" label (115/3891 vs.
153/1221). This is not two competing thresholds — it's a real, if slightly wasteful, artifact of
the library's own implementation: a throwaway `base_dataset` instance is constructed first
*solely* to run group splitting, but its own `__init__` (since it wasn't given pre-computed
`train_groups`/`val_groups`/`test_groups`) calls `_split_groups()` with *that method's own
default* ratios (0.2/0.2), immediately generates a full (and entirely discarded) sample set
under those wrong ratios, and *only after that* does `create_train_val_test_datasets` explicitly
re-call `_split_groups(test_ratio, val_ratio)` with the real 0.1/0.1 CLI-configured ratios,
whose resulting `train_groups`/`val_groups`/`test_groups` are what actually get used to build the
three real datasets. The 115/3891 numbers are pure noise from this discarded first pass and
should never be quoted as real; 153/1221 (train), 18/124 (val), 18/117 (test) are the real,
final numbers.

**Control-replacement mechanism (the actual generative task):** 51 of every 512 cells per
training sample (`int(512 * replacement_ratio=0.1)`) get their **model input** swapped for a
real, same-cell-line vehicle control cell drawn from anywhere in the dataset (ground truth stays
the original treated cell's raw counts) — chosen by iterating cell-line identity groups in
shuffled order until the quota is filled, drawing without replacement when the real candidate
pool is large enough, with-replacement (modulo reuse) only if it is not. **Real per-cell-line
control-pool sizes, now measured** (obtained via the 2026-08-18 smoke test that also surfaced
Change 0's bug — see above): **A549 5,857 / K562 3,935 / MCF7 7,786** real controls (17,578
total; the pre-fix run's own per-line breakdown included an additional 36,522 "nan"-identity
cells now confirmed to be unassignable, not real controls of any line). All three real per-line
pools are comfortably large relative to the ≤51-per-sample draw, so the with-replacement fallback
path is expected to trigger rarely if at all for the genuine cell lines — this was not directly
measured (would need instrumenting the actual training run, not just the static pool sizes), but
the margin (thousands of candidates vs. tens needed per sample) makes frequent triggering
implausible.

**Epoch-level note, worth stating precisely for reproducibility:** `resample_training_data()`
reshuffles sample *order* only, not block *membership* — the same fixed 512-cell contiguous
blocks per drug repeat across all 10 training epochs (only the internal 51-cell replacement draw
and the model's own scheduled context/query split vary draw to draw).

### A confirmed, unrelated bug — does not affect the checkpoint used for generation

The completed fine-tuning run's log shows a crash on its **final** step (`trainer.test(...,
ckpt_path="best")`, after all 10 epochs and every checkpoint save already completed
successfully): `RuntimeError: Missing key(s) in state_dict: "teacher_model...."`. Traced to
source: `LightningFinetunedModel.on_save_checkpoint` (`src/stack/finetune/lightning.py:101`)
deliberately prunes the saved checkpoint to only `model.*` (student) keys, discarding
`teacher_model.*` — but `trainer.test()` tries to reload the *full* Lightning wrapper (student +
teacher) and fails `strict=True` loading on the missing teacher keys. This is a genuine
inconsistency in `ArcInstitute/stack`'s own `launch_finetuning.py`, not anything wrong in this
project's config. **Confirmed it does not affect the checkpoint used for generation**:
`generation.py`'s own loading path (`load_model_from_checkpoint`, `model_class="ICL_FinetunedModel"`)
strips to exactly the `model.*` keys the pruned checkpoint has and loads a bare
`ICL_FinetunedModel` with no teacher branch at all — precisely what the checkpoint contains. Net
effect: the fine-tune's held-out test-split metrics were never computed (an observability gap in
the original 2026-08-12 run, not a correctness bug), and nothing about this needs fixing as part
of this design — noted here so it isn't mistaken for a new problem if seen again in a future
fine-tuning run's logs.

### Draft methods paragraph

**This paragraph describes the pre-fix (2026-08-12) run and is being superseded, not finalized**
— Change 0's fix removes the 36,522 unassignable cells before fine-tuning, so every number below
needs re-measuring after `08`/`09` are re-run. Kept here as the fully source- and log-verified
record of what the *existing* (to-be-replaced) checkpoint was actually trained on, since that
checkpoint's own already-published Check-1/Check-2 numbers are being re-derived against this
exact history, not silently discarded.

> Fine-tuning batches were constructed by ArcInstitute Stack's `MultiDatasetSplittableDataset`
> (`stack.data.finetuning.datasets`), configured via a `drug`-type `DatasetConfig` with
> `condition_col="pert_id"`, `cell_line_col="cell_line"`, and `control_condition="control"`. For
> drug-type datasets, Stack's grouping axis is the perturbation condition itself; each distinct
> value of `pert_id` (188 sci-Plex 3 compounds, or the literal control label) forms one group
> spanning all cell lines and doses of that condition, and cell line serves as the "identity"
> axis that plays the role cell type plays for donor-grouped human datasets. All 189 groups
> cleared the 128-cell eligibility floor and were split 80/10/10 at the group level (153
> train / 18 val / 18 test groups; a drug is entirely in one split, never divided across
> splits); all 189 groups also independently cleared the 512-cell floor required to contribute
> at least one training sample (1,221 train / 124 val / 117 test samples). Each 512-cell
> training sample was constructed by selecting 10% of its cells (51 cells) for control
> replacement, chosen by iterating cell-line identity groups in randomized order; for each
> selected cell, its model input was replaced with a real control (vehicle) cell of the matching
> cell line drawn from anywhere in the dataset, while the reconstruction target remained the
> original treated cell's raw counts. Training data was regenerated in sample *order* each epoch
> but not in block *membership* — the same fixed 512-cell blocks per drug recur across all 10
> epochs. The resulting batch was passed to the fine-tuning head (`ICL_FinetunedModel`) with
> `n_kept_cell=460` (`= (1 - replacement_ratio) x sample_size`), which additionally applies a
> per-forward-pass scheduled context/query split over the remaining 52 cells. Fine-tuning used
> `sample_size=512`, `replacement_ratio=0.1`, batch size 8, for 10 epochs, initialized from the
> base `bc_large.ckpt` checkpoint. **[Pre-fix run, being superseded]** input data comprised
> 799,317 sci-Plex 3 cells nominally reported as 745,217 treated / 54,100 control across 4
> recorded cell-line identities; 36,522 of that "control" figure were later found to be cells
> with no recoverable cell-line or perturbation identity (failed nuclear-hash demultiplexing
> calls in the original sci-Plex 3 processing), incorrectly swept into the control bucket by an
> ingestion bug now fixed (Change 0) — the true composition is 745,217 treated / 17,578 control
> (5,857 A549 / 3,935 K562 / 7,786 MCF7) / 36,522 excluded as unassignable.

## Alpine execution: partition selection — resolved, `ah200`

Three partitions considered, in the order tried:

**`gh200`** (2 nodes, idle) — rejected at `sbatch` submission time (`Invalid qos specification`),
confirmed via CURC's own docs (curc.readthedocs.io/en/latest/clusters/alpine/alpine-hardware.html)
to be request-only: the partition lists `gh200` as an allowed QoS, but an individual account also
needs that QoS granted at the association level via a separate CURC support-request form. Not
pursued further — out of scope unless that request is separately filed and granted.

**`ah200`** (H200 GPUs, `x86_64`, `AllowQos=admin,gpu-normal,gpu-long` — the same `gpu-normal`
QoS `aa100` already uses, no special access needed) — the first attempt (job 31416858) hit its
20-minute time limit with no output past the script's own bash `echo` lines, an artifact of
Python's default stdout buffering to a file rather than evidence of a hang (see git history on
this doc for the full diagnosis). Fixed (`PYTHONUNBUFFERED=1`, `python -u`, flushed per-step
timestamps, 30-min limit) and re-run (job 31418001) **completed successfully in 21 minutes** —
genuinely slow, not hung. Confirmed: `machine: x86_64`, `torch 2.11.0+cu128`, `cuda available:
True`, allocated a full `NVIDIA H200 NVL` (not a MIG slice — the explicit `--gres=gpu:h200:1`
type pin worked as intended), `stack` imports cleanly with no architecture issue. Most of the
21 minutes was `import torch` (~2.5 min) and `from stack.model_loading import
load_model_from_checkpoint` (~11 min) — plausibly cold NFS/first-touch cost on an unfamiliar
node, not necessarily representative of steady-state cost on a node the env has already run on.
**`ah200` is confirmed usable — this is the partition for the real Change-1/2 runs.**

**`aa100`** (already proven throughout this project's history) — the comparison run was
cancelled while still queued; no data collected, and no longer needed now that `ah200` is
confirmed. Stays the documented fallback if a future `ah200` run genuinely fails, but is not
the current plan.

**Real-run time budgeting implication, worth carrying into `04_stack_generate.sbatch`'s own
`--time` estimate:** if the ~11-minute `load_model_from_checkpoint` import cost recurs per array
task (33 drugs, `%16` concurrency) rather than being amortized across tasks sharing a node, that
overhead should be accounted for in the real generation jobs' time budget, not assumed away —
worth confirming empirically on the first real run rather than assumed to vanish on repeat use.

## Acceptance

**Change 0 (sci-Plex ingestion bug fix — prerequisite):**
- `identity_missing_mask` (or equivalent name at planning time) added to
  `src/fmharness/sciplex_prep.py`, applied in `build_sciplex_finetune.py` before any
  `is_control` detection, on raw pre-stringified columns; `"nan"` removed from `VEHICLE_NAMES`.
  Tested against a small fixture proving a cell with real, missing (not merely
  string-`"nan"`-valued) perturbation/cell-line data is dropped, not mislabeled `is_control=True`.
- `08_sciplex_prep.sbatch` re-run with the fix; its own logged control count is exactly 17,578
  (down from the buggy 54,100), matching this design's independently-verified crosstab.
- `09_stack_finetune.sbatch` re-run against the corrected `sciplex_finetune.h5ad`, producing a
  new drug-aligned checkpoint. Group-eligibility numbers (currently 189 groups / 153-18-18 split)
  re-measured against the corrected data, not assumed unchanged from the pre-fix run.
- The drug-aligned Check-1/Check-2 rows already published in `docs/tahoe_generation_results.md`
  and the harness-overview deck (from the completed leakage-aware Check-2 plan) are re-derived
  against the corrected checkpoint and the numbers updated — not left standing as-is.

**Change 1 (faithful generation procedure):**
- `03_stack_context.sbatch` writes a real, per-line single-cell `tahoe_query.h5ad` (400 total
  rows, `cell_line_id` column present, `--partition` no longer `amilan`).
- `04_stack_generate.sbatch` runs `--mode mdm` with the four schedule flags explicit; a real run
  (both the cytokine-aligned and the Change-0-corrected drug-aligned checkpoint) completes
  without the `IndexError` the current `--mode vanilla` workaround exists to avoid, on `ah200`
  (confirmed working, see "Alpine execution" above).
- A new aggregation step (module TBD at planning time, likely `src/fmharness/`) reduces
  multi-cell-per-line generated output to one row per (line, drug), filtering by `gen_logit`
  before averaging, with a calibrated (not copied-from-the-paper) threshold and explicit
  missing-value handling — proven, not just asserted, to change Check-1's `stack` row relative
  to naive unfiltered averaging on a small fixture.

**Change 2 (faithful DE-based Check-1 metrics):**
- A new cached DE-calls bundle exists (`tahoe_deltas/`-pattern), and `delta_fidelity`/
  `score_delta_sources` report DE Spearman LFC, PR-AUC, DE Overlap Accuracy, and Jaccard
  alongside the existing Pearson-Delta, for every Check-1 row (all three checkpoint variants,
  the drug-aligned one now against the Change-0-corrected checkpoint).

**Other:**
- `src/fmharness/sciplex_prep.py` gains a gene-symbol-uniqueness check (the separate 110,983-
  feature anomaly, unrelated to Change 0's identity-missingness bug).
