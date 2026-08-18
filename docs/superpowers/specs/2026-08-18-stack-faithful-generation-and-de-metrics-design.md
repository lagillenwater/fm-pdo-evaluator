# Faithful Stack generation procedure + DE-based Check-1 metrics — design

**Date:** 2026-08-18
**Status:** approved (adversarially reviewed), pending implementation plan

## Purpose

Two changes to the Tahoe generation-eval harness, following up on the completed leakage-aware
Check-2 plan and a full read of the Stack paper (Dong et al. 2026, "Stack: In-Context Learning
of Single-Cell Biology," PMC12803207, bioRxiv posted 2026-06-08):

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

Both changes were adversarially reviewed against the real `ArcInstitute/stack` source (not paper
prose alone) before this design was finalized — see "Design history" below for what that review
overturned.

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

### `n_cells` verification (required before trusting the table above)

`self.n_cells=512` was confirmed in source for both checkpoints this project can point `CKPT=`
at (the cytokine-aligned `bc_large_aligned.ckpt`, built from `ArcInstitute/stack`'s own
`configs/finetuning/ft_parsecg.yaml` which pins `sample_size: 512`; and this project's own
sci-Plex-finetuned checkpoint, via `--sample_size 512` in `09_stack_finetune.sbatch`, which
`override_model_config_n_cells` bakes into the checkpoint's own `hyper_parameters` and
`generation.py` reads with no override available at generation time). But `n_cells` is a plain
constructor kwarg with **zero effect on any tensor shape** (`TabularAttentionLayer`'s
parameters, and `query_pos_embedding`, are both cell-axis-independent) — a wrong value would
never crash, it would just silently change every `n_test_cells`/`n_base_cells` calculation
above. Verify directly against whichever `.ckpt` is actually used, once on Alpine, before the
first real generation run under the new mode:

```bash
python -c "
import torch
ckpt = torch.load('<path>.ckpt', map_location='cpu')
print(ckpt['hyper_parameters']['model_config']['n_cells'])
"
```

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
not rely on it alone for anything requiring the complete log). All numbers below are measured,
not inferred.

### Data scale

`build_sciplex_finetune.py`'s own summary line: **799,317 cells x 110,983 features, 745,217
treated / 54,100 control, 189 perturbation groups, 4 cell identities.**

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
control-pool sizes were not obtained this session** (would need a fresh, small Alpine job
grouping `sciplex_finetune.h5ad`'s obs by `cell_line` restricted to `pert_id == "control"`) —
flagged as an open item for the smoke-test phase below, not blocking this design, since the
mechanism itself is fully confirmed regardless of the exact numbers.

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

### Draft methods paragraph (now fully source- and log-verified, no remaining "inferred" caveats on the mechanism itself)

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
> cell line drawn from anywhere in the dataset (with replacement only if fewer real control
> cells of that line were available than needed — exact real per-line control-pool sizes not
> yet measured), while the reconstruction target remained the original treated cell's raw
> counts. Training data was regenerated in sample *order* each epoch but not in block
> *membership* — the same fixed 512-cell blocks per drug recur across all 10 epochs. The
> resulting batch was passed to the fine-tuning head (`ICL_FinetunedModel`) with
> `n_kept_cell=460` (`= (1 - replacement_ratio) x sample_size`), which additionally applies a
> per-forward-pass scheduled context/query split over the remaining 52 cells. Fine-tuning used
> `sample_size=512`, `replacement_ratio=0.1`, batch size 8, for 10 epochs, initialized from the
> base `bc_large.ckpt` checkpoint. Input data comprised 799,317 sci-Plex 3 cells (745,217
> treated / 54,100 control) across 4 recorded cell-line identities.

**Remaining open item, not blocking:** real per-cell-line control-pool sizes (needed to state
precisely whether/how often the control-replacement step's with-replacement fallback path
triggers) — a small, cheap Alpine job, folded into the smoke-test phase below rather than run as
a separate step.

## Alpine execution: partition and a required compatibility check

Per direction: target the `gh200` partition (2 nodes, both idle at time of writing, `AllowQos=gh200`
— note the QOS name is `gh200`, not `gpu-normal` like `aa100` uses) rather than `aa100` (currently
under load: `mix`/`drain`, 1 idle node of 12). But `gh200`'s nodes are confirmed **`aarch64`
(ARM), not `x86_64`** (`scontrol show node c3gh-c13-u26`: `Arch=aarch64`) — a real architecture
difference from `aa100`, not just a faster/newer GPU generation. Conda environments are
architecture-specific; the existing `stack` conda env (activated via `conda activate stack` in
both `04_stack_generate.sbatch` and `09_stack_finetune.sbatch`) was almost certainly built for
`x86_64` and is not guaranteed to import cleanly, let alone run correctly, on `aarch64` without a
separate ARM-native build.

**This must be checked before committing any real generation run to `gh200`.** Smoke-test plan
(small, cheap, run before the implementation plan's real work):
1. A minimal sbatch job on `gh200` (`--partition=gh200 --qos=gh200`) that only activates the
   `stack` conda env and imports `torch`/`stack` — confirms whether the existing env is usable at
   all on this architecture, before spending any GPU time on real generation.
2. If it fails: fall back to `aa100` for the real runs (already confirmed working, per this
   project's entire prior history on this branch), and treat a `gh200`-native env build as
   separate, future work, not a blocker for Change 1/Change 2 landing.
3. If it succeeds: proceed with the real Change 1/2 runs on `gh200` directly.
4. Same job (or a second tiny one) also resolves the sci-Plex control-pool open item above via a
   quick `groupby` on `sciplex_finetune.h5ad`, and verifies `n_cells=512` against the actual
   `.ckpt` files per the "n_cells verification" snippet above — batching all three cheap checks
   into one Alpine round-trip rather than three.

## Acceptance

- `03_stack_context.sbatch` writes a real, per-line single-cell `tahoe_query.h5ad` (400 total
  rows, `cell_line_id` column present, `--partition` no longer `amilan`).
- `04_stack_generate.sbatch` runs `--mode mdm` with the four schedule flags explicit; a real run
  (both checkpoints) completes without the `IndexError` the current `--mode vanilla` workaround
  exists to avoid.
- A new aggregation step (module TBD at planning time, likely `src/fmharness/`) reduces
  multi-cell-per-line generated output to one row per (line, drug), filtering by `gen_logit`
  before averaging, with a calibrated (not copied-from-the-paper) threshold and explicit
  missing-value handling — proven, not just asserted, to change Check-1's `stack` row relative
  to naive unfiltered averaging on a small fixture.
- A new cached DE-calls bundle exists (`tahoe_deltas/`-pattern), and `delta_fidelity`/
  `score_delta_sources` report DE Spearman LFC, PR-AUC, DE Overlap Accuracy, and Jaccard
  alongside the existing Pearson-Delta, for every existing Check-1 row (all three checkpoint
  variants from the completed Check-2 plan).
- `src/fmharness/sciplex_prep.py` gains a gene-symbol-uniqueness check.
- The `gh200` vs `aarch64` compatibility question is resolved (either confirmed working, or a
  documented fallback to `aa100`) before any real Change-1/2 generation run is submitted.
- Real per-cell-line sci-Plex control-pool sizes are obtained and folded into the methods
  paragraph above, closing its one remaining open item.
