# Stack Faithful Generation + DE-Based Check-1 Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix a confirmed sci-Plex ingestion bug (Change 0), make Stack's Tahoe generation use
its real scheduled/confidence-guided in-context procedure instead of the `--mode vanilla`
workaround (Change 1), and add faithful Wilcoxon-DE-based metrics to Check 1 alongside the
existing Pearson-Delta (Change 2) — then re-derive and republish the drug-aligned Check-1/Check-2
numbers under all three fixes.

**Architecture:** Three independent local-code workstreams (Change 0: `sciplex_prep.py` +
`build_sciplex_finetune.py`; Change 1: two sbatch scripts + a new aggregation module; Change 2: a
new `deltas.py` builder + `evaluation.py` scoring functions), each TDD'd and committed on its
own, followed by real-execution tasks that fan out across Alpine (Change 0's fine-tune chain,
Change 1's generation runs) and this worktree (Change 2's DE-calls build, CPU-only, no GPU/queue
needed) with explicit dependency edges, converging on one final re-derivation task.

**Tech Stack:** Python 3.11, `uv run`, `anndata`/`scipy.sparse` for the h5ad plumbing, `scanpy`
(`rank_genes_groups(method="wilcoxon")`) for DE calling, `scikit-learn`
(`average_precision_score`) + `scipy.stats.spearmanr` for the new DE metrics, `pytest` for TDD,
Slurm/`sbatch` on CU Boulder's Alpine HPC (`ralpine` — this project's read-only Alpine wrapper —
for polling/pulling; the human submits jobs).

## Global Constraints

- Line length 100, target Python 3.11 (`pyproject.toml`).
- Ruff lint set `E, F, I, B, UP, SIM, RUF` must pass clean (`uv run ruff check src tests scripts`).
- Pyright strict over `src` and `tests` (`uv run pyright`) — `scripts/` is not in pyright's
  `include` list, so new scripts are not type-checked, matching every existing script in this repo.
- No emojis anywhere (code, docs, commit messages).
- Vectorized approaches — no nested Python loops over data rows. A single (non-nested) loop over
  a small, fixed-size set (~50 cell lines, ~1,650 (line, drug) pairs) is this project's own
  established exception (see `build_tahoe_deltas`'s per-line pattern, `build_tahoe_context.py`'s
  per-drug shard loop) — every gene-level computation inside such a loop must itself be vectorized
  (numpy/scipy/scanpy), never a further per-gene loop.
- Every new public function gets a docstring explaining *why*, not just *what* — matching this
  codebase's existing style (see any function in `src/fmharness/deltas.py` or `evaluation.py`).
- `ralpine` is read-only: Claude authors and commits sbatch scripts; the human (Lucas) submits
  jobs on Alpine and pastes back logs or lets Claude poll/pull via `ralpine`. No task in this plan
  has an agent directly submitting a Slurm job.
- Commit after each task (this session's established convention — confirmed for the prior Check-2
  plan, carried forward here unless told otherwise).
- Source-verified facts only — every threshold, flag, and mechanism below is grounded in either
  this project's own confirmed Alpine runs, the `ArcInstitute/stack` source, or the Stack preprint
  (Dong et al., 2026, *Stack: In-Context Learning of Single-Cell Biology*) — cited by exact
  section where it matters, not by memory.

---

## Parallel Workstreams

Five local-code tasks are **mutually independent** (disjoint files, no shared state, no
sequential dependency) and can be implemented and reviewed concurrently:

| Task | Change | Files touched | Depends on |
|---|---|---|---|
| 1 | Change 0 | `src/fmharness/sciplex_prep.py`, `scripts/build_sciplex_finetune.py`, `tests/test_sciplex_prep.py` | none |
| 2 | Change 1 | `scripts/alpine/03_stack_context.sbatch`, `scripts/alpine/04_stack_generate.sbatch` | none |
| 3 | Change 1 | `src/fmharness/stack_aggregate.py`, `tests/test_stack_aggregate.py` | none |
| 4 | Change 2 | `src/fmharness/deltas.py` (new `build_tahoe_de_calls`), `scripts/build_tahoe_de_calls.py`, `tests/test_deltas.py` | none |
| 5 | Change 2 | `src/fmharness/evaluation.py` (new `de_fidelity`/`score_de_metrics`), `tests/test_evaluation.py` | none (schema contract fixed by this plan, not by Task 4's code) |

Real-execution tasks then fan out with these dependency edges — see **Execution ordering for
ASAP results** at the end of this plan for the full picture:

```
Task 1 ──> Task 6 (Alpine: Change 0 chain, 08->09) ──> Task 9 (Alpine: drug-aligned 04) ──┐
Task 2, 3 ──> Task 7 (Alpine: cytokine-aligned 03->04) ─────────────────────────────────┤──> Task 10 (final)
Task 4 ──> Task 8 (LOCAL: DE-calls build, no Alpine, no queue) ──────────────────────────┘
Task 5 ─────────────────────────────────────────────────────────────────────────────────┘
```

Task 8 is the fastest path to a real result (no Alpine queue at all) and should be kicked off
immediately once Task 4 lands. Tasks 6 and 7 can be submitted to Alpine back-to-back by the human
as soon as their code tasks land — they are different jobs with no shared file dependency until
Task 10.

---

### Task 1: Change 0 — fix the sci-Plex identity-missing-cell misclassification bug

**Files:**
- Modify: `src/fmharness/sciplex_prep.py`
- Modify: `scripts/build_sciplex_finetune.py`
- Test: `tests/test_sciplex_prep.py`

**Interfaces:**
- Produces: `identity_missing_mask(pert: pd.Series, cell_line: pd.Series) -> np.ndarray` in
  `fmharness.sciplex_prep` — a boolean mask, `True` for cells whose raw (pre-`.astype(str)`)
  perturbation and/or cell-line identity is missing (`NaN`). No other task in this plan consumes
  this function; it is wired directly into `scripts/build_sciplex_finetune.py`'s `main()`.

**Context:** sci-Plex 3's own nuclear-hash demultiplexing blanks `perturbation`/`cell_line`/
`dose`/`well_oligo`/`plate_oligo` together when a hash call is too ambiguous to resolve — a real,
known ~4.6% artifact of the published release (confirmed via direct crosstab investigation this
session), not a scPerturb reprocessing defect. `scripts/build_sciplex_finetune.py`'s current
`VEHICLE_NAMES = {"control", "vehicle", "dmso", "none", "nan"}` fallback `is_control` detection
includes the literal string `"nan"`; combined with `.astype(str)` turning real `NaN` into that
string, this silently misclassifies those genuinely-unassignable cells as vehicle controls — a
confirmed, reproducible bug (36,522 misclassified cells; true control count is 17,578, not the
54,100 currently logged).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_sciplex_prep.py` (alongside the existing three test functions; update the
import line at the top from
`from fmharness.sciplex_prep import check_gene_count, check_perturbation_count, check_raw_counts`
to also import `identity_missing_mask`):

```python
def test_identity_missing_mask_true_when_perturbation_is_missing() -> None:
    pert = pd.Series(["DrugA", None, "DrugB"])
    cell_line = pd.Series(["A549", "A549", "K562"])
    mask = identity_missing_mask(pert, cell_line)
    assert list(mask) == [False, True, False]


def test_identity_missing_mask_true_when_cell_line_is_missing() -> None:
    pert = pd.Series(["DrugA", "DrugB", "DrugC"])
    cell_line = pd.Series(["A549", None, "K562"])
    mask = identity_missing_mask(pert, cell_line)
    assert list(mask) == [False, True, False]


def test_identity_missing_mask_true_when_both_missing() -> None:
    pert = pd.Series(["DrugA", None])
    cell_line = pd.Series(["A549", None])
    mask = identity_missing_mask(pert, cell_line)
    assert list(mask) == [False, True]


def test_identity_missing_mask_false_when_both_present_including_literal_control_string() -> None:
    # a real, resolved vehicle-control call ("control") must NOT be flagged as missing --
    # only actual NaN identity, never a resolved-but-control-like string value.
    pert = pd.Series(["DrugA", "control"])
    cell_line = pd.Series(["A549", "K562"])
    mask = identity_missing_mask(pert, cell_line)
    assert list(mask) == [False, False]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sciplex_prep.py -k identity_missing_mask -v`
Expected: FAIL with `ImportError` / `NameError` (`identity_missing_mask` not defined).

- [ ] **Step 3: Implement `identity_missing_mask`**

Add to `src/fmharness/sciplex_prep.py` (after the existing three check functions):

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

Check the file's existing imports already include `numpy as np` and `pandas as pd` (they do, per
the existing check functions' signatures) — no new imports needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sciplex_prep.py -k identity_missing_mask -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Wire the filter into `build_sciplex_finetune.py` and remove the `"nan"` fallback**

In `scripts/build_sciplex_finetune.py`, update the import line:

```python
from fmharness.sciplex_prep import (
    check_gene_count,
    check_perturbation_count,
    check_raw_counts,
    identity_missing_mask,
)
```

Remove the literal `"nan"` from `VEHICLE_NAMES`:

```python
VEHICLE_NAMES = {"control", "vehicle", "dmso", "none"}
```

In `main()`, immediately after the existing block

```python
    if pert_col is None or line_col is None:
        raise SystemExit(f"need perturbation + cell-line columns; obs has {list(a.obs.columns)}")
```

insert:

```python
    missing = identity_missing_mask(a.obs[pert_col], a.obs[line_col])
    if missing.any():
        print(
            f"dropping {int(missing.sum())}/{a.n_obs} cells with missing perturbation and/or "
            "cell-line identity (e.g. failed hash-demultiplexing calls) before any control "
            "detection"
        )
        a = a[~missing].copy()
```

This must run BEFORE the existing `pert = a.obs[pert_col].astype(str).to_numpy()` line (further
down in `main()`) — placing it right after the `pert_col`/`line_col` resolution, and before the
raw-counts-extraction block, guarantees that. This matches the existing `--min-cells-per-cond`
filter's own visible-drop convention (prints a count, doesn't silently subset).

- [ ] **Step 6: Run the full sciplex_prep test suite and lint/type checks**

Run: `uv run pytest tests/test_sciplex_prep.py -v`
Expected: all tests PASS (the 3 pre-existing + 4 new).

Run: `uv run ruff check src/fmharness/sciplex_prep.py scripts/build_sciplex_finetune.py tests/test_sciplex_prep.py`
Expected: clean.

Run: `uv run pyright src/fmharness/sciplex_prep.py tests/test_sciplex_prep.py`
Expected: 0 errors. (`scripts/` is outside pyright's `include`, so `build_sciplex_finetune.py`
is not separately checked — matches every other script in this repo.)

- [ ] **Step 7: Commit**

```bash
git add src/fmharness/sciplex_prep.py scripts/build_sciplex_finetune.py tests/test_sciplex_prep.py
git commit -m "fix: drop sci-Plex cells with missing perturbation/cell-line identity before control detection"
```

---

### Task 2: Change 1 — faithful generation sbatch scripts (03 + 04)

**Files:**
- Modify: `scripts/alpine/03_stack_context.sbatch`
- Modify: `scripts/alpine/04_stack_generate.sbatch`

**Interfaces:**
- Produces: `tahoe_query.h5ad` with **400 rows** (8 real single control cells per DepMap line,
  drawn without replacement), one obs column `cell_line_id` (string DepMap/Cellosaurus id per
  row) — this is what Task 3's `aggregate_generated_replicates` and the real-execution tasks
  consume as the groupby key on Stack's generated output.
- These two files are edited together in one task (not split) because they are tightly coupled:
  04's `--prompt-ratio`/`--context-ratio`/`--context-ratio-min` flags are only crash-safe against
  03's specific query-pool size (400), and a reviewer approving one without the other would
  reintroduce the crash this task exists to fix.

**Context:** Stack's CLI exposes two generation modes. `--mode vanilla` (current) runs T=5 fixed
steps at a constant `context_ratio` and never uses the confidence-guided selective-unmasking
classifier. Any other `--mode` string (the CLI's own default, `mdm`) schedules `context_ratio` via
`linspace(context_ratio_min, context_ratio, 5)` and carries `is_masked`/`test_logit` between
steps — this is the paper's actually-described method (Methods 4.2.5). The scheduled
`context_ratio` makes `n_test_cells = max(1, int(512*(1-prompt_ratio-context_ratio)))` vary
179-333 cells across the 5 steps (at `--prompt-ratio 0.25 --context-ratio 0.4
--context-ratio-min 0.2`). The CLI chunks its query (`--test-adata`) into batches of
`n_test_cells`; a short last batch is padded via a single pass (`pad = test_indices[:need]`) drawn
from the FULL original query pool — this succeeds only if the pool size is `>=` the largest
`n_test_cells` across all 5 steps (281, at `context_ratio=0.20`). The current 50-row pseudobulk
query (one mean profile per line) is far short of that, causing a confirmed `IndexError`; 03
currently dodges it entirely by using `--mode vanilla`, which never schedules `context_ratio` and
so never needs more than the query's own row count. Replacing the 50-row pseudobulk with 400 real
single cells (8/line, well within the 200 real controls/line always available in
`tahoe_context.h5ad`) clears the 281-cell bound with ~40% margin, letting the CLI's true default
mode run without touching `ArcInstitute/stack` itself.

- [ ] **Step 1: Edit `03_stack_context.sbatch` — fix the stale partition and replace the query-baseline block**

Change the partition (line 3). `amilan` is confirmed retired (no longer in Alpine's partition
list at all); this project's own `08_sciplex_prep.sbatch` already established the replacement
reasoning for a CPU job at this memory profile (`--mem=48G` / `--cpus-per-task=8` = 6G/core,
above `acpu`'s `MaxMemPerCPU=3840MB` cap, so `amem` is required, exactly as `08` documents for its
own, larger job):

```
#SBATCH --partition=amem
```

Replace the entire query-baseline block (currently reading, verbatim, from `# query baseline:
control pseudobulk per DepMap line...` through the `print(...)` call just before the closing
`PY` heredoc terminator) with:

```python
# query baseline: 8 REAL single control cells per DepMap line (400 total) -- NOT a pseudobulk
# mean. Stack's true generative procedure (--mode mdm, see 04) schedules its query-cell draw per
# step (n_test_cells up to 281 of the fixed self.n_cells=512 pool, at --prompt-ratio 0.25); a
# single pseudobulk row per line can never satisfy that (04's confirmed IndexError root cause:
# the CLI's single-pass pad, `pad = test_indices[:need]`, silently truncates once `need` exceeds
# the pool it draws from). 8/line x 50 lines = 400 clears the sufficient bound (pool >=
# max(n_test_cells) = 281, since the pad always draws from the FULL original pool, not a
# per-batch remainder) with ~40% headroom -- well inside the 200 real controls/line always
# available in tahoe_context.h5ad, so no replacement is needed or used. Fixed seed (0) for
# reproducible cell selection across re-runs.
cid = obs["cell_id"].astype(str).to_numpy()
cln = obs["cell_line_id"].astype(str).to_numpy()
patient = np.where((cid != "") & (cid != "nan"), cid, cln)
ctl_idx = np.flatnonzero(is_ctl)
rng = np.random.default_rng(0)
codes, uniq = pd.factorize(patient[ctl_idx])
sel = np.sort(
    np.concatenate(
        [
            rng.choice(
                ctl_idx[codes == code], size=min(8, int((codes == code).sum())), replace=False
            )
            for code in range(len(uniq))
        ]
    )
)
qx = adata.X[sel]
qx = np.asarray(qx.todense() if sparse.issparse(qx) else qx, dtype=np.float64)
row = qx.sum(axis=1, keepdims=True)
row[row == 0] = 1.0
query = ad.AnnData(X=(qx / row * 1e6).astype("float32"))
query.obs_names = [str(i) for i in range(query.n_obs)]
query.obs["cell_line_id"] = patient[sel]
query.var_names = [str(v) for v in adata.var_names]
query.var["feature_name"] = list(query.var_names)
query.write_h5ad(repo / "tahoe_query.h5ad")
print(
    f"{len(perts)} per-drug shards, {len(pm)} pert->CID rows, "
    f"query baseline {query.n_obs} real cells x {query.n_vars} genes "
    f"({len(uniq)} lines, {query.n_obs / len(uniq):.1f} cells/line avg) -> tahoe_query.h5ad"
)
```

This drops the now-unused `_group_mean` import from the heredoc's `from fmharness.deltas import
_group_mean` line — remove that import line too (nothing else in the heredoc still calls it).

- [ ] **Step 2: Syntax-check the edited sbatch script**

Run: `bash -n scripts/alpine/03_stack_context.sbatch`
Expected: no output (clean parse). The embedded Python heredoc is not checked by `bash -n`; verify
it separately:

Run: `python -c "import ast; ast.parse(open('/dev/stdin').read())" < <(sed -n '/^python - <<.PY/,/^PY$/p' scripts/alpine/03_stack_context.sbatch | sed '1d;$d')`
Expected: no output (valid Python syntax).

- [ ] **Step 3: Edit `04_stack_generate.sbatch` — drop vanilla mode, retarget to ah200**

Change the partition/GRES lines (lines 3, 9):

```
#SBATCH --partition=ah200
```
```
#SBATCH --gres=gpu:h200:1
```

`ah200` uses the same `gpu-normal` QoS as `aa100` (confirmed via `scontrol show partition ah200`
this session — no special access needed, unlike `gh200`, which requires a separate CURC support
request). The explicit `h200` GRES type is required because `ah200` nodes mix full and
MIG-sliced H200 types; confirmed working via a real 21-minute job (31418001) this session.

Add a comment above `#SBATCH --time=04:00:00` flagging the still-open cold-cache/schedule-cost
question rather than guessing a new number:

```
#SBATCH --time=04:00:00             # smoke test measured ~11-14min cold-cache import/checkpoint-
                                     # load cost per task on ah200 (job 31418001); --mode mdm's
                                     # fuller per-step schedule may also cost more than vanilla's
                                     # fixed ratio -- confirm on Task 7's first real run and raise
                                     # this if any array task approaches the limit.
```

Update the `QUERY=` line's inline comment:

```bash
QUERY="tahoe_query.h5ad"              # 400 real control cells, 8/line (written by 03)
```

Replace the large comment block (currently reading, verbatim, from `# Stack splits its 512-cell
set...` through the line just before the `stack-generation \` invocation) and the mode/ratio
flags inside that invocation with:

```bash
# Stack's true in-context generative procedure (the CLI's own default, --mode mdm, NOT --mode
# vanilla): T=5 steps, context_ratio SCHEDULED via linspace(context_ratio_min, context_ratio, 5)
# with confidence-guided selective unmasking carried between steps (Methods 4.2.5 of the Stack
# preprint) -- this is what the paper actually describes; --mode vanilla is a materially simpler
# fixed-ratio baseline the CLI happens to also expose, not "one-shot generation." n_test_cells =
# max(1, int(512*(1-prompt_ratio-context_ratio))) is recomputed per step under the SCHEDULED
# context_ratio, ranging 179-333 cells across these 5 steps; the 400-real-cell tahoe_query.h5ad
# (03) covers every step's n_test_cells with margin (see 03's own comment for why 400 suffices).
# This pipeline previously used --mode vanilla specifically to dodge an IndexError at the old
# 50-row pseudobulk query; that workaround is no longer needed now the query pool itself is large
# enough for the model's own default mode. Explicit rather than left to CLI defaults, matching
# this repo's "log resolved CKPT/GENELIST/OUTDIR" reproducibility discipline and guarding against
# a future package upgrade silently changing a default.
#
# Output is now MULTIPLE rows per line (one per surviving real query cell, each carrying
# obs["gen_logit"]) instead of one -- run
# fmharness.stack_aggregate.aggregate_generated_replicates (Task 3) on $OUTDIR before
# build_generated_deltas/delta_fidelity, which still expect exactly one row per (line, drug).

stack-generation \
    --checkpoint "$CKPT" \
    --base-adata "$SHARD" \
    --test-adata "$QUERY" \
    --genelist "$GENELIST" \
    --gene-name-col feature_name \
    --split-column pert_id \
    --split-values "$PERT" \
    --device cuda \
    --mode mdm \
    --prompt-ratio 0.25 \
    --context-ratio 0.4 \
    --context-ratio-min 0.2 \
    --output-dir "$OUTDIR"/
```

- [ ] **Step 4: Syntax-check the edited sbatch script**

Run: `bash -n scripts/alpine/04_stack_generate.sbatch`
Expected: no output (clean parse).

- [ ] **Step 5: Commit**

```bash
git add scripts/alpine/03_stack_context.sbatch scripts/alpine/04_stack_generate.sbatch
git commit -m "fix: use Stack's real scheduled in-context generation instead of --mode vanilla"
```

---

### Task 3: Change 1 — confidence-filtered aggregation of generated replicates

**Files:**
- Create: `src/fmharness/stack_aggregate.py`
- Test: `tests/test_stack_aggregate.py`

**Interfaces:**
- Consumes: raw per-drug `.h5ad` files as Stack's `stack-generation` CLI writes them (Task 2's
  `04_stack_generate.sbatch`) — each file has multiple rows per line (one per surviving real
  query cell from Task 2's 400-cell `tahoe_query.h5ad`), with obs columns `cell_line_id` (carried
  through from the query file) and `gen_logit` (Stack's own per-cell confidence score, already
  surfaced by the CLI once mode != vanilla).
- Produces: `aggregate_generated_replicates(generated_dir: Path, out_dir: Path, *, threshold:
  float) -> pd.DataFrame` in `fmharness.stack_aggregate` — writes one reduced `<pert_id>.h5ad`
  file per input file to `out_dir` (exactly one row per line, the shape
  `fmharness.deltas.build_generated_deltas` already requires unchanged), and returns a summary
  DataFrame (`pert_id, cell_line_id, n_replicates, n_kept, dropped`) for auditing filter impact.
  No other task in this plan imports this function directly — it is invoked from the real-run
  Task 7/9 as a script step, run on `04`'s output directory before `check1_registry_driver.py`.

**Context:** Naive averaging of all replicates per line is a bias problem, not a variance
problem: a low-confidence (still-masked) replicate is mechanistically pulled toward the query
baseline under weak context support, so averaging in more of them just estimates that bias more
precisely, not less. Filter-then-average — keep only replicates Stack's own classifier judges
confidently-resolved (`gen_logit < threshold`) before averaging — is the design's required fix.
A (line, drug) with zero surviving replicates must be dropped, not silently backfilled with the
unfiltered mean (which would reintroduce exactly the bias being filtered out). `threshold` is a
parameter, not hardcoded: it must be calibrated per-checkpoint against Check-1 Pearson-Delta (see
Task 7's calibration step), not copied from the paper's own value (`2.5`, calibrated on a
different checkpoint/task).

- [ ] **Step 1: Write the failing test**

Create `tests/test_stack_aggregate.py`:

```python
"""Tests for confidence-filtered aggregation of Stack's per-query-cell generated output."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from fmharness.stack_aggregate import aggregate_generated_replicates


@pytest.fixture
def generated_dir(tmp_path: Path) -> Path:
    genes = ["A", "B"]
    gen = ad.AnnData(
        X=np.array(
            [
                [10.0, 0.0],  # L1 rep1: confidently resolved (kept)
                [12.0, 0.0],  # L1 rep2: confidently resolved (kept)
                [100.0, 0.0],  # L1 rep3: still masked/low-confidence (dropped)
                [5.0, 5.0],  # L2 rep1: low-confidence (dropped)
                [6.0, 6.0],  # L2 rep2: low-confidence (dropped)
            ],
            dtype=np.float32,
        )
    )
    gen.var_names = genes
    gen.obs["cell_line_id"] = ["L1", "L1", "L1", "L2", "L2"]
    gen.obs["gen_logit"] = [-1.0, -0.5, 3.0, 2.0, 4.0]
    d = tmp_path / "generated"
    d.mkdir()
    gen.write_h5ad(d / "drugX.h5ad")
    return d


def test_aggregate_generated_replicates_filters_by_confidence_before_averaging(
    tmp_path: Path, generated_dir: Path
) -> None:
    out_dir = tmp_path / "aggregated"

    summary = aggregate_generated_replicates(generated_dir, out_dir, threshold=0.0)

    reduced = ad.read_h5ad(out_dir / "drugX.h5ad")
    assert list(reduced.obs_names) == ["L1"]  # L2 dropped: zero replicates survive the filter
    assert np.allclose(reduced.X, [[11.0, 0.0]])  # mean of the two KEPT reps, not all three
    naive_mean_gene_a = (10.0 + 12.0 + 100.0) / 3
    assert not np.isclose(float(reduced.X[0, 0]), naive_mean_gene_a)

    l2 = summary[(summary["pert_id"] == "drugX") & (summary["cell_line_id"] == "L2")].iloc[0]
    assert l2["n_replicates"] == 2
    assert l2["n_kept"] == 0
    assert bool(l2["dropped"])
    l1 = summary[(summary["pert_id"] == "drugX") & (summary["cell_line_id"] == "L1")].iloc[0]
    assert l1["n_replicates"] == 3
    assert l1["n_kept"] == 2
    assert not bool(l1["dropped"])


def test_aggregate_generated_replicates_raises_without_required_obs_columns(
    tmp_path: Path,
) -> None:
    gen = ad.AnnData(X=np.zeros((2, 1), dtype=np.float32))
    gen.var_names = ["A"]
    d = tmp_path / "generated"
    d.mkdir()
    gen.write_h5ad(d / "drugY.h5ad")

    with pytest.raises(ValueError, match="gen_logit"):
        aggregate_generated_replicates(d, tmp_path / "out", threshold=2.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_stack_aggregate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmharness.stack_aggregate'`.

- [ ] **Step 3: Implement `aggregate_generated_replicates`**

Create `src/fmharness/stack_aggregate.py`:

```python
"""Reduce Stack's per-query-cell generated output to one row per (line, drug).

Stack's true in-context generation (04_stack_generate.sbatch, Change 1) writes one predicted
row PER QUERY CELL, cell-indexed -- with a 400-real-cell query (8/line), that means multiple
rows per line per drug, each carrying obs["gen_logit"] (Stack's own confidence classifier: high
= still-masked/unresolved, low = confidently-resolved). fmharness.deltas.build_generated_deltas
expects exactly one row per (line, drug); aggregate_generated_replicates is the step between
Stack's raw output and that function.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from fmharness.deltas import dense


def aggregate_generated_replicates(
    generated_dir: Path,
    out_dir: Path,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Filter-then-average Stack's per-query-cell generated replicates down to one row per line.

    Only replicates with ``gen_logit < threshold`` (Stack's own confidence classifier judging the
    cell confidently-resolved, not still-masked) are averaged per line. A naive unfiltered mean
    is a bias problem, not a variance one: a low-confidence replicate is mechanistically pulled
    toward the query baseline under weak context support, so averaging in more of them just
    estimates that bias more precisely. A (drug, line) with zero surviving replicates is dropped,
    not silently backfilled with the unfiltered mean, which would reintroduce exactly the bias
    being filtered out.

    Writes one reduced ``<pert_id>.h5ad`` (one row per line) per input file to ``out_dir`` --
    the shape ``build_generated_deltas`` already requires, so it needs no changes. Returns a
    summary DataFrame (``pert_id, cell_line_id, n_replicates, n_kept, dropped``) for auditing how
    much each drug's generation was affected by the filter.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[pd.DataFrame] = []
    for f in sorted(Path(generated_dir).glob("*.h5ad")):
        gen = ad.read_h5ad(f)
        if "gen_logit" not in gen.obs or "cell_line_id" not in gen.obs:
            raise ValueError(
                f"{f}: missing obs['gen_logit']/obs['cell_line_id'] -- was this generated "
                "with --mode mdm (not vanilla) against a query baseline carrying cell_line_id?"
            )
        logit = gen.obs["gen_logit"].to_numpy(dtype=float)
        line = gen.obs["cell_line_id"].astype(str).to_numpy()
        x = dense(gen.X)
        keep = logit < threshold

        # per-line mean of the KEPT replicates only, via an indicator-matmul (no explicit
        # per-line loop -- mirrors fmharness.deltas._group_mean's own indicator-matmul pattern,
        # restricted here to the kept subset).
        codes, uniq = pd.factorize(line)
        n_lines = len(uniq)
        n_total = np.bincount(codes, minlength=n_lines)
        n_kept = np.bincount(codes[keep], minlength=n_lines)
        ind = np.zeros((n_lines, len(codes)), dtype=np.float64)
        ind[codes[keep], np.flatnonzero(keep)] = 1.0
        denom = np.where(n_kept == 0, 1.0, n_kept.astype(np.float64))
        means = (ind @ x) / denom[:, None]

        have = n_kept > 0
        if have.any():
            reduced = ad.AnnData(X=means[have].astype(np.float32))
            reduced.obs_names = [str(u) for u in uniq[have]]
            reduced.var_names = [str(v) for v in gen.var_names]
            reduced.var["feature_name"] = list(reduced.var_names)
            reduced.write_h5ad(out_dir / f.name)

        summaries.append(
            pd.DataFrame(
                {
                    "pert_id": f.stem,
                    "cell_line_id": [str(u) for u in uniq],
                    "n_replicates": n_total,
                    "n_kept": n_kept,
                    "dropped": n_kept == 0,
                }
            )
        )
    if not summaries:
        return pd.DataFrame(
            columns=["pert_id", "cell_line_id", "n_replicates", "n_kept", "dropped"]
        )
    return pd.concat(summaries, ignore_index=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stack_aggregate.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src/fmharness/stack_aggregate.py tests/test_stack_aggregate.py`
Expected: clean.

Run: `uv run pyright src/fmharness/stack_aggregate.py tests/test_stack_aggregate.py`
Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add src/fmharness/stack_aggregate.py tests/test_stack_aggregate.py
git commit -m "feat: add confidence-filtered aggregation for Stack's per-query-cell generation output"
```

---

### Task 4: Change 2 — ground-truth DE-calls bundle builder

**Files:**
- Modify: `src/fmharness/deltas.py` (add `build_tahoe_de_calls`)
- Create: `scripts/build_tahoe_de_calls.py`
- Test: `tests/test_deltas.py`

**Interfaces:**
- Consumes: `tahoe_context.h5ad` (raw per-cell, already in this worktree) — same obs schema
  `build_tahoe_deltas` already reads (`cell_id`, `cell_line_id`, `pubchem_cid`, `is_control`).
- Produces: `build_tahoe_de_calls(adata: ad.AnnData, *, lfc_threshold: float = 0.25,
  fdr_threshold: float = 0.05) -> pd.DataFrame` in `fmharness.deltas` — one row per (line, drug,
  gene) with columns `patient, drug, gene, log2fc, padj, significant`. This exact column set is
  the schema Task 5's `de_fidelity` consumes; both tasks must agree on these names (they are
  fixed by this plan, not decided independently by either task's implementer).

**Context:** The paper's DE calling (Methods 4.6.3: "Wilcoxon rank-sum tests for DE detection and
Benjamini-Hochberg correction... cell-eval v0.6.6 with default parameters") needs per-cell data to
produce a p-value; `build_tahoe_deltas` collapses straight to pseudobulk means with no per-cell or
significance information retained, and the cached `tahoe_deltas/` bundle (this project's
preferred `--deltas-bundle` shortcut) is exactly that already-collapsed output — there is no path
back to per-cell data from it. This builder is the new, separate one-time-compute step that
produces a cacheable ground-truth DE-calls bundle, matching the existing `tahoe_deltas/` pattern.

Threshold values: Methods 4.6.3 itself states no specific LFC/FDR numbers for the paper's main
cell-prompting Task 1-4 benchmarks (just "cell-eval v0.6.6... default parameters"). The only place
in the entire paper stating a concrete LFC/FDR pair for a cell-eval-based DE call is Methods 4.8
(Evaluations of Perturb Sapiens, a *different* whole-organism atlas analysis using the same
cell-eval tooling): "We applied an LFC threshold of 0.25 and a FDR threshold of 0.05 for cell-eval
evaluations." Since no better-grounded number exists anywhere in the source text for this project's
own cell-eval-style Tahoe DE calling, `0.25`/`0.05` are adopted as the defaults here — cited to
Methods 4.8, not 4.6.3, to avoid mis-attributing a number the paper never actually states in that
section. Both remain function parameters (not hardcoded), trivially overridable if a better source
ever surfaces.

Uses `scanpy.tl.rank_genes_groups(method="wilcoxon")` (already implements exactly this test + BH
correction, confirmed via `scipy.stats.false_discovery_control`/scanpy availability check this
session) rather than a hand-rolled scipy loop — `scanpy>=1.10` is already a `pyproject.toml`
dependency, not previously imported anywhere in `src/`/`scripts/`. Loops once per cell LINE
(~50, comparing every drug applied in that line against its own control group in a single
`rank_genes_groups` call, since scanpy natively supports multiple groups vs. one reference) —
this is both more efficient than looping per (line, drug) pair (~1,650) and matches this
project's "loop over a small fixed set" convention at its smallest natural granularity.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_deltas.py` (near `test_build_tahoe_deltas_pseudobulks_and_logfc`; add
`build_tahoe_de_calls` to the existing `from fmharness.deltas import (...)` block at the top of
the file):

```python
def test_build_tahoe_de_calls_significant_gene_flagged_by_wilcoxon_and_lfc() -> None:
    # one line, one drug; gene A clearly separates control vs treated (Wilcoxon-significant,
    # large log2fc); gene B is flat (not significant). n=6/group gives the rank-sum test enough
    # resolution to reach padj < 0.05 on a complete separation (the minimum possible two-sided
    # exact p-value at n1=n2=6 is ~0.0043, well under 0.05; at n=2/group it could never go below
    # 1/3, so this fixture needs >=6 cells per group, not the 2-per-group used elsewhere in this
    # file for pseudobulk-only tests).
    genes = ["A", "B"]
    ctl = np.array([[1.0, 5.0]] * 6, dtype=np.float32)
    trt = np.array([[50.0, 5.0]] * 6, dtype=np.float32)
    x = np.vstack([ctl, trt])
    obs = pd.DataFrame(
        {
            "cell_id": ["ACH-1"] * 12,
            "cell_line_id": ["CVCL_1"] * 12,
            "pubchem_cid": ["0"] * 6 + ["100"] * 6,
            "is_control": [True] * 6 + [False] * 6,
        }
    )
    adata = ad.AnnData(X=x, obs=obs)
    adata.var_names = genes

    calls = build_tahoe_de_calls(adata)

    assert set(calls.columns) == {"patient", "drug", "gene", "log2fc", "padj", "significant"}
    assert set(map(tuple, calls[["patient", "drug"]].drop_duplicates().to_numpy())) == {
        ("ACH-1", "100")
    }
    a = calls[calls["gene"] == "A"].iloc[0]
    b = calls[calls["gene"] == "B"].iloc[0]
    assert bool(a["significant"])
    assert a["log2fc"] > 0  # treated > control
    assert a["padj"] < 0.05
    assert not bool(b["significant"])


def test_build_tahoe_de_calls_uses_paper_grounded_default_thresholds() -> None:
    # locks in the exact threshold decision (Methods 4.8's cell-eval LFC/FDR pair -- the only
    # concrete number the paper states anywhere for cell-eval-based DE calling) as an explicit,
    # checkable contract rather than an accidental default.
    import inspect

    sig = inspect.signature(build_tahoe_de_calls)
    assert sig.parameters["lfc_threshold"].default == 0.25
    assert sig.parameters["fdr_threshold"].default == 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_deltas.py -k build_tahoe_de_calls -v`
Expected: FAIL with `ImportError` (`build_tahoe_de_calls` not defined).

- [ ] **Step 3: Implement `build_tahoe_de_calls`**

Add to `src/fmharness/deltas.py`, after `build_tahoe_deltas` (which ends around line 680-690 —
find the end of that function's `return` statement and insert after it). Add `import scanpy as
sc` to the file's existing import block (alongside `import anndata as ad`, `import numpy as np`):

```python
def build_tahoe_de_calls(
    adata: ad.AnnData,
    *,
    lfc_threshold: float = 0.25,
    fdr_threshold: float = 0.05,
) -> pd.DataFrame:
    """Ground-truth per-(line, drug, gene) DE calls: two-sided Wilcoxon rank-sum test (treated
    vs. that line's own DMSO control cells), Benjamini-Hochberg FDR correction, significant =
    ``(padj < fdr_threshold) & (|log2fc| > lfc_threshold)`` -- the paper's own cell-eval-based DE
    procedure (Methods 4.6.3: "Wilcoxon rank-sum tests for DE detection and Benjamini-Hochberg
    correction... cell-eval v0.6.6 with default parameters"; the default thresholds here come
    from Methods 4.8's "LFC threshold of 0.25 and a FDR threshold of 0.05 for cell-eval
    evaluations" -- the only concrete LFC/FDR pair stated anywhere in the paper for a cell-eval
    DE call). Uses ``scanpy.tl.rank_genes_groups(method="wilcoxon")``, which already implements
    this exact test + BH correction, rather than a hand-rolled scipy loop.

    Loops once per cell line (not per (line, drug) pair): for each line, every drug applied in
    that line is compared against the line's own control cells in a single ``rank_genes_groups``
    call (scanpy natively supports multiple groups vs. one reference), so the gene-level
    computation for every drug in that line is vectorized together.

    Returns one row per (line, drug, gene): ``patient, drug, gene, log2fc, padj, significant`` --
    the ground-truth side of Check 1's DE metrics (``fmharness.evaluation.de_fidelity``); the
    predicted side needs no test (ranked by ``|log2fc|`` alone -- see that function's docstring).
    """
    obs = adata.obs
    genes = pd.Index([str(v) for v in adata.var_names])
    cid = obs["cell_id"].astype(str).to_numpy()
    cln = obs["cell_line_id"].astype(str).to_numpy()
    patient = np.where((cid != "") & (cid != "nan"), cid, cln)
    drug = obs["pubchem_cid"].astype(str).to_numpy()
    is_ctl = obs["is_control"].to_numpy(dtype=bool)

    x = adata.X
    xc = cast(
        "sparse.csr_matrix",
        x if sparse.issparse(x) else sparse.csr_matrix(np.asarray(x, dtype=np.float64)),
    )
    lib = np.asarray(xc.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0
    log1p_cpm = xc.multiply(1e4 / lib[:, None]).tocsr()
    log1p_cpm.data = np.log1p(log1p_cpm.data)

    rows: list[pd.DataFrame] = []
    for line in sorted(set(patient[is_ctl])):
        line_mask = patient == line
        ctl_mask = line_mask & is_ctl
        trt_mask = line_mask & ~is_ctl
        if not ctl_mask.any() or not trt_mask.any():
            continue
        idx = np.flatnonzero(ctl_mask | trt_mask)
        group = np.where(is_ctl[idx], "control", drug[idx])
        drugs_here = [d for d in pd.unique(group) if d != "control"]
        if not drugs_here:
            continue
        sub = ad.AnnData(
            X=log1p_cpm[idx],
            obs=pd.DataFrame({"de_group": group}, index=[str(i) for i in idx]),
            var=pd.DataFrame(index=genes),
        )
        sc.tl.rank_genes_groups(
            sub, groupby="de_group", groups=drugs_here, reference="control", method="wilcoxon"
        )
        # group=None returns every tested group's rows concatenated in one DataFrame (with its
        # own "group" column) -- no per-drug loop needed alongside the per-line loop above.
        res = sc.get.rank_genes_groups_df(sub, group=None)
        rows.append(
            pd.DataFrame(
                {
                    "patient": line,
                    "drug": res["group"].to_numpy(),
                    "gene": res["names"].to_numpy(),
                    "log2fc": res["logfoldchanges"].to_numpy(dtype=float),
                    "padj": res["pvals_adj"].to_numpy(dtype=float),
                }
            )
        )
    if not rows:
        raise ValueError("no (line, drug) pair had both control and treated cells")
    out = pd.concat(rows, ignore_index=True)
    out["significant"] = (out["padj"] < fdr_threshold) & (out["log2fc"].abs() > lfc_threshold)
    return out
```

Note: `cast` and `sparse` are already imported in `deltas.py` (used by `build_tahoe_deltas`
itself) — no new imports needed beyond `scanpy as sc`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_deltas.py -k build_tahoe_de_calls -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Create the thin CLI wrapper**

Create `scripts/build_tahoe_de_calls.py`:

```python
"""Build the ground-truth DE-calls bundle Check 1's DE-based metrics score against.

Wilcoxon rank-sum + Benjamini-Hochberg FDR per (line, drug, gene), from the raw per-cell
tahoe_context.h5ad -- see fmharness.deltas.build_tahoe_de_calls for the method and
docs/superpowers/specs/2026-08-18-stack-faithful-generation-and-de-metrics-design.md's Change 2
section for why this needs the raw per-cell context (not the tahoe_deltas/ pseudobulk bundle,
which retains no per-cell/significance information). Real, one-time compute (~1,650 (line, drug)
pairs); cache the output, matching the existing tahoe_deltas/ bundle pattern, rather than
repeating it on every Check-1 run.

Run (CPU-only, no GPU needed -- Wilcoxon rank-sum + BH correction is not a model call; runs
directly in this worktree since tahoe_context.h5ad is already local, no Alpine submission
required):
    uv run python scripts/build_tahoe_de_calls.py --context tahoe_context.h5ad \\
        --out tahoe_de_calls/de_calls.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad

from fmharness.deltas import build_tahoe_de_calls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context", default="tahoe_context.h5ad", help="Tahoe context AnnData")
    ap.add_argument("--out", default="tahoe_de_calls/de_calls.parquet")
    ap.add_argument("--lfc-threshold", type=float, default=0.25)
    ap.add_argument("--fdr-threshold", type=float, default=0.05)
    args = ap.parse_args()

    calls = build_tahoe_de_calls(
        ad.read_h5ad(args.context),
        lfc_threshold=args.lfc_threshold,
        fdr_threshold=args.fdr_threshold,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    calls.to_parquet(out)
    n_pairs = len(calls[["patient", "drug"]].drop_duplicates())
    n_sig = int(calls["significant"].sum())
    print(
        f"{len(calls)} (line, drug, gene) rows, {n_pairs} pairs, "
        f"{n_sig} significant calls -> {out}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Lint and type-check**

Run: `uv run ruff check src/fmharness/deltas.py scripts/build_tahoe_de_calls.py tests/test_deltas.py`
Expected: clean.

Run: `uv run pyright src/fmharness/deltas.py tests/test_deltas.py`
Expected: 0 errors.

- [ ] **Step 7: Commit**

```bash
git add src/fmharness/deltas.py scripts/build_tahoe_de_calls.py tests/test_deltas.py
git commit -m "feat: add build_tahoe_de_calls, ground-truth Wilcoxon DE-calls bundle builder"
```

---

### Task 5: Change 2 — DE-based Check-1 scoring (`de_fidelity` / `score_de_metrics`)

**Files:**
- Modify: `src/fmharness/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: a `de_calls` DataFrame in the exact schema Task 4 produces — columns `patient, drug,
  gene, log2fc, padj, significant` — and `(pred_delta, pred_key)` tuples in the same shape every
  existing delta source in this codebase already produces (`build_generated_deltas`,
  `loo_baseline_source`, etc.: `pred_delta` = pairs x genes, `pred_key` = `patient, drug` aligned
  row-for-row). This task does not depend on Task 4's code landing first — the schema is fixed by
  this plan, and this task's tests construct `de_calls` fixtures directly.
- Produces: `de_fidelity(pred_delta, pred_key, de_calls) -> pd.DataFrame` and `score_de_metrics
  (sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]], de_calls: pd.DataFrame) -> pd.DataFrame`
  in `fmharness.evaluation` — the DE-metrics analogues of the existing `delta_fidelity`/
  `score_delta_sources` (kept as a new, separate function pair rather than an extension of
  `delta_fidelity` itself, since DE metrics need discrete per-gene significance calls, not just
  continuous delta profiles — matching this project's existing "one function per metric family,
  composed by a `score_*` driver" pattern). Consumed by Task 10's final re-derivation.

**Context (metric definitions, Stack preprint Methods 4.6.3, verified against the paper's own
PDF this session, not re-derived from memory):**
- **DE Spearman LFC**: "the Spearman rank correlation between predicted and observed
  log2-fold-changes, calculated within the set of significantly differentially expressed (DE)
  genes in the ground truth" — restricted to real-significant genes only.
- **PR-AUC**: "Area under precision-recall curve using binary DE labels and −log10(p-values) as
  scores, using sklearn average precision score implementation." Our predicted side has no
  per-cell distribution to run a formal significance test against (Stack's generation is a single
  point delta per line, not multiple cells to test) — per the approved design spec, ranking by
  `|predicted delta|` in place of `−log10(predicted p-value)` is the sanctioned adaptation, since
  only the ground truth needs a formal significance call.
- **DE Overlap Accuracy**: "the overlap between the top-N genes from the true DE
  abs-log-fold-change ranking and the top-N genes from the predicted DE ranking... |top-N true ∩
  top-N predicted| / N, where N is the total number of true DE genes" — predicted ranking is by
  `|predicted delta|` (same adaptation as PR-AUC, for the same reason).
  - **Jaccard similarity**: "the Jaccard index between the set of predicted DE genes and ground
  truth DE genes... size of the intersection divided by the size of the union." Ground truth set =
  the real `significant` flag set; predicted set = the same top-N-by-`|predicted delta|`
  truncation as DE Overlap Accuracy (N = count of true-significant genes) — the natural,
  already-implied extension of the same adaptation to a discrete set metric.

A pair with zero real-significant genes has no well-defined rank-based comparison
(`de_spearman_lfc`/`de_overlap_accuracy`/`jaccard` = NaN); if additionally every gene in that pair
is either all-significant or all-non-significant, `pr_auc` (needs both classes present) is also
NaN — both cases handled explicitly, never silently defaulted to 0 or 1.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_evaluation.py` (near the existing `delta_fidelity`/`score_delta_sources`
tests; add `de_fidelity` and `score_de_metrics` to the existing `from fmharness.evaluation import
(...)` block at the top of the file):

```python
def test_de_fidelity_scores_spearman_lfc_overlap_pr_auc_jaccard() -> None:
    # one (patient, drug) pair, 4 genes; real DE calls: A, B significant (padj<0.05,
    # |log2fc|>0.25), C, D not. Predicted delta ranks A, B highest by |value| -- matches the real
    # top-N=2 exactly -> perfect overlap/Jaccard/PR-AUC, and correlates with the real log2FC on
    # the significant genes -> spearman +1.
    de_calls = pd.DataFrame(
        {
            "patient": ["ACH-1"] * 4,
            "drug": ["100"] * 4,
            "gene": ["A", "B", "C", "D"],
            "log2fc": [3.0, 2.0, 0.1, -0.05],
            "padj": [0.001, 0.01, 0.9, 0.8],
            "significant": [True, True, False, False],
        }
    )
    pred_delta = pd.DataFrame({"A": [3.5], "B": [1.5], "C": [0.2], "D": [-0.1]})
    pred_key = pd.DataFrame({"patient": ["ACH-1"], "drug": ["100"]})

    f = de_fidelity(pred_delta, pred_key, de_calls)

    assert len(f) == 1
    row = f.iloc[0]
    assert row["n_sig_genes"] == 2
    assert row["de_overlap_accuracy"] == 1.0
    assert row["jaccard"] == 1.0
    assert row["de_spearman_lfc"] == 1.0
    assert 0.0 <= row["pr_auc"] <= 1.0


def test_de_fidelity_zero_significant_genes_gives_nan_rank_metrics_but_no_crash() -> None:
    de_calls = pd.DataFrame(
        {
            "patient": ["ACH-1"] * 2,
            "drug": ["100"] * 2,
            "gene": ["A", "B"],
            "log2fc": [0.1, -0.05],
            "padj": [0.9, 0.8],
            "significant": [False, False],
        }
    )
    pred_delta = pd.DataFrame({"A": [0.2], "B": [-0.1]})
    pred_key = pd.DataFrame({"patient": ["ACH-1"], "drug": ["100"]})

    f = de_fidelity(pred_delta, pred_key, de_calls)

    assert f.iloc[0]["n_sig_genes"] == 0
    assert np.isnan(f.iloc[0]["de_spearman_lfc"])
    assert np.isnan(f.iloc[0]["de_overlap_accuracy"])
    assert np.isnan(f.iloc[0]["jaccard"])
    assert np.isnan(f.iloc[0]["pr_auc"])  # only one class present (all non-significant)


def test_score_de_metrics_builds_one_row_per_source() -> None:
    de_calls = pd.DataFrame(
        {
            "patient": ["ACH-1", "ACH-1"],
            "drug": ["100", "100"],
            "gene": ["A", "B"],
            "log2fc": [3.0, 0.1],
            "padj": [0.001, 0.9],
            "significant": [True, False],
        }
    )
    pred_key = pd.DataFrame({"patient": ["ACH-1"], "drug": ["100"]})
    sources = {
        "good": (pd.DataFrame({"A": [3.0], "B": [0.1]}), pred_key),
        "bad": (pd.DataFrame({"A": [-3.0], "B": [0.1]}), pred_key),
    }

    table = score_de_metrics(sources, de_calls)

    assert set(table["source"]) == {"good", "bad"}
    assert list(table.columns) == [
        "source",
        "de_spearman_lfc",
        "pr_auc",
        "de_overlap_accuracy",
        "jaccard",
        "n_pairs",
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluation.py -k "de_fidelity or score_de_metrics" -v`
Expected: FAIL with `ImportError` (`de_fidelity`/`score_de_metrics` not defined).

- [ ] **Step 3: Implement `de_fidelity` and `score_de_metrics`**

Add `from sklearn.metrics import average_precision_score` to `evaluation.py`'s import block
(`spearmanr` is already imported there). Add both functions after `score_delta_sources`:

```python
def de_fidelity(
    pred_delta: pd.DataFrame,
    pred_key: pd.DataFrame,
    de_calls: pd.DataFrame,
) -> pd.DataFrame:
    """DE-based faithfulness of a predicted expression delta, per (patient, drug), against
    ground-truth Wilcoxon DE calls (``fmharness.deltas.build_tahoe_de_calls``).

    For every (patient, drug) present in both ``pred_key`` and ``de_calls``, computes four
    metrics matching the Stack paper's cell-eval-based DE evaluation (Methods 4.6.3): DE Spearman
    LFC (Spearman rank correlation between predicted delta and real log2FC, restricted to the
    real-significant genes), PR-AUC (average precision of ``|predicted delta|`` as a score against
    the real ``significant`` binary label, over all tested genes), and DE Overlap Accuracy /
    Jaccard similarity (both from the top-N genes by ``|predicted delta|``, N = the number of
    real-significant genes for that pair, against the real-significant gene set -- the paper's own
    top-N-overlap definition). Our predicted side has no per-cell distribution to run a formal
    significance test against (a single generated delta per line, not multiple cells to test), so
    it is ranked by ``|predicted delta|`` alone in place of a predicted p-value -- the design's
    sanctioned adaptation, since only the ground truth needs a formal significance call.

    Returns one row per matched pair: ``patient, drug, de_spearman_lfc, pr_auc,
    de_overlap_accuracy, jaccard, n_sig_genes``. A pair with zero real-significant genes has
    ``de_spearman_lfc``/``de_overlap_accuracy``/``jaccard`` as NaN (undefined without at least one
    true positive); if every gene in that pair is one single class (all- or none-significant),
    ``pr_auc`` (needs both classes present) is NaN too -- both cases explicit, never silently
    defaulted to 0 or 1.
    """
    pk = pred_key.reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for (patient, drug), grp in de_calls.groupby(["patient", "drug"]):
        match = pk[(pk["patient"] == patient) & (pk["drug"] == drug)]
        if match.empty:
            continue
        i = int(match.index[0])
        genes = grp["gene"].to_numpy()
        pred_row = pred_delta.reindex(columns=genes).iloc[i].to_numpy(dtype=np.float64)
        have = ~np.isnan(pred_row)
        if not have.any():
            continue
        genes, pred_row = genes[have], pred_row[have]
        by_gene = grp.set_index("gene").loc[genes]
        real_lfc = by_gene["log2fc"].to_numpy(dtype=np.float64)
        sig = by_gene["significant"].to_numpy(dtype=bool)
        n_sig = int(sig.sum())

        pr_auc = (
            float(average_precision_score(sig, np.abs(pred_row)))
            if 0 < n_sig < len(sig)
            else float("nan")
        )
        if n_sig == 0:
            de_spearman_lfc = overlap = jaccard = float("nan")
        else:
            de_spearman_lfc = float(np.asarray(spearmanr(pred_row[sig], real_lfc[sig]))[0])
            order = np.argsort(-np.abs(pred_row))
            pred_top_n = set(genes[order[:n_sig]])
            true_sig = set(genes[sig])
            inter = len(pred_top_n & true_sig)
            overlap = inter / n_sig
            union = len(pred_top_n | true_sig)
            jaccard = inter / union if union else float("nan")
        rows.append(
            {
                "patient": patient,
                "drug": drug,
                "de_spearman_lfc": de_spearman_lfc,
                "pr_auc": pr_auc,
                "de_overlap_accuracy": overlap,
                "jaccard": jaccard,
                "n_sig_genes": n_sig,
            }
        )
    if not rows:
        raise ValueError("pred_key and de_calls share no (patient, drug) pairs")
    return pd.DataFrame(rows)


def score_de_metrics(
    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    de_calls: pd.DataFrame,
) -> pd.DataFrame:
    """DE-metrics analogue of ``score_delta_sources``: one row per delta source, averaged over
    its matched (patient, drug) pairs' DE Spearman LFC / PR-AUC / DE Overlap Accuracy / Jaccard
    (``de_fidelity``), against the same ground-truth DE-calls bundle
    (``fmharness.deltas.build_tahoe_de_calls``). Pairs with zero real-significant genes contribute
    NaN to the rank-based columns for that source and are excluded from those means via pandas'
    default ``skipna``, but still count toward the ``pr_auc`` mean when it is defined.
    """
    rows: list[dict[str, object]] = []
    for name, (d, kk) in sources.items():
        f = de_fidelity(d, kk, de_calls)
        rows.append(
            {
                "source": name,
                "de_spearman_lfc": round(float(f["de_spearman_lfc"].mean()), 3),
                "pr_auc": round(float(f["pr_auc"].mean()), 3),
                "de_overlap_accuracy": round(float(f["de_overlap_accuracy"].mean()), 3),
                "jaccard": round(float(f["jaccard"].mean()), 3),
                "n_pairs": len(f),
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluation.py -k "de_fidelity or score_de_metrics" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full evaluation test suite, lint, and type-check**

Run: `uv run pytest tests/test_evaluation.py -v`
Expected: all PASS (pre-existing tests + 3 new).

Run: `uv run ruff check src/fmharness/evaluation.py tests/test_evaluation.py`
Expected: clean.

Run: `uv run pyright src/fmharness/evaluation.py tests/test_evaluation.py`
Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add src/fmharness/evaluation.py tests/test_evaluation.py
git commit -m "feat: add de_fidelity/score_de_metrics, DE Spearman LFC/PR-AUC/Overlap/Jaccard scoring"
```

---

## Real-execution tasks

These tasks are not subagent-implementable in the usual sense: `ralpine` is read-only, so any
Alpine `sbatch` submission must be run by Lucas personally. Each Alpine-execution task below is a
coordination checkpoint — Claude prepares the exact command, Lucas submits, Claude polls
(`ralpine sacct`) and pulls (`ralpine pull`) results. Task 8 (Change 2's DE-calls build) is the
one exception: it needs no GPU and no Alpine queue at all, and can run directly in this worktree.

### Task 6: Alpine — Change 0 chain (`08` sci-Plex prep → `09` drug-align fine-tune)

**Depends on:** Task 1 (code fix landed and committed). Independent of Tasks 2/3/4/5/7/8.

**Files:** none (re-runs existing, unmodified `scripts/alpine/08_sciplex_prep.sbatch` and
`scripts/alpine/09_stack_finetune.sbatch` — Task 1's fix lives entirely inside
`build_sciplex_finetune.py`, which `08` already invokes unchanged).

- [ ] **Step 1: Confirm Task 1 is merged on the branch Alpine will pull/see, then hand Lucas the submission command for `08`**

```bash
sbatch scripts/alpine/08_sciplex_prep.sbatch
```

- [ ] **Step 2: Poll for completion**

Run: `ralpine sacct` (repeat until the `sciplex-prep` job shows `COMPLETED`).

- [ ] **Step 3: Verify the corrected control count in the log**

Run: `ralpine tail sciplex-prep` (or `ralpine log <jobid>`, whichever this project's `ralpine`
wrapper exposes). Confirm the logged control cell count is exactly **17,578** (not 54,100) and
that a line reading `dropping 36,522/799,317 cells with missing perturbation and/or cell-line
identity...` (Task 1's new print) appears. If either number differs, stop and investigate before
proceeding to `09` — do not fine-tune on an unverified input.

- [ ] **Step 4: Hand Lucas the submission command for `09`**

```bash
sbatch scripts/alpine/09_stack_finetune.sbatch
```

- [ ] **Step 5: Poll for completion and record the new checkpoint path**

Run: `ralpine sacct` until `stack-finetune` shows `COMPLETED`. Run `ralpine ls
/scratch/alpine/$USER/sciplex_finetune/` to get the new checkpoint's exact filename (bakes in a
val_loss, e.g. `finetuned-epoch=N-val_loss=X.ckpt` — will differ from the pre-fix run's
`val_loss=5.0847`). Record this path — it is `CKPT_SCIPLEX` for Task 9.

- [ ] **Step 6: No commit** (this task produces Alpine-side artifacts only; nothing in this
worktree changes).

---

### Task 7: Alpine — Change 1 cytokine-aligned generation (`03` → `04`)

**Depends on:** Task 2 (sbatch code) and Task 3 (aggregation module, needed in Step 4 below).
Independent of Task 6 — uses the already-existing cytokine-aligned checkpoint
(`stack-aligned/bc_large_aligned.ckpt`), not Task 6's new drug-aligned one.

**Files:** none (re-runs `scripts/alpine/03_stack_context.sbatch` and
`scripts/alpine/04_stack_generate.sbatch`, both edited by Task 2).

- [ ] **Step 1: Hand Lucas the submission command for `03`**

```bash
sbatch scripts/alpine/03_stack_context.sbatch
```

- [ ] **Step 2: Poll for completion, verify the new query shape**

Run: `ralpine sacct` until `tahoe-context` shows `COMPLETED`. Pull `tahoe_query.h5ad` (`ralpine
pull`) and confirm locally: `uv run python -c "import anndata as ad; a =
ad.read_h5ad('tahoe_query.h5ad'); print(a.n_obs, 'cell_line_id' in a.obs.columns)"` — expect
`400 True`.

- [ ] **Step 3: Hand Lucas the submission command for `04` (cytokine-aligned, default `CKPT`)**

```bash
sbatch scripts/alpine/04_stack_generate.sbatch
```

(No `--export=CKPT=...` needed — the script's default `CKPT` is already the cytokine-aligned
checkpoint.)

- [ ] **Step 4: Poll for completion; watch for time-limit issues on the new `--mode mdm` schedule**

Run: `ralpine sacct` until all 33 array tasks show `COMPLETED`. If any task shows `TIMEOUT`,
resubmit that single array index with a raised `--time` (per Task 2's flagged open question about
`mdm`'s per-step cost vs. `vanilla`'s) — do not silently drop it from the aggregate.

- [ ] **Step 5: Pull the generated output and run the new aggregation step, then calibrate the confidence threshold**

Pull the `generated/` directory locally via `ralpine pull`. Run the aggregation module (Task 3)
at a few candidate thresholds and pick the one that most improves Check-1 Pearson-Delta relative
to no filtering — the design's required calibration, since the paper's own value (`2.5`) was
tuned on a different checkpoint/task:

```bash
for T in 0.0 1.0 2.0 2.5 3.0; do
  uv run python -c "
from pathlib import Path
from fmharness.stack_aggregate import aggregate_generated_replicates
s = aggregate_generated_replicates(Path('generated'), Path(f'generated_agg_{$T}'), threshold=$T)
print('threshold=$T', 'kept_frac=', 1 - s['dropped'].mean())
"
done
```

For each threshold's `generated_agg_<T>/` directory, run `check1_registry_driver.py` (using
`--deltas-bundle tahoe_deltas` per its own documented reproducibility caveat) and compare the
`stack` row's Pearson-Delta `r`. Pick the threshold with the best `r`; record the choice and its
rationale in `docs/tahoe_generation_results.md` in Task 10.

- [ ] **Step 6: No commit yet** (results feed into Task 10's single docs-update commit).

---

### Task 8: Local — Change 2 DE-calls bundle build (no Alpine)

**Depends on:** Task 4 (code landed and committed). Fully independent of Tasks 6/7/9 — kick this
off immediately once Task 4 lands; it is the fastest path to a real result in this plan.

**Files:** none new (runs Task 4's `scripts/build_tahoe_de_calls.py` against the already-local
`tahoe_context.h5ad`).

- [ ] **Step 1: Run the build locally**

```bash
uv run python scripts/build_tahoe_de_calls.py --context tahoe_context.h5ad \
    --out tahoe_de_calls/de_calls.parquet
```

- [ ] **Step 2: Sanity-check the output against the already-known pair count**

`tahoe_deltas/real_key.parquet` already has one row per (line, drug) treated pair (~1,650, per
this project's own prior investigation). Confirm the new bundle's pair count matches:

```bash
uv run python -c "
import pandas as pd
de = pd.read_parquet('tahoe_de_calls/de_calls.parquet')
real_key = pd.read_parquet('tahoe_deltas/real_key.parquet')
n_de_pairs = len(de[['patient', 'drug']].drop_duplicates())
print('de_calls pairs:', n_de_pairs, ' real_key pairs:', len(real_key))
print('significant fraction:', de['significant'].mean())
"
```

If `n_de_pairs` differs materially from `real_key`'s count, investigate before proceeding (a
mismatch would mean the per-line control/treated grouping diverged between `build_tahoe_deltas`
and `build_tahoe_de_calls`, which must agree since both read the same `tahoe_context.h5ad`).

- [ ] **Step 3: If runtime is a concern, note it — do not silently sample down**

This step is real, one-time compute over ~1,650 pairs x ~15,012 genes. If it runs long locally,
that is worth noting in Task 10's writeup (and is grounds for moving this specific step to an
Alpine CPU job on a future run), but do not truncate the gene panel or pair count to speed it up
without flagging that explicitly — a silently-partial DE-calls bundle would understate Check 1's
real coverage.

- [ ] **Step 4: No commit** (the parquet bundle is a large generated artifact, matching
`tahoe_deltas/`'s own git-ignored status — confirm `tahoe_de_calls/` is covered by `.gitignore`
the same way, adding an entry if it is not already covered by a broader pattern).

---

### Task 9: Alpine — Change 1 drug-aligned generation (`04` again, new checkpoint)

**Depends on:** Task 6 (new drug-aligned checkpoint must exist) AND Task 2/3 (sbatch + aggregation
code). This is the convergence point of the Change-0 and Change-1 workstreams.

**Files:** none (re-runs `04_stack_generate.sbatch` with `CKPT` overridden to Task 6's new
checkpoint).

- [ ] **Step 1: Hand Lucas the submission command, `CKPT` overridden to Task 6's new checkpoint**

```bash
sbatch --export=CKPT=<Task 6's recorded checkpoint path>,OUTDIR=generated_drug_aligned \
    scripts/alpine/04_stack_generate.sbatch
```

(Per `04`'s own documented gotcha: must use `--export=CKPT=...,OUTDIR=...` explicitly, never
`--export=ALL` and never a bare prefix assignment — both silently fail in exactly the ways this
project has already hit in production.)

- [ ] **Step 2: Poll for completion**

Run: `ralpine sacct` until all array tasks show `COMPLETED` (same time-limit caveat as Task 7 Step
4).

- [ ] **Step 3: Pull, aggregate at the threshold Task 7 calibrated**

```bash
uv run python -c "
from pathlib import Path
from fmharness.stack_aggregate import aggregate_generated_replicates
aggregate_generated_replicates(
    Path('generated_drug_aligned'), Path('generated_drug_aligned_agg'), threshold=<Task 7's chosen threshold>
)
"
```

- [ ] **Step 4: No commit yet** (results feed into Task 10).

---

### Task 10: Re-derive and republish Check-1/Check-2 numbers

**Depends on:** Tasks 5, 7, 8, 9 all complete (Task 6 transitively, via Task 9).

**Files:**
- Modify: `docs/tahoe_generation_results.md`
- Modify: `scripts/update_harness_overview_slides.py`
- Modify: `scripts/plot_generation_eval_summary.py`

- [ ] **Step 1: Run Check 1 for both checkpoints, redirecting output to files (not inline)**

```bash
uv run python scripts/check1_registry_driver.py \
    --deltas-bundle tahoe_deltas \
    --query-baseline tahoe_query.h5ad \
    --generated-dir generated_agg_<Task 7's threshold> \
    --pert-map context_by_drug/pert_to_cid.tsv \
    --checkpoint-label cytokine-aligned > /tmp/check1_cytokine_aligned_v2.txt

uv run python scripts/check1_registry_driver.py \
    --deltas-bundle tahoe_deltas \
    --query-baseline tahoe_query.h5ad \
    --generated-dir generated_drug_aligned_agg \
    --pert-map context_by_drug/pert_to_cid.tsv \
    --checkpoint-label drug-aligned > /tmp/check1_drug_aligned_v2.txt
```

- [ ] **Step 2: Run the new DE metrics for both checkpoints**

`check1_registry_driver.py`'s `run_check1` builds its `sources` dict as a local variable (not
returned) — rather than changing that already-tested function's public contract for this one-off
analysis, reconstruct `sources`/`fd`/`fk` the same way it does, using the same public functions,
then score with `score_de_metrics` (Task 5). Write this to a not-committed one-off script (matches
this project's own `/tmp/*.txt` real-run convention) and run it once per checkpoint, e.g.
`/tmp/de_metrics_check1.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd

from fmharness.deltas import build_generated_deltas, learned_gene_panel, load_pert_map, loo_baseline_source
from fmharness.evaluation import score_de_metrics

repo = Path(".")
real_delta = pd.read_parquet("tahoe_deltas/real_delta.parquet")
real_key = pd.read_parquet("tahoe_deltas/real_key.parquet")
base = pd.read_parquet("tahoe_deltas/base.parquet")
de_calls = pd.read_parquet("tahoe_de_calls/de_calls.parquet")
pert_to_drug = load_pert_map(Path("context_by_drug/pert_to_cid.tsv"))

learned_genes = learned_gene_panel(real_delta, repo / "data/static/hallmark_signatures.gmt")
sources = {
    "additive": loo_baseline_source("additive", real_delta, real_key, base, k=10),
    "knn": loo_baseline_source("knn", real_delta, real_key, base, k=10),
    "pca": loo_baseline_source("pca", real_delta, real_key, base, k=10, genes=learned_genes),
    "nmf": loo_baseline_source("nmf", real_delta, real_key, base, k=10, genes=learned_genes),
    "stack": build_generated_deltas(
        Path("generated_agg_<threshold>"), Path("tahoe_query.h5ad"), pert_to_drug
    ),
}
print(score_de_metrics(sources, de_calls).to_string(index=False))
```

Run once per checkpoint (swap the `stack` source's `generated_agg_<threshold>` path for
`generated_drug_aligned_agg` on the second run), redirecting each to its own file:

```bash
uv run python /tmp/de_metrics_check1.py > /tmp/check1_de_metrics_cytokine_aligned_v2.txt
uv run python /tmp/de_metrics_check1.py > /tmp/check1_de_metrics_drug_aligned_v2.txt   # after editing the stack path
```

This intentionally does not apply Step 1's leakage filter (`filter_leakage`) — the DE-calls bundle
is scored on the full pair set here; if a leakage-filtered DE comparison is wanted later, that is
a follow-on, not required for this plan's exit criteria. Sanity-check: every source's `n_pairs` in
this output should be close to (not necessarily identical to, since this run is unfiltered) Step
1's Pearson-Delta table `n_pairs` for the same checkpoint.

- [ ] **Step 3: Sanity-check new numbers against the already-published pre-fix ones before trusting them**

Read the current `docs/tahoe_generation_results.md`'s existing Check-1 table. The new numbers
should differ (that is the point of this plan) but should not be wildly implausible — e.g. Stack's
Pearson-Delta `r` moving from a specific pre-fix value to a new one is expected; a jump to
exactly 0 or 1, or a sign flip on `r_offdiag`, would indicate a real problem (wrong file pulled,
mismatched checkpoint/generated-dir pairing) rather than a genuine result. If anything looks
implausible, stop and investigate rather than publishing it.

- [ ] **Step 4: Update `docs/tahoe_generation_results.md`**

Following this doc's own established long-format convention (extra rows, not columns — see the
prior Check-2 plan's Task 5 for the exact precedent), add new rows for both checkpoints' Pearson-
Delta AND the four new DE metrics, each row noting the run date and the three changes now in
effect (Change 0 checkpoint / Change 1 procedure+threshold / Change 2 metric). Do not delete the
prior (pre-fix) rows — this doc's history is a record of what changed and why, matching the
established pattern.

- [ ] **Step 5: Update the deck-generation scripts' hardcoded tables**

`scripts/update_harness_overview_slides.py`'s hardcoded row tuples and
`scripts/plot_generation_eval_summary.py`'s hardcoded dicts need the new numbers added, following
the exact pattern the prior Check-2 plan's Task 5 established (read both files' current hardcoded
structures before editing — do not guess the shape).

- [ ] **Step 6: Regenerate the deck, confirm with Lucas before copying back**

Copy the gitignored `.pptx` deck in from the main worktree, run the slide-update script, and
confirm with Lucas before copying the regenerated deck back out — per this project's established
convention (the deck lives outside version control and copy-back is a one-way, human-confirmed
step).

- [ ] **Step 7: Final docs-only commit**

```bash
git add docs/tahoe_generation_results.md scripts/update_harness_overview_slides.py scripts/plot_generation_eval_summary.py
git commit -m "docs: re-derive Check-1/Check-2 numbers under Change 0/1/2 (faithful generation + DE metrics)"
```

---

## Execution ordering for ASAP results

Read this section top-to-bottom to know what to kick off first.

**Immediately, in parallel (5 independent code tasks — dispatch/implement concurrently):**
Task 1, Task 2, Task 3, Task 4, Task 5.

**As soon as each lands:**
- Task 4 done → **Task 8 immediately** (local, CPU-only, no Alpine queue — the single fastest
  path to a real result in this entire plan; do not wait for anything else).
- Task 1 done → Task 6 (Alpine: `08` then `09`, hard sequential chain internally, ~8h `08` +
  ~8h `09` budget per their own `--time` directives).
- Task 2 + Task 3 done → Task 7 (Alpine: `03` then `04`, cytokine-aligned — independent of Task
  6, can run concurrently with it on Alpine; different jobs, no shared file).

**Critical path:** Task 1 → Task 6 → Task 9 → Task 10. Task 6's `09` fine-tune (`--time=08:00:00`
budget) is the single longest step in the whole plan — everything else (Task 7, Task 8) should be
finished well before Task 9 can even start, so there is no benefit to rushing Tasks 2-5's review
once Task 1 is moving; the bottleneck is Alpine GPU time on Task 6, not local code review speed.

**Convergence:** Task 9 depends on Task 6 (new checkpoint) AND Task 2/3 (sbatch+aggregation code)
— submit it the moment Task 6's checkpoint path is in hand. Task 10 depends on Tasks 5, 7, 8, 9 —
by the time Task 9 completes, Tasks 5, 7, and 8 should already be long done, so Task 10 can start
immediately after Task 9's log confirms `COMPLETED`.

**What NOT to serialize:** Do not wait for Task 7 (cytokine-aligned generation) to finish before
submitting Task 6's `08`/`09` chain, or vice versa — they touch disjoint files and disjoint
checkpoints. Do not wait for either Alpine workstream before running Task 8 — it needs no GPU and
no queue at all.

---

## Plan exit criteria

- All five code tasks (1-5) committed, each with passing tests, clean `ruff`, clean `pyright`.
- Task 6's Alpine chain produces a drug-aligned checkpoint with a verified 17,578-control input
  (not 54,100).
- Task 7 and Task 9 both produce a `--mode mdm` generation run with zero `IndexError`s across all
  33 array tasks, aggregated via Task 3's confidence filter at a calibrated (not paper-copied)
  threshold.
- Task 8 produces a DE-calls bundle whose pair count matches `tahoe_deltas/real_key.parquet`'s
  existing pair count.
- `docs/tahoe_generation_results.md` and the harness-overview deck carry new rows for both
  checkpoints' Pearson-Delta and all four new DE metrics, without deleting the prior (pre-fix)
  rows.

## Not in this plan

- Fixing the other sbatch scripts' stale `--partition=amilan` references (`00`, `01`, `02`, `05`,
  `07`, `delta_reproducibility`) — explicitly out of scope per the approved design spec; only `03`
  is touched here because Change 1 already modifies it.
- Re-deriving Check 2 (biomarker/leakage-aware scoring) under the new checkpoint/procedure — this
  plan only re-derives Check 1's Pearson-Delta and the new DE metrics; a Check-2 re-derivation
  under Change 0's corrected checkpoint would be a natural follow-on but is not required for this
  plan's own exit criteria.
- Moving Task 8's DE-calls build to an Alpine CPU job — only warranted if local runtime proves
  prohibitive (see Task 8 Step 3); not assumed necessary here.
- Investigating the ~82 thin (line, drug) base/prompt pools (down to 17 cells, concentrated on
  ACH-000628/ACH-000311) flagged during the original brainstorming pass — a real, open question
  about generation quality for those specific pairs, but a separate investigation from the three
  changes this plan implements.
