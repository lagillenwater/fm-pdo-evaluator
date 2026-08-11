# Stack Drug-Alignment + Registry-Driven Check 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the four documented blockers stopping Stack's drug-alignment fine-tune from running on Alpine, and build a registry-driven Check-1 (delta-Pearson generation quality) driver that reproduces the published table and adds the drug-aligned checkpoint as a new row.

**Architecture:** Phase 1 (Tasks 1-5) edits `scripts/alpine/08_sciplex_prep.sbatch`, `scripts/build_sciplex_finetune.py`, and `scripts/alpine/09_stack_finetune.sbatch` so a real fine-tune can run on Alpine (Lucas submits, Claude polls via `ralpine`). Phase 2 (Tasks 6-9) moves the reusable Check-1 scoring logic already living inside `scripts/score_generation_eval.py` into `fmharness/deltas.py`, adds a `Generator`-protocol wrapper over Stack's pre-generated output, and builds a new driver script that reproduces the existing Check-1 table and (once Phase 1 lands) adds the drug-aligned row.

**Tech Stack:** Python 3.11, pandas/numpy/scipy, anndata, pytest, uv. Alpine SLURM sbatch scripts (bash) for Phase 1; `ralpine` (read-only helper) for polling/pulling Alpine results.

## Global Constraints

- Line length 100 (`[tool.ruff]` in `pyproject.toml`).
- `target-version = "py311"`; pyright `typeCheckingMode = "strict"` over `src` and `tests`.
- Ruff lint selects `E, F, I, B, UP, SIM, RUF`. `tests/**` ignores `E501` only.
- Run everything through `uv`: `uv run pytest`, `uv run python`, `uv run ruff`, `uv run pyright`.
- No emojis anywhere -- code, comments, output, commit messages.
- Vectorized only. No nested Python loops over data rows. A loop over a small, fixed set
  (drugs, delta sources, cell lines in a leave-one-out rebuild) is fine; a loop over samples
  or genes is not.
- New public functions/classes need docstrings: what it computes, and why that is the right
  design.
- Every new `Protocol` implementation must be provable via `isinstance` against the real
  protocol (`Generator` from `fmharness.model_protocols`), not just structurally assumed.
- `ralpine` (`scripts/alpine/ralpine`) is read-only by design: `status`, `sq`, `sacct`, `ls`,
  `tail`, `cat`, `log`, `rev`, `du`, `run <allowlisted-cmd>`, `pull <remote> <local>`. No
  `sbatch`/`scancel` -- job submission and cancellation are Lucas's, always. Tasks 1-5 produce
  committed sbatch-script edits for Lucas to submit; they do not submit anything themselves.
- Commit after each task (worktree-local; this branch's standing arrangement from the prior
  SDD execution continues -- see `.superpowers/sdd/` ledgers from the modular-harness-core plan
  if precedent is needed).

---

### Task 1: Switch sci-Plex source to the full-gene scPerturb release

The chemCPA-processed sci-Plex file (`sciplex_complete_middle_subset.h5ad`) is pre-subset to
2,000 HVGs; Stack's generation panel is 15,012 genes. Fine-tuning on 2,000 genes and then
generating over 15,012 is a train/test mismatch. The scPerturb-hosted release
(`SrivatsanTrapnell2020_sciplex3.h5ad`, Zenodo record 13350497) is 2.5 GB for ~650k cells --
verified live (`curl -I` returns `content-length: 2526631614`) -- far too large to be a
2,000-gene subset, so almost certainly a much larger gene panel.
`scripts/build_sciplex_finetune.py`'s docstring already documents this file's schema as its
"scPerturb" auto-detected flavor; this task adds a real, enforced check instead of relying on
eyeballing logs.

**Files:**
- Create: `src/fmharness/sciplex_prep.py`
- Modify: `scripts/build_sciplex_finetune.py`
- Modify: `scripts/alpine/08_sciplex_prep.sbatch`
- Test: `tests/test_sciplex_prep.py`

**Interfaces:**
- Produces: `check_gene_count(n_vars: int, *, min_genes: int = 5000) -> None` (raises
  `SystemExit` if too few genes). Used by Task 2 and Task 3's module (same file).

- [ ] **Step 1: Write the failing test**

Create `tests/test_sciplex_prep.py`:

```python
"""Tests for sci-Plex fine-tune input validation (raw counts, gene panel, name collisions)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from fmharness.sciplex_prep import check_gene_count


def test_check_gene_count_raises_on_a_subset_panel() -> None:
    with pytest.raises(SystemExit, match="2000 genes"):
        check_gene_count(2000, min_genes=5000)


def test_check_gene_count_passes_on_a_near_full_panel() -> None:
    check_gene_count(15012, min_genes=5000)  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sciplex_prep.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmharness.sciplex_prep'`

- [ ] **Step 3: Write minimal implementation**

Create `src/fmharness/sciplex_prep.py`:

```python
"""Input validation for sci-Plex 3 data feeding Stack's drug-alignment fine-tune.

``scripts/build_sciplex_finetune.py`` reformats a sci-Plex AnnData for
``stack-finetune``; these checks catch the two failure modes found (2026-08-06) but never
enforced -- a pre-subset gene panel and non-raw-count input -- plus a name-collision
diagnostic for a third (upstream perturbation-name truncation). Kept importable and testable
independently of the CLI script, which only orchestrates argument parsing and I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def check_gene_count(n_vars: int, *, min_genes: int = 5000) -> None:
    """Raise if the input looks like a pre-subset HVG panel, not a near-full transcriptome.

    Stack's generation panel is 15,012 genes; fine-tuning the generation head on a much
    smaller subset (e.g. the chemCPA sci-Plex release's 2,000-gene HVG subset) and then
    generating over the full panel is a train/test mismatch. ``min_genes=5000`` is a loose
    floor -- well above 2,000, well below a full transcriptome (~20,000+) -- so it catches
    the known-bad case without demanding an exact match to Stack's panel size.
    """
    if n_vars < min_genes:
        raise SystemExit(
            f"input has only {n_vars} genes (< {min_genes}) -- looks like a pre-subset HVG "
            "panel, not a near-full transcriptome. Stack's generation panel is 15,012 genes; "
            "fine-tuning on a much smaller subset would be a train/test mismatch. Use a "
            "full-gene source instead (see scripts/alpine/08_sciplex_prep.sbatch's SCIPLEX_URL)."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sciplex_prep.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Wire the check into `build_sciplex_finetune.py`**

In `scripts/build_sciplex_finetune.py`, add the import near the top (after the existing
`from scipy import sparse` line):

```python
from fmharness.sciplex_prep import check_gene_count
```

In `main()`, immediately after the line `a = ad.read_h5ad(args.input)`, add:

```python
    check_gene_count(a.n_vars)
```

- [ ] **Step 6: Switch the default source in `08_sciplex_prep.sbatch`**

In `scripts/alpine/08_sciplex_prep.sbatch`, change line 30's comment and line 31's default URL:

```bash
# 1. Download the sci-Plex 3 AnnData (scPerturb's full-gene h5ad, Zenodo record 13350497 --
#    see the chemCPA "middle_subset" alternative below for the pre-subset-to-2000-genes
#    version this replaces; the full-gene version is what Stack's generation panel needs).
#    Override SCIPLEX_URL to swap it.
SCIPLEX_URL="${SCIPLEX_URL:-https://zenodo.org/records/13350497/files/SrivatsanTrapnell2020_sciplex3.h5ad?download=1}"
```

- [ ] **Step 7: Verify the script still parses**

Run: `bash -n scripts/alpine/08_sciplex_prep.sbatch`
Expected: no output, exit 0 (syntax valid)

Run: `uv run python -c "import ast; ast.parse(open('scripts/build_sciplex_finetune.py').read())"`
Expected: no error

- [ ] **Step 8: Lint and typecheck**

Run: `uv run ruff check src/fmharness/sciplex_prep.py scripts/build_sciplex_finetune.py tests/test_sciplex_prep.py && uv run pyright src tests`
Expected: no errors (pyright: only the one pre-existing, unrelated `src/fmharness/deltas.py:194`
error).

- [ ] **Step 9: Commit**

```bash
git add src/fmharness/sciplex_prep.py tests/test_sciplex_prep.py scripts/build_sciplex_finetune.py scripts/alpine/08_sciplex_prep.sbatch
git commit -m "fix: source full-gene sci-Plex from scPerturb, enforce the gene-count floor"
```

---

### Task 2: Add a real raw-counts check

`build_sciplex_finetune.py` currently only prints `.X (VERIFY these are raw counts, not
normalized)` when it falls back to `.X` -- nothing has ever acted on that warning. Stack is a
count model (NB likelihood); normalized input breaks it.

**Files:**
- Modify: `src/fmharness/sciplex_prep.py`
- Modify: `scripts/build_sciplex_finetune.py`
- Test: `tests/test_sciplex_prep.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `check_raw_counts(x: scipy.sparse.csr_matrix, source: str) -> None` (raises
  `SystemExit` on negative or non-integer values).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sciplex_prep.py`:

```python
from fmharness.sciplex_prep import check_gene_count, check_raw_counts


def test_check_raw_counts_raises_on_negative_values() -> None:
    x = sparse.csr_matrix(np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32))
    with pytest.raises(SystemExit, match="raw counts"):
        check_raw_counts(x, "layer 'counts'")


def test_check_raw_counts_raises_on_non_integer_values() -> None:
    x = sparse.csr_matrix(np.array([[1.5, 2.0], [3.0, 4.0]], dtype=np.float32))
    with pytest.raises(SystemExit, match="raw counts"):
        check_raw_counts(x, ".X")


def test_check_raw_counts_passes_on_real_counts() -> None:
    x = sparse.csr_matrix(np.array([[0.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    check_raw_counts(x, "layer 'counts'")  # must not raise


def test_check_raw_counts_passes_on_an_all_zero_matrix() -> None:
    x = sparse.csr_matrix(np.zeros((2, 2), dtype=np.float32))
    check_raw_counts(x, "layer 'counts'")  # empty x.data -- must not raise
```

Update the existing `from fmharness.sciplex_prep import check_gene_count` line to import both
names, as shown above (replaces the single-name import from Task 1).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sciplex_prep.py -v`
Expected: FAIL with `ImportError: cannot import name 'check_raw_counts'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/fmharness/sciplex_prep.py`:

```python
def check_raw_counts(x: sparse.csr_matrix, source: str) -> None:
    """Raise if ``x`` does not look like raw counts (Stack is a count model, NB likelihood).

    Checked on ``x.data`` -- the flat array of stored (nonzero) values in the CSR structure
    -- so this covers dense-then-sparsified input identically to native sparse input, with
    no per-row/per-cell loop.
    """
    if x.data.size and (not np.all(x.data >= 0) or not np.allclose(x.data, np.round(x.data))):
        raise SystemExit(
            f"{source} does not look like raw counts (found negative or non-integer values); "
            "pass --counts-layer to point at the correct layer"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sciplex_prep.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Wire the check into `build_sciplex_finetune.py`**

Add the import (extend the Task-1 import line to include `check_raw_counts`):

```python
from fmharness.sciplex_prep import check_gene_count, check_raw_counts
```

Immediately after the existing line
`x = (x if sparse.issparse(x) else sparse.csr_matrix(np.asarray(x))).tocsr().astype(np.float32)`,
add:

```python
    check_raw_counts(x, src)
```

(`src` is already defined two lines above by the existing `if args.counts_layer: ... elif
"counts" in a.layers: ... else: x, src = a.X, ".X (VERIFY these are raw counts, not
normalized)"` block -- reuse it, do not redefine.)

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest -v`
Expected: all pass, no regressions.

- [ ] **Step 7: Lint and typecheck**

Run: `uv run ruff check src/fmharness/sciplex_prep.py scripts/build_sciplex_finetune.py tests/test_sciplex_prep.py && uv run pyright src tests`
Expected: no errors (same one pre-existing exception as Task 1).

- [ ] **Step 8: Commit**

```bash
git add src/fmharness/sciplex_prep.py tests/test_sciplex_prep.py scripts/build_sciplex_finetune.py
git commit -m "fix: enforce the raw-counts check in sci-Plex prep instead of only warning"
```

---

### Task 3: Add a drug-name-collision diagnostic

Found in the chemCPA source specifically: `pert_id` values truncated at first whitespace
("AZ", "GSK", "ZM", ...). Switching to the scPerturb source (Task 1) may or may not carry this
forward -- this is a diagnostic re-check, not an assumption the old finding still applies.

**Files:**
- Modify: `src/fmharness/sciplex_prep.py`
- Modify: `scripts/build_sciplex_finetune.py`
- Test: `tests/test_sciplex_prep.py`

**Interfaces:**
- Produces: `check_perturbation_count(perturbations: pd.Series, *, expected_min_distinct: int
  = 100) -> None` (prints a warning, does not raise -- see rationale below).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sciplex_prep.py`:

```python
from fmharness.sciplex_prep import check_gene_count, check_perturbation_count, check_raw_counts


def test_check_perturbation_count_warns_below_the_floor(capsys: pytest.CaptureFixture[str]) -> None:
    perts = pd.Series(["AZ", "AZ", "GSK", "GSK", "control"])
    check_perturbation_count(perts, expected_min_distinct=100)
    assert "WARNING" in capsys.readouterr().out


def test_check_perturbation_count_silent_above_the_floor(capsys: pytest.CaptureFixture[str]) -> None:
    perts = pd.Series([f"drug{i}" for i in range(150)])
    check_perturbation_count(perts, expected_min_distinct=100)
    assert capsys.readouterr().out == ""
```

Update the import line to include the new name (extends Tasks 1-2's import).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sciplex_prep.py -v`
Expected: FAIL with `ImportError: cannot import name 'check_perturbation_count'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/fmharness/sciplex_prep.py`:

```python
def check_perturbation_count(perturbations: pd.Series, *, expected_min_distinct: int = 100) -> None:
    """Warn (not raise) if too few distinct perturbation names survived.

    Upstream name truncation -- seen once already in the chemCPA sci-Plex release
    ("AZ", "GSK", "ZM" from names cut at the first whitespace) -- silently collapses
    distinct compounds into one label. sci-Plex 3 has ~188 published compounds;
    ``expected_min_distinct=100`` is a loose floor. A warning, not a hard failure: the exact
    expected count depends on which upstream release is in use and how doses/controls are
    represented, so this flags a suspicious count for a human to check rather than blocking
    on an assumption about the exact number.
    """
    n_distinct = perturbations.nunique()
    if n_distinct < expected_min_distinct:
        print(
            f"WARNING: only {n_distinct} distinct perturbations found (expected ~188 for "
            f"sci-Plex 3, floor {expected_min_distinct}) -- check for upstream name "
            f"truncation/collisions before training on this data"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sciplex_prep.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Wire the check into `build_sciplex_finetune.py`**

Extend the import line:

```python
from fmharness.sciplex_prep import check_gene_count, check_perturbation_count, check_raw_counts
```

Immediately after the existing line `pert = a.obs[pert_col].astype(str).to_numpy()`, add:

```python
    check_perturbation_count(pd.Series(pert))
```

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest -v`
Expected: all pass, no regressions.

- [ ] **Step 7: Lint and typecheck**

Run: `uv run ruff check src/fmharness/sciplex_prep.py scripts/build_sciplex_finetune.py tests/test_sciplex_prep.py && uv run pyright src tests`
Expected: no errors (same one pre-existing exception).

- [ ] **Step 8: Commit**

```bash
git add src/fmharness/sciplex_prep.py tests/test_sciplex_prep.py scripts/build_sciplex_finetune.py
git commit -m "fix: warn on suspiciously few distinct sci-Plex perturbations (name-truncation check)"
```

---

### Task 4: Fix the `GLIBCXX_3.4.29` crash in both `09` and `04`

The `stack` conda env's numpy is built against a newer libstdc++ than the Alpine node
provides, crashing at import (`ImportError: /lib64/libstdc++.so.6: version GLIBCXX_3.4.29 not
found`, raised from `numpy/fft/_pocketfft_umath`). Confirmed present in both
`09_stack_finetune.sbatch` (line 29: `conda activate stack`) and `04_stack_generate.sbatch`
(line 34: `conda activate stack        # <-- CHANGE to your Stack env`) -- same conda env name,
neither script currently sets `LD_LIBRARY_PATH`. Environment bug, not a modelling problem; fix
via `LD_LIBRARY_PATH` ordering, not by mutating the shared conda env.

**Files:**
- Modify: `scripts/alpine/09_stack_finetune.sbatch`
- Modify: `scripts/alpine/04_stack_generate.sbatch`

**Interfaces:** none (shell scripts only).

- [ ] **Step 1: Fix `09_stack_finetune.sbatch`**

Insert immediately after line 29 (`conda activate stack`), before the blank line that precedes
the `SCIPLEX=` variable block:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"  # GLIBCXX_3.4.29 fix
```

- [ ] **Step 2: Fix `04_stack_generate.sbatch`**

Insert immediately after line 34 (`conda activate stack        # <-- CHANGE to your Stack
env`), before the `export HF_HOME=...` line:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"  # GLIBCXX_3.4.29 fix
```

- [ ] **Step 3: Verify both scripts still parse**

Run: `bash -n scripts/alpine/09_stack_finetune.sbatch && bash -n scripts/alpine/04_stack_generate.sbatch`
Expected: no output, exit 0.

- [ ] **Step 4: Commit**

```bash
git add scripts/alpine/09_stack_finetune.sbatch scripts/alpine/04_stack_generate.sbatch
git commit -m "fix: prepend CONDA_PREFIX/lib to LD_LIBRARY_PATH to fix the GLIBCXX_3.4.29 crash"
```

---

### Task 5: Promote `09_stack_finetune.sbatch` from smoke test to a real run

Current config is a 1-epoch smoke test on the testing QOS. `04_stack_generate.sbatch` (the
sibling generation script, already a real, non-testing job) uses `--qos=gpu-normal` --
reuse that exact, already-working value rather than guessing at Alpine's QOS naming.

**Files:**
- Modify: `scripts/alpine/09_stack_finetune.sbatch`

**Interfaces:** none (shell script only).

- [ ] **Step 1: Change the QOS**

Line 4, from:
```bash
#SBATCH --qos=gpu-testing
```
to:
```bash
#SBATCH --qos=gpu-normal
```

- [ ] **Step 2: Raise the epoch count**

Line 43, from:
```bash
    --max_epochs 1 \
```
to:
```bash
    --max_epochs 10 \
```

(10 is a reasonable starting real-run epoch count for a fine-tune of this size; there is no
observed-epoch-time data yet to compute an exact value from, since the smoke-test job never
successfully ran past the import crash Task 4 fixes. Note in the job's trailing comment, per
Step 3 below, that `--time` may need another adjustment once real epoch time is observed.)

- [ ] **Step 3: Raise the wall-clock time and update its comment**

Line 10, from:
```bash
#SBATCH --time=1:00:00              # training run; raise/lower once epoch time is known
```
to:
```bash
#SBATCH --time=8:00:00              # real run, 10 epochs; re-tune once epoch time is observed
```

- [ ] **Step 4: Verify the script still parses**

Run: `bash -n scripts/alpine/09_stack_finetune.sbatch`
Expected: no output, exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/alpine/09_stack_finetune.sbatch
git commit -m "feat: promote sci-Plex drug-alignment fine-tune from smoke test to a real run"
```

**This is the end of Phase 1's committable work.** The next step is Lucas submitting, in
order: `08_sciplex_prep.sbatch` (CPU, `amilan`), then `09_stack_finetune.sbatch` (GPU,
`aa100`), then (once a checkpoint exists) `04_stack_generate.sbatch` with `CKPT`/`OUTDIR`
overrides pointing at the new checkpoint, per `09`'s own trailing comment. This plan does not
submit anything -- see Task 9 for what to do once results land.

---

### Task 6: Move `_loo_baseline_source` and the learned-gene-panel builder into `fmharness/deltas.py`

`score_generation_eval.py` currently defines `_loo_baseline_source` (the leave-one-line-out
rebuild used for every baseline delta row) and inlines the HVG+Hallmark gene-panel
construction for the `pca`/`nmf` sources -- both needed, unchanged, by Task 8's new driver.
Moving them into `fmharness/deltas.py` (already the home of every other delta-building
function, and already tested) avoids duplicating ~50 lines of rebuild logic across two
scripts. Pure refactor: `score_generation_eval.py`'s own behavior and output must not change.

**Files:**
- Modify: `src/fmharness/deltas.py`
- Modify: `scripts/score_generation_eval.py`
- Test: `tests/test_deltas.py`

**Interfaces:**
- Consumes: `build_additive_deltas`, `build_knn_deltas`, `build_learned_deltas` (all existing,
  unchanged, already in `deltas.py`); `load_hallmark` (`fmharness.signatures`, existing).
- Produces: `loo_baseline_source(kind: str, real_delta: pd.DataFrame, real_key: pd.DataFrame,
  base: pd.DataFrame, *, k: int, genes: pd.Index | None = None) -> tuple[pd.DataFrame,
  pd.DataFrame]` and `learned_gene_panel(real_delta: pd.DataFrame, hallmark_path: Path, *,
  n_hvg: int = 2000) -> pd.Index`. Used by Task 8.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_deltas.py` (extend the existing import block at the top to add
`loo_baseline_source` and `learned_gene_panel` to the `from fmharness.deltas import (...)`
list, alphabetically):

```python
def test_loo_baseline_source_additive_never_sees_its_own_held_out_line() -> None:
    # 3 lines, 1 drug each with a distinct delta value; additive's held-out prediction for
    # each line must come only from the OTHER two lines' mean, never its own value.
    genes = ["A"]
    real_delta = pd.DataFrame({"A": [10.0, 20.0, 30.0]})
    real_key = pd.DataFrame({"patient": ["L1", "L2", "L3"], "drug": ["d1", "d1", "d1"]})
    base = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=pd.Index(["L1", "L2", "L3"]))

    delta, key = loo_baseline_source("additive", real_delta, real_key, base, k=1)

    assert len(delta) == 3
    want = {"L1": (20.0 + 30.0) / 2, "L2": (10.0 + 30.0) / 2, "L3": (10.0 + 20.0) / 2}
    for line, expected in want.items():
        row = delta[key["patient"].to_numpy() == line]
        assert np.isclose(float(row["A"].iloc[0]), expected)


def test_loo_baseline_source_raises_on_unknown_kind() -> None:
    real_delta = pd.DataFrame({"A": [1.0]})
    real_key = pd.DataFrame({"patient": ["L1"], "drug": ["d1"]})
    base = pd.DataFrame({"A": [0.0]}, index=pd.Index(["L1"]))
    with pytest.raises(ValueError, match="unknown baseline source"):
        loo_baseline_source("bogus", real_delta, real_key, base, k=1)


def test_learned_gene_panel_unions_hvgs_and_hallmark_genes(tmp_path: Path) -> None:
    gmt = tmp_path / "hallmark.gmt"
    gmt.write_text("HALLMARK_TEST\thttp://example\tSIGGENE1\tSIGGENE2\n")
    # 4 genes, HVG1 has the highest variance (picked at n_hvg=1); SIGGENE1/SIGGENE2 come in
    # from the hallmark set regardless of their own variance.
    real_delta = pd.DataFrame(
        {
            "HVG1": [1.0, 100.0, -50.0],
            "HVG2": [1.0, 1.0, 1.0],
            "SIGGENE1": [1.0, 1.0, 1.0],
            "SIGGENE2": [1.0, 1.0, 1.0],
        }
    )
    panel = learned_gene_panel(real_delta, gmt, n_hvg=1)
    assert set(panel) == {"HVG1", "SIGGENE1", "SIGGENE2"}
```

Add `import pytest` to `tests/test_deltas.py`'s imports if not already present (check the
existing file first -- Task 6's implementer should read the current import block before
editing, since other tasks in this plan do not touch this file).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deltas.py -v`
Expected: FAIL with `ImportError: cannot import name 'loo_baseline_source'`

- [ ] **Step 3: Move the implementation into `deltas.py`**

Add to `src/fmharness/deltas.py` (near the end, after `build_knn_deltas`; needs
`from pathlib import Path` if not already imported -- check the existing import block first):

```python
def loo_baseline_source(
    kind: str,
    real_delta: pd.DataFrame,
    real_key: pd.DataFrame,
    base: pd.DataFrame,
    *,
    k: int,
    genes: pd.Index | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-cell-line-out baseline deltas: for each line, rebuild the source from the
    OTHER lines and predict the held-out line, so it never sees its own treated cells.

    ``additive``/``knn`` use all genes; ``pca``/``nmf`` (``build_learned_deltas``) reduce on
    the ``genes`` HVG panel, which keeps the per-line PCA/NMF fast and well-conditioned (49
    lines vs ~50k genes is hopelessly p>>n; on ~2k informative genes it is sane).
    """
    pats = real_key["patient"].astype(str).to_numpy()
    rdl = real_delta if genes is None else real_delta[[g for g in genes if g in real_delta.columns]]
    bl = base if genes is None else base[[g for g in rdl.columns if g in base.columns]]
    d_blocks: list[pd.DataFrame] = []
    k_blocks: list[pd.DataFrame] = []
    for line in [str(i) for i in base.index]:
        tr = pats != line
        if not tr.any():
            continue
        rd = real_delta[tr].reset_index(drop=True)
        rk = real_key[tr].reset_index(drop=True)
        if kind == "additive":
            d, kk = build_additive_deltas(rd, rk, [line])
        elif kind == "knn":
            d, kk = build_knn_deltas(base.drop(index=line), rd, rk, base.loc[[line]], [line], k=k)
        elif kind in ("pca", "nmf"):
            d, kk = build_learned_deltas(
                bl.drop(index=line),
                rdl[tr].reset_index(drop=True),
                rk,
                bl.loc[[line]],
                [line],
                reducer=kind,
            )
        else:
            raise ValueError(f"unknown baseline source {kind!r}")
        d_blocks.append(d)
        k_blocks.append(kk)
    if not d_blocks:
        raise ValueError(f"no held-out lines produced a {kind} delta")
    return pd.concat(d_blocks, ignore_index=True), pd.concat(k_blocks, ignore_index=True)


def learned_gene_panel(real_delta: pd.DataFrame, hallmark_path: Path, *, n_hvg: int = 2000) -> pd.Index:
    """HVG-union-Hallmark gene panel for the ``pca``/``nmf`` delta sources.

    The top ``n_hvg`` most-variable genes of the real delta, unioned with every gene named
    in any Hallmark signature -- so the learned reducers see both the highest-signal genes
    and the genes the fixed-signature readouts score on, keeping the two checks comparable.
    """
    hallmark = load_hallmark(hallmark_path)
    sig_genes = pd.Index(sorted({g for genes, _ in hallmark.values() for g in genes}))
    hvg = pd.Index(real_delta.var(axis=0).sort_values(ascending=False).index[:n_hvg])
    return hvg.union(sig_genes)
```

Add the import this needs at the top of `deltas.py`:

```python
from fmharness.signatures import load_hallmark
```

(Check the existing import block first -- add alphabetically alongside whatever's already
imported from `fmharness.*`, if anything.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_deltas.py -v`
Expected: PASS (all existing tests plus the 3 new ones)

- [ ] **Step 5: Update `score_generation_eval.py` to import instead of redefine**

Delete the local `_loo_baseline_source` function definition (its full body, currently
duplicating what Step 3 just moved) and the inline `learned_genes` construction
(the `hallmark = load_hallmark(...)` / `sig_genes = ...` / `hvg = ...` / `learned_genes =
hvg.union(sig_genes)` block). Add near the top-level imports:

```python
from fmharness.deltas import learned_gene_panel, loo_baseline_source
```

Replace every call site of `_loo_baseline_source(...)` with `loo_baseline_source(...)`
(same arguments, name only). Replace the inline `learned_genes` construction with:

```python
    learned_genes = learned_gene_panel(real_delta, repo / "data/static/hallmark_signatures.gmt", n_hvg=args.n_hvg)
```

This keeps every existing call site and CLI behavior identical -- pure rename plus one
inlined-to-function-call replacement, no logic change.

- [ ] **Step 6: Verify the script still imports cleanly**

Run: `uv run python -c "import ast; ast.parse(open('scripts/score_generation_eval.py').read())"`
Expected: no error. Do not execute `scripts/score_generation_eval.py` itself -- it needs real
Tahoe data this worktree does not have (see Task 9).

- [ ] **Step 7: Run the full test suite**

Run: `uv run pytest -v`
Expected: all pass, no regressions.

- [ ] **Step 8: Lint and typecheck**

Run: `uv run ruff check src/fmharness/deltas.py scripts/score_generation_eval.py tests/test_deltas.py && uv run pyright src tests`
Expected: no errors (same one pre-existing exception; `scripts/` is not under pyright's
strict scope but ruff must still be clean on it).

- [ ] **Step 9: Commit**

```bash
git add src/fmharness/deltas.py scripts/score_generation_eval.py tests/test_deltas.py
git commit -m "refactor: move loo_baseline_source and learned_gene_panel into fmharness.deltas"
```

---

### Task 7: `PregeneratedStackGenerator` -- Generator-protocol wrapper over pre-generated Stack output

Stack generation runs on Alpine GPU (`scripts/alpine/04_stack_generate.sbatch`), writing
`<pert_id>.h5ad` files to an output directory -- this class does not run inference; it
satisfies the `Generator` protocol (`fmharness.model_protocols`) by resolving a requested
perturbation to its pre-generated file, using the same resolution rule
`build_generated_deltas`'s `_drug_of` already implements (filename stem, sanitized-underscore
stem, or `.uns` fallback), so a caller using only the `Generator` protocol reaches the exact
same generation output the existing bulk `build_generated_deltas` path already validates.

**Files:**
- Create: `src/fmharness/models/stack_generator.py`
- Test: `tests/test_stack_generator.py`

**Interfaces:**
- Consumes: `Generator` protocol (`fmharness.model_protocols`), `PerturbationNotInContext`
  (same module), `ModelMetadata` (`fmharness.schema`).
- Produces: `PregeneratedStackGenerator` class. Used by Task 8 and Task 9.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stack_generator.py`:

```python
"""Tests for the Generator-protocol wrapper over Stack's pre-generated output."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from fmharness.model_protocols import Generator, PerturbationNotInContext
from fmharness.models.stack_generator import PregeneratedStackGenerator


def _write_adata(path: Path, x: list[list[float]], obs: list[str], var: list[str]) -> None:
    a = ad.AnnData(X=np.asarray(x, dtype=np.float32))
    a.obs_names = obs
    a.var_names = var
    a.write_h5ad(path)


def test_satisfies_generator_protocol(tmp_path: Path) -> None:
    gen = PregeneratedStackGenerator(tmp_path, {}, checkpoint_label="test")
    assert isinstance(gen, Generator)


def test_context_coverage_matches_the_declared_pert_map(tmp_path: Path) -> None:
    gen = PregeneratedStackGenerator(
        tmp_path, {"BRD-1": "D1", "BRD-2": "D2"}, checkpoint_label="test"
    )
    assert gen.context_coverage(["D1", "D2", "D3"]) == {"D1", "D2"}


def test_generate_reads_the_matching_pregenerated_file(tmp_path: Path) -> None:
    _write_adata(tmp_path / "BRD-1.h5ad", [[1.0, 2.0]], ["o1"], ["A", "B"])
    gen = PregeneratedStackGenerator(tmp_path, {"BRD-1": "D1"}, checkpoint_label="test")
    out = gen.generate(ad.AnnData(X=np.zeros((1, 2), dtype=np.float32)), "D1")
    assert list(out.obs_names) == ["o1"]
    assert np.allclose(out.X, [[1.0, 2.0]])


def test_generate_handles_space_sanitized_filenames(tmp_path: Path) -> None:
    # stack-generation sanitizes spaces in the split name to underscores when writing.
    _write_adata(tmp_path / "Retinoic_acid.h5ad", [[5.0]], ["o1"], ["A"])
    gen = PregeneratedStackGenerator(
        tmp_path, {"Retinoic acid": "D1"}, checkpoint_label="test"
    )
    out = gen.generate(ad.AnnData(X=np.zeros((1, 1), dtype=np.float32)), "D1")
    assert np.allclose(out.X, [[5.0]])


def test_generate_raises_on_a_drug_with_no_pert_map_entry(tmp_path: Path) -> None:
    gen = PregeneratedStackGenerator(tmp_path, {}, checkpoint_label="test")
    with pytest.raises(PerturbationNotInContext):
        gen.generate(ad.AnnData(X=np.zeros((1, 1), dtype=np.float32)), "unknown_drug")


def test_generate_raises_when_the_file_is_missing(tmp_path: Path) -> None:
    gen = PregeneratedStackGenerator(tmp_path, {"BRD-1": "D1"}, checkpoint_label="test")
    with pytest.raises(PerturbationNotInContext):
        gen.generate(ad.AnnData(X=np.zeros((1, 1), dtype=np.float32)), "D1")


def test_version_includes_the_checkpoint_label() -> None:
    gen = PregeneratedStackGenerator(Path("."), {}, checkpoint_label="drug-aligned")
    assert "drug-aligned" in gen.version()


def test_metadata_defaults_to_no_declared_leakage_corpus() -> None:
    gen = PregeneratedStackGenerator(Path("."), {}, checkpoint_label="test")
    assert gen.pretraining_lines() is None
    assert gen.pretraining_drugs() is None


def test_metadata_reports_a_declared_leakage_corpus_when_given() -> None:
    gen = PregeneratedStackGenerator(
        Path("."),
        {},
        checkpoint_label="drug-aligned",
        pretraining_lines={"ACH-000681"},
        pretraining_drugs={"Trametinib"},
        task_signal_in_pretrain="adjacent",
    )
    assert gen.pretraining_lines() == {"ACH-000681"}
    assert gen.pretraining_drugs() == {"Trametinib"}
    assert gen.metadata().task_signal_in_pretrain == "adjacent"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stack_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmharness.models.stack_generator'`

- [ ] **Step 3: Write the implementation**

Create `src/fmharness/models/stack_generator.py`:

```python
"""``Generator``-protocol wrapper over Stack's already-generated output.

Stack generation runs on Alpine GPU (``scripts/alpine/04_stack_generate.sbatch``), writing
``<pert_id>.h5ad`` files under an output directory -- this class does not run inference; it
resolves a requested perturbation to its pre-generated file using the same rule
``fmharness.deltas.build_generated_deltas``'s file-matching already implements (filename
stem, then the same stem with spaces sanitized to underscores, matching how stack-generation
writes output), so a driver using only the ``Generator`` protocol reaches the identical
generation output the existing bulk-scoring path already validates.

Optionally ``LeakageQueryable`` (``pretraining_lines``/``pretraining_drugs``): both default
to ``None`` (no declared corpus, ``filter_leakage`` reports ``basis="unknown"``) so a
checkpoint whose overlap with the eval cohort has not been measured is never silently
assumed clean; pass real sets once a measurement exists (see Task 9).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import anndata as ad

from fmharness.model_protocols import PerturbationNotInContext
from fmharness.schema import ModelMetadata, TaskSignal


class PregeneratedStackGenerator:
    def __init__(
        self,
        generated_dir: Path,
        pert_to_drug: dict[str, str],
        *,
        checkpoint_label: str,
        pretraining_lines: set[str] | None = None,
        pretraining_drugs: set[str] | None = None,
        task_signal_in_pretrain: TaskSignal = "none",
    ) -> None:
        self.generated_dir = Path(generated_dir)
        self.pert_to_drug = pert_to_drug
        self.drug_to_pert = {v: k for k, v in pert_to_drug.items()}
        self.checkpoint_label = checkpoint_label
        self._pretraining_lines = pretraining_lines
        self._pretraining_drugs = pretraining_drugs
        self._task_signal_in_pretrain: TaskSignal = task_signal_in_pretrain

    def version(self) -> str:
        return f"stack-generated@{self.checkpoint_label}"

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            pretraining_corpus=f"pregenerated:{self.checkpoint_label}",
            pretraining_cutoff_date=date(1970, 1, 1),
            task_signal_in_pretrain=self._task_signal_in_pretrain,
            expected_input="raw_counts",
        )

    def context_coverage(self, perturbations: object) -> set[str]:
        return {p for p in perturbations if p in self.drug_to_pert}  # type: ignore[attr-defined]

    def generate(self, baseline: ad.AnnData, perturbation: str) -> ad.AnnData:
        pert_id = self.drug_to_pert.get(perturbation)
        if pert_id is None:
            raise PerturbationNotInContext(
                f"{perturbation!r} has no pre-generated file (checkpoint "
                f"{self.checkpoint_label!r}) -- not in the declared pert_to_drug map"
            )
        path = self._resolve_file(pert_id)
        if path is None:
            raise PerturbationNotInContext(
                f"{perturbation!r} (pert_id {pert_id!r}) is declared but no matching file "
                f"exists under {self.generated_dir}"
            )
        return ad.read_h5ad(path)

    def _resolve_file(self, pert_id: str) -> Path | None:
        direct = self.generated_dir / f"{pert_id}.h5ad"
        if direct.exists():
            return direct
        sanitized = self.generated_dir / f"{pert_id.replace(' ', '_')}.h5ad"
        if sanitized.exists():
            return sanitized
        return None

    def pretraining_lines(self) -> set[str] | None:
        return self._pretraining_lines

    def pretraining_drugs(self) -> set[str] | None:
        return self._pretraining_drugs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stack_generator.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v`
Expected: all pass, no regressions.

- [ ] **Step 6: Lint and typecheck**

Run: `uv run ruff check src/fmharness/models/stack_generator.py tests/test_stack_generator.py && uv run pyright src tests` (project-wide, not just the new files -- a prior task's reviewer
caught a strict-mode failure a single-file check missed).
Expected: no errors (same one pre-existing exception).

- [ ] **Step 7: Commit**

```bash
git add src/fmharness/models/stack_generator.py tests/test_stack_generator.py
git commit -m "feat: add PregeneratedStackGenerator, a real Generator over Stack's output"
```

---

### Task 8: Check-1 registry-driven driver script

Reproduces `docs/tahoe_generation_results.md`'s Check-1 table through the harness-core
registries: `filter_leakage` against a `PregeneratedStackGenerator` (real `LeakageQueryable`
declaration when given one, `basis="unknown"` otherwise -- honest either way), the same
`loo_baseline_source`/`delta_fidelity` functions `score_generation_eval.py` already uses
(now shared, Task 6), and `build_generated_deltas` for the Stack row's bulk delta (the
`Generator` wrapper proves protocol conformance; scoring still uses the existing, already-
validated bulk path rather than looping `generate()` once per perturbation, since
`delta_fidelity` operates on the whole delta/key frame at once).

TDD'd here against small synthetic fixtures (no local Tahoe data exists in this worktree --
see Task 9 for the real-data run once Alpine produces results).

**Files:**
- Create: `scripts/check1_registry_driver.py`
- Test: `tests/test_check1_registry_driver.py`

**Interfaces:**
- Consumes: `loo_baseline_source`, `learned_gene_panel`, `build_generated_deltas`,
  `build_tahoe_deltas` (all `fmharness.deltas`, existing/Task 6), `delta_fidelity`
  (`fmharness.evaluation`, existing), `filter_leakage` (`fmharness.leakage`, existing),
  `PregeneratedStackGenerator` (Task 7).
- Produces: `run_check1(real_delta, real_key, base, *, query_baseline, generated_dir,
  pert_to_drug, checkpoint_label, hallmark_path, n_hvg=2000, k=10, pretraining_lines=None,
  pretraining_drugs=None, task_signal_in_pretrain="none") -> pd.DataFrame` (importable by
  Task 9's real-data invocation and by the test file; the script's `main()` is a thin CLI
  wrapper around it, matching this project's established script/library split).

- [ ] **Step 1: Write the failing test**

Create `tests/test_check1_registry_driver.py`:

```python
"""Tests for the registry-driven Check-1 driver, against small synthetic fixtures."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from check1_registry_driver import run_check1


def _write_adata(path: Path, x: list[list[float]], obs: list[str], var: list[str]) -> None:
    a = ad.AnnData(X=np.asarray(x, dtype=np.float32))
    a.obs_names = obs
    a.var_names = var
    a.write_h5ad(path)


def _hallmark_gmt(tmp_path: Path) -> Path:
    gmt = tmp_path / "hallmark.gmt"
    gmt.write_text("HALLMARK_TEST\thttp://example\tA\n")
    return gmt


def _fixture(
    tmp_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, dict[str, str]]:
    # 4 lines, 1 drug each, 3 genes -- big enough for a 1-component PCA/NMF reduction with
    # 3 training lines per leave-one-out fold.
    genes = ["A", "B", "C"]
    lines = ["L1", "L2", "L3", "L4"]
    rng = np.random.default_rng(0)
    real_delta = pd.DataFrame(rng.standard_normal((4, 3)) + 5.0, columns=pd.Index(genes))
    real_key = pd.DataFrame({"patient": lines, "drug": ["d1"] * 4})
    base = pd.DataFrame(rng.standard_normal((4, 3)) + 10.0, columns=pd.Index(genes), index=pd.Index(lines))

    # query_baseline is what build_generated_deltas reads from disk (the real driver's
    # --query-baseline path) -- same values as `base` here, since the synthetic fixture only
    # needs the wiring to be correct, not a realistic raw-counts-vs-CPM distinction.
    query_baseline = tmp_path / "query_baseline.h5ad"
    _write_adata(query_baseline, base.to_numpy().tolist(), lines, genes)

    gdir = tmp_path / "generated"
    gdir.mkdir()
    generated_vals = real_delta.to_numpy() + base.to_numpy() + 0.1  # close to real + baseline
    _write_adata(gdir / "BRD-1.h5ad", generated_vals.tolist(), lines, genes)
    pert_to_drug = {"BRD-1": "d1"}
    return real_delta, real_key, base, query_baseline, gdir, pert_to_drug


def test_run_check1_reports_one_row_per_source_including_stack(tmp_path: Path) -> None:
    real_delta, real_key, base, query_baseline, gdir, pert_to_drug = _fixture(tmp_path)
    table = run_check1(
        real_delta,
        real_key,
        base,
        query_baseline=query_baseline,
        generated_dir=gdir,
        pert_to_drug=pert_to_drug,
        checkpoint_label="test-checkpoint",
        n_hvg=3,
        k=1,
        hallmark_path=_hallmark_gmt(tmp_path),
    )
    assert set(table["source"]) == {"additive", "knn", "pca", "nmf", "stack"}
    assert {"r", "r_offdiag", "rank", "n_pairs", "n_genes"} <= set(table.columns)


def test_run_check1_stack_row_uses_the_generated_files(tmp_path: Path) -> None:
    # A stack row with a near-perfect predicted delta (generated - query_baseline ~=
    # real_delta) must score a high r -- proves the driver actually reads the written
    # generated file through build_generated_deltas, not a stub.
    real_delta, real_key, base, query_baseline, gdir, pert_to_drug = _fixture(tmp_path)
    table = run_check1(
        real_delta,
        real_key,
        base,
        query_baseline=query_baseline,
        generated_dir=gdir,
        pert_to_drug=pert_to_drug,
        checkpoint_label="test-checkpoint",
        n_hvg=3,
        k=1,
        hallmark_path=_hallmark_gmt(tmp_path),
    )
    stack_row = table[table["source"] == "stack"].iloc[0]
    assert stack_row["r"] > 0.9


def test_run_check1_applies_leakage_filtering_when_a_corpus_is_declared(tmp_path: Path) -> None:
    real_delta, real_key, base, query_baseline, gdir, pert_to_drug = _fixture(tmp_path)
    table = run_check1(
        real_delta,
        real_key,
        base,
        query_baseline=query_baseline,
        generated_dir=gdir,
        pert_to_drug=pert_to_drug,
        checkpoint_label="test-checkpoint",
        n_hvg=3,
        k=1,
        hallmark_path=_hallmark_gmt(tmp_path),
        pretraining_lines={"L1"},
        pretraining_drugs={"d1"},
        task_signal_in_pretrain="adjacent",
    )
    # L1 x d1 is doubly-exposed -- every row (scored on the filtered real_key) must show 3
    # pairs, not 4, since filtering happens before any source is built.
    for _, row in table.iterrows():
        assert row["n_pairs"] == 3


def test_run_check1_reports_no_leakage_filtering_without_a_declared_corpus(tmp_path: Path) -> None:
    real_delta, real_key, base, query_baseline, gdir, pert_to_drug = _fixture(tmp_path)
    table = run_check1(
        real_delta,
        real_key,
        base,
        query_baseline=query_baseline,
        generated_dir=gdir,
        pert_to_drug=pert_to_drug,
        checkpoint_label="test-checkpoint",
        n_hvg=3,
        k=1,
        hallmark_path=_hallmark_gmt(tmp_path),
    )
    for _, row in table.iterrows():
        assert row["n_pairs"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_check1_registry_driver.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'check1_registry_driver'`

- [ ] **Step 3: Write the implementation**

Create `scripts/check1_registry_driver.py`:

```python
"""Check 1 (delta-Pearson generation quality) through the harness-core registries.

Reproduces docs/tahoe_generation_results.md's Check-1 table via the same underlying
functions scripts/score_generation_eval.py uses (loo_baseline_source / build_generated_deltas
/ delta_fidelity, all shared from fmharness.deltas as of the plan that added this script).
The Stack row's SCORING reuses build_generated_deltas directly, unchanged -- it is already
correct and already vectorized; there is no reason to re-derive it. What is new is that the
Stack checkpoint's LeakageQueryable declaration (fmharness.models.stack_generator's
PregeneratedStackGenerator) drives filter_leakage (fmharness.leakage) before any source is
built, so a checkpoint's measured pretraining overlap with the eval cohort actually strips
contaminated rows -- the composition this whole harness-core effort exists for -- rather than
leakage-filtering staying a dormant, uncalled function. Baselines (additive/knn/pca/nmf) are
not the swappable dimension this proves -- they stay exactly as score_generation_eval.py
already builds them.

Run (once Alpine has produced the inputs -- see the implementation plan's Task 9 for the
exact `ralpine pull` commands and real-data invocation):
  uv run python scripts/check1_registry_driver.py \\
      --context tahoe_context.h5ad \\
      --query-baseline tahoe_query.h5ad \\
      --generated-dir generated \\
      --pert-map context_by_drug/pert_to_cid.tsv \\
      --checkpoint-label cytokine-aligned
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd

from fmharness.deltas import build_generated_deltas, build_tahoe_deltas, learned_gene_panel, loo_baseline_source
from fmharness.evaluation import delta_fidelity
from fmharness.leakage import filter_leakage
from fmharness.models.stack_generator import PregeneratedStackGenerator
from fmharness.schema import TaskSignal


def _load_pert_map(path: Path) -> dict[str, str]:
    m: dict[str, str] = {}
    for line in path.read_text().splitlines():
        pert, _, cid = line.partition("\t")
        if pert.strip() and cid.strip():
            m[pert.strip()] = cid.strip()
    return m


def run_check1(
    real_delta: pd.DataFrame,
    real_key: pd.DataFrame,
    base: pd.DataFrame,
    *,
    query_baseline: Path,
    generated_dir: Path,
    pert_to_drug: dict[str, str],
    checkpoint_label: str,
    hallmark_path: Path,
    n_hvg: int = 2000,
    k: int = 10,
    pretraining_lines: set[str] | None = None,
    pretraining_drugs: set[str] | None = None,
    task_signal_in_pretrain: TaskSignal = "none",
) -> pd.DataFrame:
    """Check-1 table: one row per delta source, including the Stack generator.

    ``real_delta``/``real_key``/``base`` are the ground-truth triple from
    ``build_tahoe_deltas`` (or the parquet-bundle equivalent). ``query_baseline`` is the
    AnnData path fed to Stack generation as ``--test-adata`` -- what ``build_generated_deltas``
    needs to compute ``generated - baseline``; it is a different representation from ``base``
    (CPM-normalized query file vs. raw pseudobulk), matching score_generation_eval.py's own
    ``--context``/``--query-baseline`` split. ``pretraining_lines``/``pretraining_drugs``
    (both default ``None``) declare the Stack checkpoint's measured pretraining overlap with
    this eval cohort, if known -- when given, ``filter_leakage`` drops the doubly-exposed
    (line, drug) pairs from every source before scoring, matching the tiered rule the rest of
    the harness already applies. When not given, the design is scored unfiltered.
    """
    model = PregeneratedStackGenerator(
        generated_dir,
        pert_to_drug,
        checkpoint_label=checkpoint_label,
        pretraining_lines=pretraining_lines,
        pretraining_drugs=pretraining_drugs,
        task_signal_in_pretrain=task_signal_in_pretrain,
    )
    filtered_key, _profile = filter_leakage(real_key, model)
    keep = filtered_key.index
    fd = real_delta.loc[keep].reset_index(drop=True)
    fk = filtered_key.reset_index(drop=True)

    learned_genes = learned_gene_panel(fd, hallmark_path, n_hvg=n_hvg)
    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "additive": loo_baseline_source("additive", fd, fk, base, k=k),
        "knn": loo_baseline_source("knn", fd, fk, base, k=k),
        "pca": loo_baseline_source("pca", fd, fk, base, k=k, genes=learned_genes),
        "nmf": loo_baseline_source("nmf", fd, fk, base, k=k, genes=learned_genes),
        "stack": build_generated_deltas(generated_dir, query_baseline, pert_to_drug),
    }

    rows: list[dict[str, object]] = []
    for name, (d, kk) in sources.items():
        # delta_fidelity inner-joins pred_key/real_key on (patient, drug) itself (evaluation.py's
        # own pk.merge(rk, on=["patient","drug"], how="inner")) -- the stack source's key (built
        # from the full generated directory, independent of the leakage filter above) is
        # automatically restricted to fk's already-filtered pairs by that join; no separate
        # pre-filter is needed here, and adding one would just duplicate delta_fidelity's own
        # contract.
        f = delta_fidelity(d, kk, fd, fk, n_hvg=n_hvg)
        rows.append(
            {
                "source": name,
                "r": round(float(f["r"].mean()), 3),
                "r_offdiag": round(float(f["r_offdiag"].mean()), 3),
                "rank": round(float(f["rank"].mean()), 3),
                "n_pairs": len(f),
                "n_genes": int(f["n_genes"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context", required=True, help="Tahoe context AnnData (build_tahoe_context)")
    ap.add_argument("--query-baseline", required=True, help="AnnData fed to stack-generation as --test-adata")
    ap.add_argument("--generated-dir", required=True, help="dir of Stack-generated <pert_id>.h5ad files")
    ap.add_argument("--pert-map", required=True, help="TSV 'pert_id<TAB>cid' (context split writes this)")
    ap.add_argument("--checkpoint-label", required=True, help="e.g. cytokine-aligned or drug-aligned")
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--hallmark-path", default="data/static/hallmark_signatures.gmt", help="Hallmark .gmt path"
    )
    ap.add_argument("--corpus-lines", default=None, help="comma-separated declared pretraining lines")
    ap.add_argument("--corpus-drugs", default=None, help="comma-separated declared pretraining drugs")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    real_delta, real_key, base = build_tahoe_deltas(ad.read_h5ad(args.context))
    pert_to_drug = _load_pert_map(Path(args.pert_map))
    table = run_check1(
        real_delta,
        real_key,
        base,
        query_baseline=Path(args.query_baseline),
        generated_dir=Path(args.generated_dir),
        pert_to_drug=pert_to_drug,
        checkpoint_label=args.checkpoint_label,
        hallmark_path=repo / args.hallmark_path,
        n_hvg=args.n_hvg,
        k=args.k,
        pretraining_lines=set(args.corpus_lines.split(",")) if args.corpus_lines else None,
        pretraining_drugs=set(args.corpus_drugs.split(",")) if args.corpus_drugs else None,
        task_signal_in_pretrain="adjacent" if args.corpus_lines else "none",
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_check1_registry_driver.py -v`
Expected: PASS (4 tests). If `test_run_check1_stack_row_uses_the_generated_files`'s `r > 0.9`
threshold is flaky, tighten the fixture's `generated_vals` construction (currently
`real_delta + base + 0.1`, a small, fixed offset) rather than loosening the threshold -- the
point is that the driver reads real generated data, not that it barely beats chance.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v`
Expected: all pass, no regressions.

- [ ] **Step 6: Lint and typecheck**

Run: `uv run ruff check scripts/check1_registry_driver.py tests/test_check1_registry_driver.py && uv run pyright src tests`
Expected: no errors on the new files (`scripts/` is outside pyright's strict scope, but ruff
must be clean on it); same one pre-existing pyright exception elsewhere.

- [ ] **Step 7: Commit**

```bash
git add scripts/check1_registry_driver.py tests/test_check1_registry_driver.py
git commit -m "feat: add the registry-driven Check-1 driver (Generator + filter_leakage + delta_fidelity)"
```

---

### Task 9: Real-data verification and the drug-aligned row

Gated on two things landing first: Phase 1's Alpine jobs (submitted by Lucas, per Task 5's
closing note) producing a drug-aligned checkpoint and its generation output, and this
worktree having zero local Tahoe/Stack data today (confirmed: no `tahoe_query.h5ad`, no
`context_by_drug/`, no `generated/` directory, no `results/` directory exist anywhere in this
repo -- everything currently lives only on Alpine scratch). This task documents the exact
commands; it cannot be "done" until both land, but every command in it is real, not a
placeholder.

**Files:** none (execution-only task; no new source files).

- [ ] **Step 1: Pull the existing (cytokine-aligned) inputs from Alpine**

Once an open `ssh alpine` ControlMaster socket exists this session (`ralpine status` reports
it), pull the data the *existing*, already-published Check-1 table was computed from:

```bash
scripts/alpine/ralpine pull "$REMOTE_ROOT/tahoe_query.h5ad" ./tahoe_query.h5ad
scripts/alpine/ralpine pull "$REMOTE_ROOT/tahoe_context.h5ad" ./tahoe_context.h5ad
scripts/alpine/ralpine pull "$REMOTE_ROOT/context_by_drug" ./context_by_drug
scripts/alpine/ralpine pull "$REMOTE_ROOT/generated" ./generated
```

(`$REMOTE_ROOT` defaults to `/projects/lgillenwater@xsede.org/repositories/fm-pdo-evaluator`
per `scripts/alpine/ralpine`'s own config -- confirm the exact remote paths with `ralpine ls
"$REMOTE_ROOT"` first if any of these differ from what a real prior run actually wrote.)

- [ ] **Step 2: Reproduce the published cytokine-aligned Check-1 table**

```bash
uv run python scripts/check1_registry_driver.py \
    --context tahoe_context.h5ad \
    --query-baseline tahoe_query.h5ad \
    --generated-dir generated \
    --pert-map context_by_drug/pert_to_cid.tsv \
    --checkpoint-label cytokine-aligned
```

Expected, matching `docs/tahoe_generation_results.md`'s Check-1 table: `additive` r=0.225,
`nmf` r=0.221, `pca` r=0.207, `knn` r=0.178, `stack` r=0.012 -- or every difference explained
and justified in writing (this task's acceptance bar, inherited from the original Arm-2
spec's Phase-2 acceptance criterion). A difference here means investigating before trusting
anything from Step 4 -- the reproduction check exists precisely to catch a driver bug before
it's mistaken for a real finding.

- [ ] **Step 3: Once Alpine has produced the drug-aligned checkpoint's generation output, re-measure its real leakage overlap**

The existing measured overlap (A549 line, 6-7 drugs, 6 doubly-exposed pairs = 0.4%) was
computed against the *chemCPA* sci-Plex source; Task 1 switched to a different (scPerturb)
source, so this must be re-derived, not assumed to carry over. Once `context_by_drug/manifest.tsv`
(Tahoe's 33 drugs) and the new `sciplex_finetune.h5ad`'s perturbation categories are both
available locally (pull the sci-Plex file's obs via a small `ad.read_h5ad(path,
backed="r").obs` read, not the full 2.5 GB matrix), compute:

```python
import anndata as ad
import pandas as pd

tahoe_query = ad.read_h5ad("tahoe_query.h5ad", backed="r")
tahoe_lines = set(tahoe_query.obs_names.astype(str))

pert_to_cid = pd.read_csv("context_by_drug/pert_to_cid.tsv", sep="\t", header=None, names=["pert_id", "cid"])
tahoe_drugs = set(pert_to_cid["pert_id"])

sciplex = ad.read_h5ad("sciplex_finetune.h5ad", backed="r")
sciplex_lines = set(sciplex.obs["cell_line"].astype(str).unique())
sciplex_drugs = set(sciplex.obs["pert_id"].astype(str).unique()) - {"control"}

line_overlap = tahoe_lines & sciplex_lines
drug_overlap = tahoe_drugs & sciplex_drugs  # exact-string match only; check for name variants by hand
print(f"line overlap: {len(line_overlap)} of {len(tahoe_lines)} -- {sorted(line_overlap)}")
print(f"drug overlap: {len(drug_overlap)} of {len(tahoe_drugs)} -- {sorted(drug_overlap)}")
```

Check for name-variant matches by hand the same way the original measurement did (e.g.
`Fluorouracil` vs `5-Fluorouracil`) -- exact-string overlap alone will undercount them.

- [ ] **Step 4: Report the drug-aligned row**

```bash
uv run python scripts/check1_registry_driver.py \
    --context tahoe_context.h5ad \
    --query-baseline tahoe_query.h5ad \
    --generated-dir generated_sciplex \
    --pert-map context_by_drug/pert_to_cid.tsv \
    --checkpoint-label drug-aligned \
    --corpus-lines <line_overlap, comma-separated, from Step 3> \
    --corpus-drugs <drug_overlap, comma-separated, from Step 3>
```

Run once with `--corpus-lines`/`--corpus-drugs` (leaked pairs excluded) and once without
(full cohort) -- report both, per the design spec's "exclude the 6 doubly-exposed pairs from
Check 1, or report with and without."

**Acceptance:** a table with the drug-aligned Stack row's delta-Pearson `r` reported next to
cytokine-aligned (reproduced in Step 2), `additive` (0.225), and the ceiling (0.30 raw / 0.46
Spearman-Brown, from `docs/tahoe_generation_results.md`), with and without the leaked pairs.

---

## Plan exit criteria

- Tasks 1-8's tests all pass; `uv run pytest -v` is green for the full suite.
- `uv run ruff check src tests` and `uv run pyright src tests` are both clean (the one
  pre-existing, unrelated `src/fmharness/deltas.py:194` error is not this plan's to fix).
- `PregeneratedStackGenerator` is proven via `isinstance` against the real `Generator`
  protocol.
- Task 9 is documented with real, runnable commands even though its acceptance cannot be
  verified until Phase 1's Alpine jobs (submitted by Lucas) complete -- it is not blocking
  Tasks 1-8 being marked done.

## Not in this plan

- Check 2 (end-to-end GDSC2 AUC ladder, selection gap@k, MOA stratification) -- separate,
  future work per the design spec.
- Phases 3-6 of the original Arm-2 spec (positive controls/MDE, n~500 re-anchor, panel-scale
  Stack embeddings, full LOO/LODO CV).
- Alpine safety hardening -- explicitly paused earlier in this project, not reopened here.
- Any change to the Soragni/Arm-1 cohort or PDTO application.
