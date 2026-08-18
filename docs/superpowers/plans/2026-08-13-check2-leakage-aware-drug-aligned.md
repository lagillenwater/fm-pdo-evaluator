# Leakage-Aware Check 2, Drug-Aligned Stack Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-aware, registry-driven Check-2 driver (`scripts/check2_registry_driver.py`), run it for the cytokine-aligned and drug-aligned Stack checkpoints, and write the results into `docs/tahoe_generation_results.md` and `docs/prospective_evaluation_harness_overview.pptx`.

**Architecture:** Extract Check 2's scoring composition (fixed-signature readouts + representation-controlled penalized grid) out of `scripts/score_generation_eval.py` into a new `src/fmharness/check2.py`, leakage-agnostic like `evaluation.score_delta_sources`. A new `scripts/check2_registry_driver.py` mirrors `scripts/check1_registry_driver.py`'s shape: build the `PregeneratedStackGenerator`, run `filter_leakage` -- pointed at the GDSC2 AUC `design` frame instead of Tahoe's `real_key`, since `design` is Check 2's actual evaluation set -- then call `fmharness.check2.score_check2`. Three small CLI-validation helpers currently private to `check1_registry_driver.py` move into `fmharness/leakage.py` so both drivers share one copy.

**Tech Stack:** Python 3.11, pandas/numpy/scipy/scikit-learn, anndata, pytest, uv.

## Global Constraints

- Line length 100 (`[tool.ruff]` in `pyproject.toml`).
- `target-version = "py311"`; pyright `typeCheckingMode = "strict"` over `src` and `tests`
  (`scripts/` is on pyright's `extraPaths` for import resolution but not in `include` --
  analyzed, not strict-checked -- matching how `check1_registry_driver.py` is treated today).
- Ruff lint selects `E, F, I, B, UP, SIM, RUF`. `tests/**` ignores `E501` only.
- Run everything through `uv`: `uv run pytest`, `uv run python`, `uv run ruff`, `uv run pyright`.
- No emojis anywhere -- code, comments, output, commit messages.
- Vectorized only. No nested Python loops over data rows. A loop over a small, fixed set
  (delta sources, penalties, folds, cell lines in a leave-one-out rebuild) is fine; a loop
  over samples or genes is not.
- New public functions/classes need docstrings: what it computes, and why that is the right
  design.
- `pyproject.toml`'s `[tool.pytest.ini_options]` sets `pythonpath = ["scripts"]` -- test files
  import driver scripts directly by module name (`from check2_registry_driver import
  run_check2`), no package installation needed.
- Commit after each task (worktree-local; this branch's standing arrangement continues). Lucas
  has a standing preference to run `git commit` himself in VS Code -- when this plan is
  executed, confirm with him whether Claude or Lucas runs each task's commit step; this plan
  writes the commit step into every task per the repo's own convention regardless.

---

### Task 1: Move Check-2's low-level scoring helpers into `fmharness/check2.py`

`scripts/score_generation_eval.py` currently defines `_make_penalty`, `_load_line_matrix`,
`_repr_by_drug`, `_penalized_preds`, and the `PROLIFERATION`/`FIXED_READOUTS`/`PENALTY_NAMES`
constants inline. Move them (public names, same behavior) into a new module so
`check2_registry_driver.py` (Task 4) can reuse them without duplicating ~90 lines -- the same
move Task 6 of `docs/superpowers/plans/2026-08-11-stack-drug-alignment-and-check1.md` already
made for `loo_baseline_source`/`learned_gene_panel`. This task moves the helpers only; Task 2
moves the composition that calls them.

**Files:**
- Create: `src/fmharness/check2.py`
- Modify: `scripts/score_generation_eval.py`
- Test: `tests/test_check2.py`

**Interfaces:**
- Produces: `PROLIFERATION: tuple[str, ...]`, `FIXED_READOUTS: tuple[str, ...]`,
  `PENALTY_NAMES: tuple[str, ...]`, `make_penalty(name: str) -> object`,
  `load_line_matrix(path: Path) -> pd.DataFrame`, `repr_by_drug(delta: pd.DataFrame, key:
  pd.DataFrame, genes: pd.Index) -> dict[str, pd.DataFrame]`, `penalized_preds(feat, design:
  pd.DataFrame, fold_of: dict[str, int], n_folds: int, uniq_lines: list[str], penalty: str, *,
  min_lines: int = 8, min_train: int = 5) -> pd.DataFrame`. Used by Task 2's `score_check2` and
  (transitively, via `score_check2`) Task 4's `check2_registry_driver.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_check2.py`:

```python
"""Tests for fmharness.check2 -- Check-2 scoring composition helpers."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from fmharness.check2 import load_line_matrix, make_penalty, penalized_preds, repr_by_drug


def test_make_penalty_returns_the_named_sklearn_model() -> None:
    assert isinstance(make_penalty("l2"), RidgeCV)
    assert isinstance(make_penalty("l1"), LassoCV)
    assert isinstance(make_penalty("en"), ElasticNetCV)


def test_make_penalty_raises_on_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown penalty"):
        make_penalty("bogus")


def test_repr_by_drug_splits_the_delta_into_one_frame_per_drug() -> None:
    delta = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    key = pd.DataFrame({"patient": ["L1", "L2", "L3"], "drug": ["d1", "d1", "d2"]})
    out = repr_by_drug(delta, key, pd.Index(["A", "B"]))
    assert set(out) == {"d1", "d2"}
    assert list(out["d1"].index) == ["L1", "L2"]
    assert list(out["d2"].index) == ["L3"]
    assert np.allclose(out["d1"].to_numpy(), [[1.0, 4.0], [2.0, 5.0]])


def test_repr_by_drug_fills_missing_genes_with_zero() -> None:
    delta = pd.DataFrame({"A": [1.0]})
    key = pd.DataFrame({"patient": ["L1"], "drug": ["d1"]})
    out = repr_by_drug(delta, key, pd.Index(["A", "B"]))
    assert np.allclose(out["d1"].to_numpy(), [[1.0, 0.0]])


def _write_adata(path: Path, x: list[list[float]], obs: list[str]) -> None:
    a = ad.AnnData(X=np.asarray(x, dtype=np.float32))
    a.obs_names = obs
    a.var_names = [f"g{i}" for i in range(len(x[0]))]
    a.write_h5ad(path)


def test_load_line_matrix_reads_h5ad(tmp_path: Path) -> None:
    path = tmp_path / "emb.h5ad"
    _write_adata(path, [[1.0, 2.0], [3.0, 4.0]], ["L1", "L2"])
    df = load_line_matrix(path)
    assert list(df.index) == ["L1", "L2"]
    assert np.allclose(df.to_numpy(), [[1.0, 2.0], [3.0, 4.0]])


def test_load_line_matrix_reads_csv(tmp_path: Path) -> None:
    path = tmp_path / "emb.csv"
    pd.DataFrame({"a": [1.0, 3.0], "b": [2.0, 4.0]}, index=["L1", "L2"]).to_csv(path)
    df = load_line_matrix(path)
    assert list(df.index) == ["L1", "L2"]
    assert np.allclose(df.to_numpy(), [[1.0, 2.0], [3.0, 4.0]])


def test_load_line_matrix_reads_parquet(tmp_path: Path) -> None:
    path = tmp_path / "emb.parquet"
    pd.DataFrame({"a": [1.0, 3.0], "b": [2.0, 4.0]}, index=["L1", "L2"]).to_parquet(path)
    df = load_line_matrix(path)
    assert list(df.index) == ["L1", "L2"]
    assert np.allclose(df.to_numpy(), [[1.0, 2.0], [3.0, 4.0]])


def test_penalized_preds_predicts_the_held_out_fold_per_drug() -> None:
    # 8 lines, 1 drug, a feature that equals the AUC exactly -- the fitted model must recover
    # held-out y_true closely, proving the fold-split + fit + predict wiring is correct, not
    # just that it runs.
    lines = [f"L{i}" for i in range(8)]
    y = {ln: float(i) for i, ln in enumerate(lines)}
    feat = {"d1": pd.DataFrame({"x": [y[ln] for ln in lines]}, index=pd.Index(lines))}
    design = pd.DataFrame({"patient": lines, "drug": ["d1"] * 8, "y": [y[ln] for ln in lines]})
    fold_of = {ln: i % 2 for i, ln in enumerate(lines)}
    preds = penalized_preds(feat, design, fold_of, 2, lines, "l2", min_lines=4, min_train=2)
    assert set(preds["patient"]) == set(lines)
    assert np.corrcoef(preds["y_true"], preds["y_pred"])[0, 1] > 0.9


def test_penalized_preds_skips_a_drug_below_min_lines() -> None:
    feat = {"d1": pd.DataFrame({"x": [1.0, 2.0]}, index=pd.Index(["L1", "L2"]))}
    design = pd.DataFrame({"patient": ["L1", "L2"], "drug": ["d1", "d1"], "y": [0.1, 0.2]})
    fold_of = {"L1": 0, "L2": 1}
    preds = penalized_preds(feat, design, fold_of, 2, ["L1", "L2"], "l2", min_lines=8)
    assert preds.empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_check2.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmharness.check2'`

- [ ] **Step 3: Write the implementation**

Create `src/fmharness/check2.py`:

```python
"""Check-2 scoring building blocks: penalty models, per-drug representation splitting, and
the leave-cell-line-out penalized-regression fit, plus ``score_check2``, the composition of
these into the fixed-signature-readout scoring and the representation-controlled penalized
grid.

Shared by ``scripts/score_generation_eval.py`` and ``scripts/check2_registry_driver.py``.
Stays leakage-agnostic -- ``score_check2`` scores whatever ``design`` frame it is handed, the
same way ``evaluation.score_delta_sources`` does for Check 1. ``filter_leakage`` is always the
caller's job, not this module's.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler

# The Hallmark proliferation sets -- the two that cleared the gate's random-gene-set control
# (G2M clearly, E2F marginally); the death sets (P53, apoptosis) add only noise on Tahoe. A
# ``proliferation`` readout scores just these, so a real but weak signal is not diluted away.
PROLIFERATION = ("HALLMARK_E2F_TARGETS", "HALLMARK_G2M_CHECKPOINT")
FIXED_READOUTS = ("hallmark", "proliferation")  # fixed-signature readouts, applied to delta sources
PENALTY_NAMES = ("l2", "l1", "en")  # penalized regressions for the representation-controlled grid


def make_penalty(name: str) -> object:
    """A fresh ALPHA-CV-TUNED penalized model: l2=RidgeCV (efficient GCV), l1=LassoCV, en=
    ElasticNetCV (both inner 3-fold on the training lines). Tuning the penalty per representation
    makes the grid model-fair -- a fixed alpha over-/under-regularizes some representations and
    flips the ranking (Kurilov 2020)."""
    if name == "l2":
        return RidgeCV(alphas=np.logspace(-2, 3, 12))
    if name == "l1":
        return LassoCV(n_alphas=30, cv=3, max_iter=20000, random_state=0)
    if name == "en":
        return ElasticNetCV(l1_ratio=0.5, n_alphas=30, cv=3, max_iter=20000, random_state=0)
    raise ValueError(f"unknown penalty {name!r}")


def load_line_matrix(path: Path) -> pd.DataFrame:
    """Load a per-cell-line feature matrix (index = line id) for the check-2 grid, from a
    ``.h5ad`` (X + obs_names), ``.parquet``, or ``.csv``. Used to fold a precomputed FM
    embedding (one vector per line) in head-to-head with expr/pca."""
    if path.suffix == ".h5ad":
        a = ad.read_h5ad(path)
        x = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        return pd.DataFrame(x, index=pd.Index([str(o) for o in a.obs_names])).astype(float)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, index_col=0)
    df.index = pd.Index([str(i) for i in df.index])
    return df.astype(float)


def repr_by_drug(delta: pd.DataFrame, key: pd.DataFrame, genes: pd.Index) -> dict[str, pd.DataFrame]:
    """Split a delta source into ``{drug: DataFrame[line x genes]}`` for per-drug regression."""
    d = delta.reindex(columns=genes).fillna(0.0)
    pat = key["patient"].astype(str).to_numpy()
    drg = key["drug"].astype(str).to_numpy()
    out: dict[str, pd.DataFrame] = {}
    for drug in pd.unique(drg):
        m = d[drg == drug]
        m.index = pd.Index(pat[drg == drug])
        out[str(drug)] = m
    return out


def penalized_preds(
    feat: dict[str, pd.DataFrame] | Callable[[str], pd.DataFrame],
    design: pd.DataFrame,
    fold_of: dict[str, int],
    n_folds: int,
    uniq_lines: list[str],
    penalty: str,
    *,
    min_lines: int = 8,
    min_train: int = 5,
) -> pd.DataFrame:
    """Per-drug penalized regression (representation -> AUC), leave-cell-line-out by fold.

    ``feat`` maps a drug to a (line x gene) frame -- a dict for a delta source, or a callable for a
    drug-independent representation (baseline expression). For each drug the model is fit on the
    training-fold lines' features vs AUC and predicts the held-fold lines; the StandardScaler is fit
    on the training lines only, so a single held line (true LOO) is scored leakage-free. All
    representations share one model class, so a difference is the representation, not the model.
    Returns preds (patient, drug, y_true, y_pred); y_pred is an AUC estimate (same sign as y_true).
    """
    auc_by_drug = {
        str(d): dict(zip(g["patient"].astype(str), g["y"], strict=False))
        for d, g in design.groupby("drug")
    }
    rows: list[tuple[str, str, float, float]] = []
    for drug, auc in auc_by_drug.items():
        fdf = feat(drug) if callable(feat) else feat.get(drug)  # type: ignore[union-attr]
        if fdf is None or fdf.empty:
            continue
        fdf = fdf.copy()
        fdf.index = pd.Index([str(i) for i in fdf.index])
        lines_d = [ln for ln in fdf.index if ln in auc]
        if len(lines_d) < min_lines:
            continue
        for f in range(n_folds):
            held = {ln for ln in uniq_lines if fold_of[ln] == f}
            tr = [ln for ln in lines_d if ln not in held]
            te = [ln for ln in lines_d if ln in held]
            if len(tr) < min_train or not te:
                continue
            sc = StandardScaler().fit(fdf.loc[tr].to_numpy(dtype=np.float64))
            # make_penalty returns `object` (RidgeCV/LassoCV/ElasticNetCV share no common
            # base class pyright can see .fit/.predict on) -- both calls need an ignore.
            model = make_penalty(penalty).fit(  # type: ignore[attr-defined]
                sc.transform(fdf.loc[tr].to_numpy(dtype=np.float64)), [auc[ln] for ln in tr]
            )
            te_x = sc.transform(fdf.loc[te].to_numpy(dtype=np.float64))
            pred = model.predict(te_x)  # type: ignore[attr-defined]
            rows.extend(
                (ln, drug, float(auc[ln]), float(p)) for ln, p in zip(te, pred, strict=False)
            )
    cols = ["patient", "drug", "y_true", "y_pred"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_check2.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Wire `score_generation_eval.py` to import from `fmharness.check2`**

In `scripts/score_generation_eval.py`:

Delete the `PROLIFERATION`/`FIXED_READOUTS`/`PENALTY_NAMES` constant block and the
`_make_penalty`, `_load_line_matrix`, `_repr_by_drug`, `_penalized_preds` function definitions
(everything between the `from fmharness.signatures import ...` import line and the `def _rel(repo:
Path, p: str) -> Path:` definition -- keep `_rel`, it is a generic path helper, not check2-specific).

Replace the deleted block with an import, added to the existing `from fmharness...` import group
(alphabetical by module):

```python
from fmharness.check2 import (
    FIXED_READOUTS,
    PENALTY_NAMES,
    PROLIFERATION,
    load_line_matrix,
    penalized_preds,
    repr_by_drug,
)
```

(`make_penalty` is not imported here -- this file never calls it directly, only the moved
`penalized_preds` does, internally within `fmharness.check2`; importing it anyway would be an
unused import, which `ruff`'s `F401` flags.) Rename the three remaining call sites in `main()`:
- `emb = _load_line_matrix(_rel(repo, p.strip()))` -> `emb = load_line_matrix(_rel(repo, p.strip()))`
- `representations[name] = _repr_by_drug(d, kk, hvg)` -> `representations[name] = repr_by_drug(d, kk, hvg)`
- `preds = _penalized_preds(feat, design_target, fold_of, n_folds, uniq_lines, pen)` ->
  `preds = penalized_preds(feat, design_target, fold_of, n_folds, uniq_lines, pen)`

- [ ] **Step 6: Verify the script still imports cleanly**

Run: `uv run python -c "import ast; ast.parse(open('scripts/score_generation_eval.py').read())"`
Expected: no error. Do not execute `scripts/score_generation_eval.py` itself -- it needs real
Tahoe data this worktree's test environment does not exercise in CI.

- [ ] **Step 7: Run the full test suite**

Run: `uv run pytest -v`
Expected: all pass, no regressions.

- [ ] **Step 8: Lint and typecheck**

Run: `uv run ruff check src/fmharness/check2.py scripts/score_generation_eval.py tests/test_check2.py && uv run pyright src tests`
Expected: no errors (pyright: only the one pre-existing, unrelated `src/fmharness/deltas.py:194`
error).

- [ ] **Step 9: Commit**

```bash
git add src/fmharness/check2.py scripts/score_generation_eval.py tests/test_check2.py
git commit -m "refactor: move Check-2's penalty/representation helpers into fmharness.check2"
```

---

### Task 2: Add `score_check2`, the leakage-agnostic Check-2 scoring composition

Compose Task 1's helpers into the fixed-signature-readout loop and representation-grid loop
currently inline in `score_generation_eval.py`'s `main()`. `score_check2` takes an already-built
`design` frame and does not know `filter_leakage` exists -- exactly like
`evaluation.score_delta_sources` for Check 1 -- so `check2_registry_driver.py` (Task 4) can hand
it a leakage-filtered `design` and `score_generation_eval.py` can hand it an unfiltered one, with
identical downstream code.

**Files:**
- Modify: `src/fmharness/check2.py`
- Modify: `scripts/score_generation_eval.py`
- Test: `tests/test_check2.py`

**Interfaces:**
- Consumes: `make_penalty`, `repr_by_drug`, `penalized_preds` (Task 1, same module);
  `build_adapters` (`fmharness.adapters`, existing); `score_predictions`
  (`fmharness.evaluation`, existing).
- Produces: `score_check2(sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]], real_key:
  pd.DataFrame, base: pd.DataFrame, hvg: pd.Index, design: pd.DataFrame, *, hallmark: dict[str,
  tuple[list[str], str]], fixed_methods: tuple[str, ...] = FIXED_READOUTS, penalties:
  tuple[str, ...] = PENALTY_NAMES, folds: int = 5, stack_emb: dict[str, pd.DataFrame] | None =
  None, n_permutations: int = 1000) -> pd.DataFrame`. Used by Task 4's `run_check2`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_check2.py` (extend the top-level import to add `score_check2`):

```python
from fmharness.check2 import load_line_matrix, make_penalty, penalized_preds, repr_by_drug, score_check2


def _check2_fixture() -> tuple[
    dict[str, tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame, pd.DataFrame, pd.Index, pd.DataFrame
]:
    # 10 lines, 1 drug ("d1"), 2 genes -- 10 lines so the representation grid's default
    # min_lines=8/min_train=5 are both satisfiable with folds=2 (5 train / 5 test per fold).
    genes = pd.Index(["A", "B"])
    lines = [f"L{i}" for i in range(10)]
    rng = np.random.default_rng(0)
    delta = pd.DataFrame(rng.standard_normal((10, 2)) + 1.0, columns=genes)
    key = pd.DataFrame({"patient": lines, "drug": ["d1"] * 10})
    base = pd.DataFrame(
        rng.standard_normal((10, 2)) + 5.0, columns=genes, index=pd.Index(lines)
    )
    sources = {"additive": (delta, key)}
    hallmark = {"HALLMARK_TEST": (["A"], "http://example")}
    design = pd.DataFrame(
        {"patient": lines, "drug": ["d1"] * 10, "y": rng.standard_normal(10).tolist()}
    )
    return sources, key, base, genes, design, hallmark  # type: ignore[return-value]


def test_score_check2_reports_fixed_readout_and_representation_grid_rows() -> None:
    sources, real_key, base, hvg, design, hallmark = _check2_fixture()
    table = score_check2(
        sources, real_key, base, hvg, design, hallmark=hallmark, folds=2,
    )
    assert not table.empty
    assert {"source", "method", "global", "interaction", "perdrug", "p_label", "n"} <= set(
        table.columns
    )
    # fixed-signature readout rows (method in hallmark/proliferation) score the "additive"
    # delta source; representation-grid rows (method in l2/l1/en) additionally include "expr".
    assert {"hallmark", "proliferation"} <= set(table["method"])
    assert {"l2", "l1", "en"} <= set(table["method"])
    assert "expr" in set(table["source"])


def test_score_check2_includes_a_stack_emb_representation_when_given() -> None:
    sources, real_key, base, hvg, design, hallmark = _check2_fixture()
    emb = pd.DataFrame(
        {"x": range(10), "y": range(10, 20)}, index=pd.Index([f"L{i}" for i in range(10)])
    )
    table = score_check2(
        sources, real_key, base, hvg, design, hallmark=hallmark, folds=2,
        stack_emb={"base_embed": emb},
    )
    assert "base_embed" in set(table["source"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_check2.py -v -k score_check2`
Expected: FAIL with `ImportError: cannot import name 'score_check2'`

- [ ] **Step 3: Write the implementation**

Add to `src/fmharness/check2.py` (needs three new imports at the top: `from collections.abc
import Callable`, `from fmharness.adapters import build_adapters`, `from fmharness.evaluation
import score_predictions` -- add alongside the existing imports, standard-library/third-party/
local groups per ruff's `I` sort):

```python
def score_check2(
    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    real_key: pd.DataFrame,
    base: pd.DataFrame,
    hvg: pd.Index,
    design: pd.DataFrame,
    *,
    hallmark: dict[str, tuple[list[str], str]],
    fixed_methods: tuple[str, ...] = FIXED_READOUTS,
    penalties: tuple[str, ...] = PENALTY_NAMES,
    folds: int = 5,
    stack_emb: dict[str, pd.DataFrame] | None = None,
    n_permutations: int = 1000,
) -> pd.DataFrame:
    """Check-2 table: fixed-signature readouts + representation-controlled penalized grid.

    ``design`` is the (patient, drug, y) AUC label frame -- the caller's responsibility to
    leakage-filter first (this function does not know ``filter_leakage`` exists; it scores
    whatever ``design`` it is handed, exactly like ``evaluation.score_delta_sources`` does for
    Check 1). (a) scores every ``sources`` delta through each named fixed Hallmark-derived
    readout (sensitivity -> ``-score`` vs AUC); (b) fits the SAME penalized regression
    (RidgeCV/LassoCV/ElasticNetCV, one per ``penalties`` entry) to the untreated ``base``
    expression, every ``sources`` delta, and any ``stack_emb`` embedding, leave-cell-line-out
    by grouped fold, so a difference across representations is the representation and not the
    model (Kurilov 2020). Returns one row per (source, method) with
    global/interaction/perdrug/p_label/regret@1/regret@3/n.
    """
    fixed_sigs = {
        "hallmark": hallmark,
        "proliferation": {n: hallmark[n] for n in PROLIFERATION if n in hallmark},
    }
    fixed_readouts = {
        m: build_adapters(["hallmark"], signatures=fixed_sigs[m])[0]
        for m in fixed_methods
        if m in fixed_sigs
    }
    uniq_lines = sorted(set(real_key["patient"].astype(str)))
    n_folds = max(1, min(folds, len(uniq_lines)))
    fold_of = {ln: i % n_folds for i, ln in enumerate(uniq_lines)}
    target_drugs = set(real_key["drug"].astype(str))
    design_target = design[design["drug"].astype(str).isin(target_drugs)]

    def _row(s: dict[str, float]) -> dict[str, object]:
        return {
            "global": s["global"],
            "interaction": s["interaction"],
            "perdrug": s["perdrug"],
            "p_label": s["p_label"],
            "regret@1": s["regret@1"],
            "regret@3": s["regret@3"],
            "n": int(s["n"]),
        }

    out: list[dict[str, object]] = []

    # (a) fixed-signature readouts on the delta sources (sensitivity -> -y_pred vs AUC).
    for name, (d, kk) in sources.items():
        for method, adapter in fixed_readouts.items():
            sens = np.asarray(adapter.predict(d), dtype=float)
            merged = pd.DataFrame(
                {"patient": kk["patient"].to_numpy(), "drug": kk["drug"].to_numpy(), "_s": sens}
            ).merge(design.rename(columns={"y": "y_true"}), on=["patient", "drug"], how="inner")
            if merged.empty:
                continue
            preds = pd.DataFrame(
                {
                    "patient": merged["patient"],
                    "drug": merged["drug"],
                    "y_true": merged["y_true"].to_numpy(),
                    "y_pred": -merged["_s"].to_numpy(),
                }
            )
            s = score_predictions(preds, n_perm=n_permutations)
            out.append({"source": name, "method": method, **_row(s)})

    # (b) representation-controlled penalized regression: baseline expression + every delta source.
    base_hvg = base.reindex(columns=hvg).fillna(0.0)
    representations: dict[str, dict[str, pd.DataFrame] | Callable[[str], pd.DataFrame]] = {
        "expr": lambda _drug: base_hvg
    }
    for name, (d, kk) in sources.items():
        representations[name] = repr_by_drug(d, kk, hvg)
    for label, emb in (stack_emb or {}).items():
        representations[label] = (lambda e: lambda _drug: e)(emb)
    for repr_name, feat in representations.items():
        for pen in penalties:
            preds = penalized_preds(feat, design_target, fold_of, n_folds, uniq_lines, pen)
            if preds.empty:
                continue
            s = score_predictions(preds, n_perm=n_permutations)
            out.append({"source": repr_name, "method": pen, **_row(s)})

    return pd.DataFrame(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_check2.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Update `score_generation_eval.py`'s `main()` to call `score_check2`**

In `scripts/score_generation_eval.py`, REPLACE the Task 1 import line entirely (do not just
extend it) -- once the scoring composition below moves into `score_check2`, this file no
longer calls `penalized_preds`, `repr_by_drug`, or references `PROLIFERATION` directly (they
were only used inside the inline loops Step 5 deletes next), so keeping them imported would be
an unused import (`ruff` `F401`):

```python
from fmharness.check2 import FIXED_READOUTS, PENALTY_NAMES, load_line_matrix, score_check2
```

Replace everything in `main()` from the `fixed_methods = [m.strip() for m in args.methods...`
line through the final `print(pd.DataFrame(out).to_string(index=False) if out else "(no scored
pairs)")` line (this deletes the `_row` local function, the two scoring loops, and the
`representations`/`base_hvg` construction -- all now inside `score_check2`) with:

```python
    stack_emb_map: dict[str, pd.DataFrame] = {}
    for spec in args.stack_emb or []:
        label, _, p = spec.partition("=")
        if not (label.strip() and p.strip()):
            ap.error(f"--stack-emb expects 'label=path', got {spec!r}")
        stack_emb_map[label.strip()] = load_line_matrix(_rel(repo, p.strip()))

    fixed_methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    penalties = tuple(p.strip() for p in args.penalties.split(",") if p.strip())
    out_df = score_check2(
        sources,
        real_key,
        base,
        hvg,
        design,
        hallmark=hallmark,
        fixed_methods=fixed_methods,
        penalties=penalties,
        folds=args.folds,
        stack_emb=stack_emb_map,
        n_permutations=args.n_permutations,
    )
    print(f"\n=== check 2: end-to-end vs {args.auc_tranche} AUC (leave-cell-line-out) ===")
    print(out_df.to_string(index=False) if not out_df.empty else "(no scored pairs)")
```

- [ ] **Step 6: Verify the script still imports cleanly**

Run: `uv run python -c "import ast; ast.parse(open('scripts/score_generation_eval.py').read())"`
Expected: no error.

- [ ] **Step 7: Run the full test suite**

Run: `uv run pytest -v`
Expected: all pass, no regressions.

- [ ] **Step 8: Lint and typecheck**

Run: `uv run ruff check src/fmharness/check2.py scripts/score_generation_eval.py tests/test_check2.py && uv run pyright src tests`
Expected: no errors (same one pre-existing exception).

- [ ] **Step 9: Commit**

```bash
git add src/fmharness/check2.py scripts/score_generation_eval.py tests/test_check2.py
git commit -m "refactor: extract score_check2, a leakage-agnostic Check-2 scoring composition"
```

---

### Task 3: Move the corpus-declaration CLI helpers into `fmharness/leakage.py`

`check1_registry_driver.py` currently owns `corpus_declared_partially`,
`ground_truth_source_declared_ambiguously`, and `parse_corpus_set` as module-level functions.
Task 4's `check2_registry_driver.py` needs the identical validation -- move them into
`fmharness/leakage.py` (alongside `filter_leakage`, which they support) so both driver scripts
import one copy instead of `check2_registry_driver.py` importing from a sibling script.

**Files:**
- Modify: `src/fmharness/leakage.py`
- Modify: `scripts/check1_registry_driver.py`
- Test: `tests/test_leakage.py`
- Test: `tests/test_check1_registry_driver.py`

**Interfaces:**
- Produces (moved, unchanged signatures): `corpus_declared_partially(corpus_lines: str | None,
  corpus_drugs: str | None) -> bool`, `ground_truth_source_declared_ambiguously(context: str |
  None, deltas_bundle: str | None) -> bool`, `parse_corpus_set(raw: str | None) -> set[str] |
  None`. Used by Task 4's `check2_registry_driver.py` `main()`.

- [ ] **Step 1: Move the three functions and their docstrings**

In `src/fmharness/check1_registry_driver.py` -- actually `scripts/check1_registry_driver.py`
-- cut `corpus_declared_partially`, `ground_truth_source_declared_ambiguously`, and
`parse_corpus_set` (their full bodies, unchanged, including docstrings) from between
`run_check1` and `main()`. Paste them at the end of `src/fmharness/leakage.py`, after
`filter_leakage`.

`src/fmharness/leakage.py`'s new tail (append after the existing `filter_leakage` function):

```python
def corpus_declared_partially(corpus_lines: str | None, corpus_drugs: str | None) -> bool:
    """True iff exactly one of --corpus-lines/--corpus-drugs was given, not both, not neither.

    filter_leakage only filters when it has a measured declaration on BOTH axes; a
    half-declared corpus silently scores identically to an unfiltered run (basis="unknown"),
    with no signal in the output that the declared corpus was ignored. A driver's main()
    should reject this combination up front rather than letting it through.
    """
    return (corpus_lines is None) != (corpus_drugs is None)


def ground_truth_source_declared_ambiguously(
    context: str | None, deltas_bundle: str | None
) -> bool:
    """True iff --context/--deltas-bundle are both given or both omitted.

    Exactly one must select the ground-truth (real_delta, real_key, base) source: --context
    rebuilds it live from a single-cell AnnData, --deltas-bundle reads a precomputed pseudobulk
    parquet bundle. The two are not interchangeable in practice -- a live --context rebuild is
    whatever Tahoe context snapshot currently sits on Alpine, not necessarily the one a
    published table was computed from -- so a driver's main() should reject an ambiguous
    combination rather than silently picking one.
    """
    return (context is None) == (deltas_bundle is None)


def parse_corpus_set(raw: str | None) -> set[str] | None:
    """Parse a comma-separated --corpus-lines/--corpus-drugs value into a set.

    The documented workflow has a human copy-paste a comma-separated list printed by an
    earlier step into these flags -- strip whitespace around each entry and drop empty
    entries, so a stray space after a comma ("A, B, C") does not silently produce a corpus
    entry (" B") that can never match a real line/drug name and weakens the leakage filter
    with no error. ``None`` (flag not given) stays ``None``.
    """
    if raw is None:
        return None
    return {s.strip() for s in raw.split(",") if s.strip()}
```

In `scripts/check1_registry_driver.py`, add an import (alongside the existing `from
fmharness.leakage import filter_leakage` line -- merge into one alphabetical import):

```python
from fmharness.leakage import (
    corpus_declared_partially,
    filter_leakage,
    ground_truth_source_declared_ambiguously,
    parse_corpus_set,
)
```

- [ ] **Step 2: Move the four tests**

In `tests/test_check1_registry_driver.py`, remove `test_parse_corpus_set_strips_whitespace_and_drops_empties`,
`test_parse_corpus_set_passes_none_through`, `test_corpus_declared_partially_rejects_a_half_declared_corpus`,
and `test_ground_truth_source_declared_ambiguously_requires_exactly_one` (full bodies). Update
the top import block to drop the three moved names, keeping only:

```python
from check1_registry_driver import run_check1
```

In `tests/test_leakage.py`, add near the bottom (after the existing `filter_leakage` tests;
extend the top import line `from fmharness.leakage import LeakageQueryable, filter_leakage` to
add the three new names, alphabetically):

```python
def test_parse_corpus_set_strips_whitespace_and_drops_empties() -> None:
    # The documented workflow has a human copy-paste a comma-separated list into these flags --
    # a stray space after a comma (or a trailing comma) must not produce a corpus entry like
    # " B" or "" that can never match a real line/drug name.
    assert parse_corpus_set(" A , B ,") == {"A", "B"}


def test_parse_corpus_set_passes_none_through() -> None:
    assert parse_corpus_set(None) is None


def test_corpus_declared_partially_rejects_a_half_declared_corpus() -> None:
    # filter_leakage only filters when BOTH pretraining_lines and pretraining_drugs are given
    # -- a half-declared corpus (e.g. only --corpus-lines) silently scores identically to an
    # unfiltered run, with nothing in the output to show the declared corpus was ignored. A
    # driver's main() must reject this combination via ap.error before it reaches
    # filter_leakage.
    assert corpus_declared_partially("L1", None) is True
    assert corpus_declared_partially(None, "d1") is True
    assert corpus_declared_partially("L1", "d1") is False
    assert corpus_declared_partially(None, None) is False


def test_ground_truth_source_declared_ambiguously_requires_exactly_one() -> None:
    # a live --context rebuild is not guaranteed to match a --deltas-bundle built from an
    # earlier Tahoe context snapshot (confirmed diverging in production 2026-08-12) -- exactly
    # one must be given, not both (ambiguous which wins) and not neither (nothing to score).
    assert ground_truth_source_declared_ambiguously(None, None) is True
    assert ground_truth_source_declared_ambiguously("tahoe_context.h5ad", "tahoe_deltas") is True
    assert ground_truth_source_declared_ambiguously("tahoe_context.h5ad", None) is False
    assert ground_truth_source_declared_ambiguously(None, "tahoe_deltas") is False
```

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -v`
Expected: all pass (the moved tests pass under their new location; `test_check1_registry_driver.py`
loses 4 tests, `test_leakage.py` gains 4 -- total count unchanged).

- [ ] **Step 4: Lint and typecheck**

Run: `uv run ruff check src/fmharness/leakage.py scripts/check1_registry_driver.py tests/test_leakage.py tests/test_check1_registry_driver.py && uv run pyright src tests`
Expected: no errors (same one pre-existing exception).

- [ ] **Step 5: Commit**

```bash
git add src/fmharness/leakage.py scripts/check1_registry_driver.py tests/test_leakage.py tests/test_check1_registry_driver.py
git commit -m "refactor: move corpus-declaration CLI helpers into fmharness.leakage"
```

---

### Task 4: `scripts/check2_registry_driver.py` -- the leakage-aware Check-2 driver

Mirrors `check1_registry_driver.py`'s shape. The design decision this task implements: filter
the GDSC2 AUC `design` frame (not the Tahoe `real_key`) via the same `filter_leakage`/
`PregeneratedStackGenerator` composition Check 1 uses. Every representation (`expr`,
`additive`, `knn`, `pca`, `nmf`, `stack`, any `--stack-emb`) is scored via a merge/groupby
against `design` downstream (inside `score_check2`), so filtering it once uniformly restricts
every representation to the same surviving pairs -- the delta sources themselves stay built
from the full, unfiltered Tahoe triple.

**Files:**
- Create: `scripts/check2_registry_driver.py`
- Test: `tests/test_check2_registry_driver.py`

**Interfaces:**
- Consumes: `FIXED_READOUTS`, `PENALTY_NAMES`, `load_line_matrix`, `score_check2`
  (`fmharness.check2`, Tasks 1-2); `filter_leakage`, `corpus_declared_partially`,
  `ground_truth_source_declared_ambiguously`, `parse_corpus_set` (`fmharness.leakage`, Task 3);
  `loo_baseline_source`, `learned_gene_panel`, `build_generated_deltas`, `build_tahoe_deltas`,
  `load_pert_map` (`fmharness.deltas`, existing); `build_sample_design`
  (`fmharness.evaluation`, existing); `load_tranche` (`fmharness.data.loaders`, existing);
  `load_hallmark` (`fmharness.signatures`, existing); `PregeneratedStackGenerator`
  (`fmharness.models.stack_generator`, existing).
- Produces: `run_check2(real_delta, real_key, base, *, query_baseline, generated_dir,
  pert_to_drug, checkpoint_label, hallmark_path, auc_design, n_hvg=2000, k=10,
  fixed_methods=FIXED_READOUTS, penalties=PENALTY_NAMES, folds=5, stack_emb=None,
  n_permutations=1000, pretraining_lines=None, pretraining_drugs=None,
  task_signal_in_pretrain="none") -> pd.DataFrame`. Importable by tests and by Task 5's
  real-data invocation.

- [ ] **Step 1: Write the failing test**

Create `tests/test_check2_registry_driver.py`:

```python
"""Tests for the registry-driven Check-2 driver, against small synthetic fixtures."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from check2_registry_driver import run_check2


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, dict[str, str], pd.DataFrame]:
    # 10 lines, 1 drug ("d1") each, 3 genes -- 10 lines so the representation grid's default
    # min_lines=8/min_train=5 are both satisfiable with folds=2 (5 train / 5 test per fold).
    genes = ["A", "B", "C"]
    lines = [f"L{i}" for i in range(10)]
    rng = np.random.default_rng(0)
    real_delta = pd.DataFrame(rng.standard_normal((10, 3)) + 5.0, columns=pd.Index(genes))
    real_key = pd.DataFrame({"patient": lines, "drug": ["d1"] * 10})
    base = pd.DataFrame(
        rng.standard_normal((10, 3)) + 10.0, columns=pd.Index(genes), index=pd.Index(lines)
    )

    query_baseline = tmp_path / "query_baseline.h5ad"
    _write_adata(query_baseline, base.to_numpy().tolist(), lines, genes)

    gdir = tmp_path / "generated"
    gdir.mkdir()
    # build_generated_deltas scores logcpm(generated) - logcpm(baseline), so
    # baseline * exp(real_delta) is the exact-recovery construction (matching
    # test_check1_registry_driver.py's own fixture convention).
    generated_vals = base.to_numpy() * np.exp(real_delta.to_numpy())
    _write_adata(gdir / "BRD-1.h5ad", generated_vals.tolist(), lines, genes)
    pert_to_drug = {"BRD-1": "d1"}

    # AUC labels for every line -- values themselves don't matter for the wiring tests below,
    # only that every line has a measured AUC for d1.
    auc_design = pd.DataFrame(
        {"patient": lines, "drug": ["d1"] * 10, "y": rng.standard_normal(10).tolist()}
    )
    return real_delta, real_key, base, query_baseline, gdir, pert_to_drug, auc_design


def test_run_check2_reports_rows_for_every_representation(tmp_path: Path) -> None:
    real_delta, real_key, base, query_baseline, gdir, pert_to_drug, auc_design = _fixture(tmp_path)
    table = run_check2(
        real_delta,
        real_key,
        base,
        query_baseline=query_baseline,
        generated_dir=gdir,
        pert_to_drug=pert_to_drug,
        checkpoint_label="test-checkpoint",
        hallmark_path=_hallmark_gmt(tmp_path),
        auc_design=auc_design,
        n_hvg=3,
        k=1,
        folds=2,
    )
    assert {"additive", "knn", "pca", "nmf", "stack", "expr"} <= set(table["source"])
    assert {"global", "interaction", "perdrug", "p_label", "n"} <= set(table.columns)


def test_run_check2_applies_leakage_filtering_to_every_representation(tmp_path: Path) -> None:
    # L0 x d1 is doubly-exposed -- filtering happens on auc_design BEFORE any representation is
    # scored, so every fixed-readout row (one per source x method, covering additive/knn/pca/
    # nmf/stack -- not just stack) must lose exactly L0's one pair.
    real_delta, real_key, base, query_baseline, gdir, pert_to_drug, auc_design = _fixture(tmp_path)
    kwargs = dict(
        real_delta=real_delta,
        real_key=real_key,
        base=base,
        query_baseline=query_baseline,
        generated_dir=gdir,
        pert_to_drug=pert_to_drug,
        checkpoint_label="test-checkpoint",
        hallmark_path=_hallmark_gmt(tmp_path),
        auc_design=auc_design,
        n_hvg=3,
        k=1,
        folds=2,
    )
    unfiltered = run_check2(**kwargs)
    filtered = run_check2(
        **kwargs,
        pretraining_lines={"L0"},
        pretraining_drugs={"d1"},
        task_signal_in_pretrain="adjacent",
    )
    fixed_before = unfiltered[unfiltered["method"].isin(["hallmark", "proliferation"])]
    fixed_after = filtered[filtered["method"].isin(["hallmark", "proliferation"])]
    merged = fixed_before.merge(
        fixed_after, on=["source", "method"], suffixes=("_before", "_after")
    )
    assert len(merged) == len(fixed_before) == len(fixed_after) > 0
    assert (merged["n_before"] - merged["n_after"] == 1).all()


def test_run_check2_reports_no_leakage_filtering_without_a_declared_corpus(tmp_path: Path) -> None:
    real_delta, real_key, base, query_baseline, gdir, pert_to_drug, auc_design = _fixture(tmp_path)
    table = run_check2(
        real_delta,
        real_key,
        base,
        query_baseline=query_baseline,
        generated_dir=gdir,
        pert_to_drug=pert_to_drug,
        checkpoint_label="test-checkpoint",
        hallmark_path=_hallmark_gmt(tmp_path),
        auc_design=auc_design,
        n_hvg=3,
        k=1,
        folds=2,
    )
    fixed = table[table["method"].isin(["hallmark", "proliferation"])]
    assert (fixed["n"] == 10).all()


def test_run_check2_prints_the_leakage_basis_when_a_corpus_is_declared(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    real_delta, real_key, base, query_baseline, gdir, pert_to_drug, auc_design = _fixture(tmp_path)
    run_check2(
        real_delta,
        real_key,
        base,
        query_baseline=query_baseline,
        generated_dir=gdir,
        pert_to_drug=pert_to_drug,
        checkpoint_label="test-checkpoint",
        hallmark_path=_hallmark_gmt(tmp_path),
        auc_design=auc_design,
        n_hvg=3,
        k=1,
        folds=2,
        pretraining_lines={"L0"},
        pretraining_drugs={"d1"},
        task_signal_in_pretrain="adjacent",
    )
    out = capsys.readouterr().out
    assert "basis=measured" in out
    assert "doubly_exposed_frac" in out


def test_run_check2_prints_unknown_basis_without_a_declared_corpus(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    real_delta, real_key, base, query_baseline, gdir, pert_to_drug, auc_design = _fixture(tmp_path)
    run_check2(
        real_delta,
        real_key,
        base,
        query_baseline=query_baseline,
        generated_dir=gdir,
        pert_to_drug=pert_to_drug,
        checkpoint_label="test-checkpoint",
        hallmark_path=_hallmark_gmt(tmp_path),
        auc_design=auc_design,
        n_hvg=3,
        k=1,
        folds=2,
    )
    out = capsys.readouterr().out
    assert "basis=unknown" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_check2_registry_driver.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'check2_registry_driver'`

- [ ] **Step 3: Write the implementation**

Create `scripts/check2_registry_driver.py`:

```python
"""Check 2 (end-to-end GDSC2 AUC prediction) through the harness-core registries, leakage-aware.

Filters the GDSC2 AUC ``design`` frame (not the Tahoe ``real_key``) via the same
``filter_leakage``/``PregeneratedStackGenerator``/``LeakageQueryable`` composition
``check1_registry_driver.py`` uses -- ``design`` is Check 2's actual evaluation set. Every
representation (``expr``, ``additive``, ``knn``, ``pca``, ``nmf``, ``stack``, any
``--stack-emb``) is scored via a merge/groupby against ``design`` downstream (inside
``fmharness.check2.score_check2``), so filtering it once uniformly restricts every
representation to the same surviving (patient, drug) pairs -- the same same-pair-count parity
Check 1's table has, without needing to filter the delta sources themselves; they stay built
from the full, unfiltered Tahoe triple. See
docs/superpowers/specs/2026-08-13-check2-leakage-aware-drug-aligned-design.md for the full
design rationale.

Run (see check1_registry_driver.py's own --context-vs---deltas-bundle caveat -- it applies here
identically: prefer --deltas-bundle unless you have specifically verified --context agrees):
  uv run python scripts/check2_registry_driver.py \\
      --deltas-bundle tahoe_deltas \\
      --query-baseline tahoe_query.h5ad \\
      --generated-dir generated_sciplex \\
      --pert-map context_by_drug/pert_to_cid.tsv \\
      --checkpoint-label drug-aligned \\
      --corpus-lines ACH-000681 \\
      --corpus-drugs 6918289,11626560,104741,11707110,3385
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd

from fmharness.check2 import FIXED_READOUTS, PENALTY_NAMES, load_line_matrix, score_check2
from fmharness.data.loaders import load_tranche
from fmharness.deltas import (
    build_generated_deltas,
    build_tahoe_deltas,
    learned_gene_panel,
    load_pert_map,
    loo_baseline_source,
)
from fmharness.evaluation import build_sample_design
from fmharness.leakage import (
    corpus_declared_partially,
    filter_leakage,
    ground_truth_source_declared_ambiguously,
    parse_corpus_set,
)
from fmharness.models.stack_generator import PregeneratedStackGenerator
from fmharness.schema import TaskSignal
from fmharness.signatures import load_hallmark


def _rel(repo: Path, p: str) -> Path:
    """Resolve ``p`` against the repo root unless it is already absolute."""
    q = Path(p)
    return q if q.is_absolute() else repo / q


def run_check2(
    real_delta: pd.DataFrame,
    real_key: pd.DataFrame,
    base: pd.DataFrame,
    *,
    query_baseline: Path,
    generated_dir: Path,
    pert_to_drug: dict[str, str],
    checkpoint_label: str,
    hallmark_path: Path,
    auc_design: pd.DataFrame,
    n_hvg: int = 2000,
    k: int = 10,
    fixed_methods: tuple[str, ...] = FIXED_READOUTS,
    penalties: tuple[str, ...] = PENALTY_NAMES,
    folds: int = 5,
    stack_emb: dict[str, pd.DataFrame] | None = None,
    n_permutations: int = 1000,
    pretraining_lines: set[str] | None = None,
    pretraining_drugs: set[str] | None = None,
    task_signal_in_pretrain: TaskSignal = "none",
) -> pd.DataFrame:
    """Check-2 table: fixed-signature readouts + representation grid, leakage-filtered.

    ``real_delta``/``real_key``/``base`` are the Tahoe ground-truth triple, used UNFILTERED to
    build every delta source (additive/knn/pca/nmf/stack) -- a source's prediction for a
    contaminated pair sitting unused in the sources dict is harmless; it never gets scored once
    ``auc_design`` excludes that pair. ``auc_design`` is the GDSC2 (patient, drug, y) AUC label
    frame (``build_sample_design``'s own output) -- THIS is what gets leakage-filtered, once,
    before any scoring: every representation is scored via a merge/groupby against it
    downstream, so filtering it once uniformly restricts every representation (not just
    ``stack``) to the same surviving pairs. ``pretraining_lines``/``pretraining_drugs`` (both
    default ``None``) declare the Stack checkpoint's measured pretraining overlap, exactly as
    ``check1_registry_driver.run_check1`` -- both must be given together for filtering to
    activate.
    """
    model = PregeneratedStackGenerator(
        generated_dir,
        pert_to_drug,
        checkpoint_label=checkpoint_label,
        pretraining_lines=pretraining_lines,
        pretraining_drugs=pretraining_drugs,
        task_signal_in_pretrain=task_signal_in_pretrain,
    )
    filtered_design, profile = filter_leakage(auc_design, model)
    if profile.basis == "measured":
        print(
            f"Check-2 leakage filter: basis=measured, "
            f"doubly_exposed_frac={profile.doubly_exposed_frac:.3f}, "
            f"line_overlap_frac={profile.line_overlap_frac:.3f}, "
            f"drug_overlap_fraction={profile.drug_overlap_fraction:.3f}"
        )
    else:
        print(f"Check-2 leakage filter: basis={profile.basis} (no corpus declared -- unfiltered)")

    hallmark = load_hallmark(hallmark_path)
    hvg = pd.Index(real_delta.var(axis=0).sort_values(ascending=False).index[:n_hvg])
    learned_genes = learned_gene_panel(real_delta, hallmark_path, n_hvg=n_hvg)
    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "additive": loo_baseline_source("additive", real_delta, real_key, base, k=k),
        "knn": loo_baseline_source("knn", real_delta, real_key, base, k=k),
        "pca": loo_baseline_source("pca", real_delta, real_key, base, k=k, genes=learned_genes),
        "nmf": loo_baseline_source("nmf", real_delta, real_key, base, k=k, genes=learned_genes),
        "stack": build_generated_deltas(generated_dir, query_baseline, pert_to_drug),
    }

    return score_check2(
        sources,
        real_key,
        base,
        hvg,
        filtered_design,
        hallmark=hallmark,
        fixed_methods=fixed_methods,
        penalties=penalties,
        folds=folds,
        stack_emb=stack_emb,
        n_permutations=n_permutations,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context", default=None, help="Tahoe context AnnData (build_tahoe_context)")
    ap.add_argument(
        "--deltas-bundle",
        default=None,
        help="dir of real_delta.parquet/real_key.parquet/base.parquet -- provide this OR "
        "--context, not both (a live --context rebuild is not guaranteed to match a bundle "
        "built earlier from a different Tahoe context snapshot -- see "
        "check1_registry_driver.py's own --deltas-bundle help text for the full caveat)",
    )
    ap.add_argument(
        "--query-baseline", required=True, help="AnnData fed to stack-generation as --test-adata"
    )
    ap.add_argument("--generated-dir", required=True, help="dir of Stack-generated <pert>.h5ad")
    ap.add_argument("--pert-map", required=True, help="TSV 'pert_id<TAB>cid' (context split)")
    ap.add_argument("--checkpoint-label", required=True, help="e.g. cytokine- or drug-aligned")
    ap.add_argument("--auc-tranche", default="gdscv2", help="measured-AUC cohort for check 2")
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--hallmark-path", default="data/static/hallmark_signatures.gmt", help="Hallmark .gmt path"
    )
    ap.add_argument(
        "--methods",
        default=",".join(FIXED_READOUTS),
        help="fixed-signature readouts on the delta sources (subset of hallmark, proliferation)",
    )
    ap.add_argument(
        "--penalties",
        default=",".join(PENALTY_NAMES),
        help="penalized regressions for the representation grid (subset of l2, l1, en)",
    )
    ap.add_argument(
        "--folds",
        type=int,
        default=5,
        help="grouped-by-cell-line folds for the leakage-free penalized fit; set >= #lines "
        "(e.g. 999) for true leave-one-cell-line-out",
    )
    ap.add_argument(
        "--stack-emb",
        nargs="*",
        default=None,
        help="precomputed per-line FM embeddings to add as check-2 representations, each "
        "'label=path' (path .h5ad/.parquet/.csv, index/obs = cell line id). Repeatable, e.g. "
        "--stack-emb base=emb_base.h5ad aligned=emb_aligned.h5ad",
    )
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--corpus-lines", default=None, help="comma-separated declared pretrain lines")
    ap.add_argument("--corpus-drugs", default=None, help="comma-separated declared pretrain drugs")
    args = ap.parse_args()

    if corpus_declared_partially(args.corpus_lines, args.corpus_drugs):
        ap.error(
            "--corpus-lines and --corpus-drugs must both be given together (or neither, to "
            "run unfiltered) -- giving only one silently disables leakage filtering"
        )
    if ground_truth_source_declared_ambiguously(args.context, args.deltas_bundle):
        ap.error("provide exactly one of --context (single-cell) or --deltas-bundle (pseudobulk)")

    repo = Path(__file__).resolve().parent.parent
    if args.deltas_bundle:
        bdir = _rel(repo, args.deltas_bundle)
        real_delta = pd.read_parquet(bdir / "real_delta.parquet")
        real_key = pd.read_parquet(bdir / "real_key.parquet")
        base = pd.read_parquet(bdir / "base.parquet")
    else:
        real_delta, real_key, base = build_tahoe_deltas(ad.read_h5ad(args.context))
    pert_to_drug = load_pert_map(_rel(repo, args.pert_map))

    _, auc_design = build_sample_design(
        load_tranche(args.auc_tranche, repo), "all", "auc", drug_key="pubchem_cid"
    )

    stack_emb_map: dict[str, pd.DataFrame] = {}
    for spec in args.stack_emb or []:
        label, _, p = spec.partition("=")
        if not (label.strip() and p.strip()):
            ap.error(f"--stack-emb expects 'label=path', got {spec!r}")
        stack_emb_map[label.strip()] = load_line_matrix(_rel(repo, p.strip()))

    fixed_methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    penalties = tuple(p.strip() for p in args.penalties.split(",") if p.strip())
    table = run_check2(
        real_delta,
        real_key,
        base,
        query_baseline=_rel(repo, args.query_baseline),
        generated_dir=_rel(repo, args.generated_dir),
        pert_to_drug=pert_to_drug,
        checkpoint_label=args.checkpoint_label,
        hallmark_path=repo / args.hallmark_path,
        auc_design=auc_design,
        n_hvg=args.n_hvg,
        k=args.k,
        fixed_methods=fixed_methods,
        penalties=penalties,
        folds=args.folds,
        stack_emb=stack_emb_map,
        n_permutations=args.n_permutations,
        pretraining_lines=parse_corpus_set(args.corpus_lines),
        pretraining_drugs=parse_corpus_set(args.corpus_drugs),
        task_signal_in_pretrain="adjacent" if args.corpus_lines else "none",
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_check2_registry_driver.py -v`
Expected: PASS (5 tests). If the leakage-parity test is flaky, it is almost certainly because
`auc_design`'s synthetic `y` values happened to make some fixed-readout `merged` frame empty
for a source/method combination on one run but not the other -- fix by widening the fixture
(more lines), not by loosening the assertion.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v`
Expected: all pass, no regressions.

- [ ] **Step 6: Lint and typecheck**

Run: `uv run ruff check scripts/check2_registry_driver.py tests/test_check2_registry_driver.py && uv run pyright src tests`
Expected: no errors on the new files (`scripts/` is outside pyright's strict scope, matching
`check1_registry_driver.py`); same one pre-existing pyright exception elsewhere.

- [ ] **Step 7: Commit**

```bash
git add scripts/check2_registry_driver.py tests/test_check2_registry_driver.py
git commit -m "feat: add the leakage-aware, registry-driven Check-2 driver"
```

---

### Task 5: Run for real, and write the results into the docs and the deck

All data this task needs already exists locally in this worktree (`generated/` = cytokine-
aligned Stack output, 33 files; `generated_sciplex/` = drug-aligned Stack output, 33 files;
`tahoe_deltas/` = the ground-truth bundle; GDSC2 AUC data already loadable via `--auc-tranche
gdscv2`) -- unlike Task 9 of `docs/superpowers/plans/2026-08-11-stack-drug-alignment-and-check1.md`,
this task is not gated on anything outstanding and its acceptance can be fully verified now.

**Files:**
- Modify: `docs/tahoe_generation_results.md`
- Modify: `scripts/update_harness_overview_slides.py`
- Modify: `scripts/plot_generation_eval_summary.py`
- Modify (outside this worktree, copied in and back out): `docs/prospective_evaluation_harness_overview.pptx`

- [ ] **Step 1: Re-run Check 1 for all three checkpoint variants**

The handoff doc (`docs/superpowers/specs/2026-08-13-check2-leakage-aware-drug-aligned.md`)
only ever reported the drug-aligned `r` value (0.021 both ways), not the full row
(`r_offdiag`/`rank`/`n_pairs`) needed for the docs table and the deck -- re-run all three to get
complete rows on data/code that has not changed since:

```bash
uv run python scripts/check1_registry_driver.py \
    --deltas-bundle tahoe_deltas --query-baseline tahoe_query.h5ad \
    --generated-dir generated --pert-map context_by_drug/pert_to_cid.tsv \
    --checkpoint-label cytokine-aligned > /tmp/check1_cytokine.txt
uv run python scripts/check1_registry_driver.py \
    --deltas-bundle tahoe_deltas --query-baseline tahoe_query.h5ad \
    --generated-dir generated_sciplex --pert-map context_by_drug/pert_to_cid.tsv \
    --checkpoint-label drug-aligned-unfiltered > /tmp/check1_drug_unfiltered.txt
uv run python scripts/check1_registry_driver.py \
    --deltas-bundle tahoe_deltas --query-baseline tahoe_query.h5ad \
    --generated-dir generated_sciplex --pert-map context_by_drug/pert_to_cid.tsv \
    --checkpoint-label drug-aligned --corpus-lines ACH-000681 \
    --corpus-drugs 6918289,11626560,104741,11707110,3385 > /tmp/check1_drug_filtered.txt
cat /tmp/check1_cytokine.txt /tmp/check1_drug_unfiltered.txt /tmp/check1_drug_filtered.txt
```

The cytokine-aligned run's `stack` row must match the published table exactly (r=0.012,
r_offdiag=-0.002, rank=0.644, n_pairs=1568) -- if it does not, that is a regression in this
plan's earlier tasks to fix before proceeding, not a result to report. The drug-aligned rows'
`r` must match the handoff doc's already-measured 0.021 (both variants) -- same check.

- [ ] **Step 2: Run Check 2 for the cytokine-aligned checkpoint**

```bash
uv run python scripts/check2_registry_driver.py \
    --deltas-bundle tahoe_deltas \
    --query-baseline tahoe_query.h5ad \
    --generated-dir generated \
    --pert-map context_by_drug/pert_to_cid.tsv \
    --checkpoint-label cytokine-aligned \
    > /tmp/check2_cytokine.txt
cat /tmp/check2_cytokine.txt
```

No `--corpus-lines`/`--corpus-drugs` -- this checkpoint was aligned on CELLxGENE + Parse PBMC
cytokines only, no Tahoe/GDSC2 line or drug overlap to declare. Expect `basis=unknown` printed.
This run's numbers must match the existing `stack (gen)` row currently in
`docs/tahoe_generation_results.md`'s Check-2 tables exactly, since Task 2's refactor changed no
behavior -- if they differ, that is a regression in this plan's earlier tasks to fix before
proceeding, not a result to report.

- [ ] **Step 3: Run Check 2 for the drug-aligned checkpoint, unfiltered**

```bash
uv run python scripts/check2_registry_driver.py \
    --deltas-bundle tahoe_deltas \
    --query-baseline tahoe_query.h5ad \
    --generated-dir generated_sciplex \
    --pert-map context_by_drug/pert_to_cid.tsv \
    --checkpoint-label drug-aligned-unfiltered \
    > /tmp/check2_drug_unfiltered.txt
cat /tmp/check2_drug_unfiltered.txt
```

Deliberately no corpus flags -- this is the filtered-vs-unfiltered comparison row, mirroring
Check 1's existing table.

- [ ] **Step 4: Run Check 2 for the drug-aligned checkpoint, leakage-filtered**

```bash
uv run python scripts/check2_registry_driver.py \
    --deltas-bundle tahoe_deltas \
    --query-baseline tahoe_query.h5ad \
    --generated-dir generated_sciplex \
    --pert-map context_by_drug/pert_to_cid.tsv \
    --checkpoint-label drug-aligned \
    --corpus-lines ACH-000681 \
    --corpus-drugs 6918289,11626560,104741,11707110,3385 \
    > /tmp/check2_drug_filtered.txt
cat /tmp/check2_drug_filtered.txt
```

Expect `basis=measured` printed, with `doubly_exposed_frac` close to Check 1's measured 0.003
on the same declared corpus (not necessarily identical -- Check 1 filters Tahoe pairs, this
filters GDSC2 AUC pairs, a different frame that may not have identical coverage of A549 x the 5
drugs). If it differs substantially, note why in the results doc rather than silently using it.

- [ ] **Step 5: Write the drug-aligned Check-1 row into `docs/tahoe_generation_results.md`**

Read the current `## Check 1` table (rows for additive/nmf/pca/knn/stack). Replace the single
`stack (gen)` row with three, long-format (one more row each, not new columns), using Step 1's
complete output:

```markdown
| stack (gen, cytokine-aligned) | 0.012 | -0.002 | 0.644 | 1568 |
| stack (gen, drug-aligned, unfiltered) | 0.021 | <r_offdiag from Step 1> | <rank from Step 1> | <n_pairs from Step 1> |
| stack (gen, drug-aligned, leak-excluded) | 0.021 | <r_offdiag from Step 1> | <rank from Step 1> | <n_pairs from Step 1> |
```

Add one sentence to the existing takeaway paragraph noting drug alignment roughly doubles
Stack's Check-1 correlation vs. cytokine-aligned but both stay far below the additive floor and
ceiling, and that leakage filtering does not change the drug-aligned number (matching the
handoff doc's own phrasing).

- [ ] **Step 6: Write the new Check-2 rows into `docs/tahoe_generation_results.md`**

In the `## Check 2` section's fixed-signature-readout table and representation-controlled
ladder table, replace the single `stack (gen)` row in each with three rows -- `stack (gen,
cytokine-aligned)`, `stack (gen, drug-aligned, unfiltered)`, `stack (gen, drug-aligned,
leak-excluded)` -- populated from Steps 2-4's `/tmp/check2_*.txt` output. Add a sentence to the
existing takeaway noting whether leakage filtering changed anything for Check 2 (per Step 4's
`doubly_exposed_frac`), the same way it didn't for Check 1.

- [ ] **Step 7: Update `scripts/update_harness_overview_slides.py`'s hardcoded rows**

In `CHECK1_ROWS`, replace the `("stack (generated)", "0.012", "-0.002", "0.644", "1568", "red")`
entry with three rows (same 6-tuple shape: label, r, off-diag, rank, pairs, tint), using the
same numbers written into the docs file in Step 5. In `SIG_ROWS` and `LADDER_ROWS`, replace each
existing `("stack (gen)", ...)` / `("stack (gen delta)", ...)` entry with three rows using Step
6's numbers, keeping every other column and the tuple shape unchanged.

- [ ] **Step 8: Update `scripts/plot_generation_eval_summary.py`'s hardcoded dicts**

In `CHECK1`, replace the `"stack\n(generated)": (0.012, -0.002)` entry with three entries (e.g.
`"stack\n(cytokine)"`, `"stack\n(drug, unfilt.)"`, `"stack\n(drug)"`) using Step 5's numbers. In
`CHECK2_RIDGE` and `SEL_GAP`, replace each `"stack\n(gen delta)"` entry with the same three
labels, using Step 6's global/interaction and gap@1/gap@3 numbers respectively. Keep every
other dict entry and the `_edges` function's `"stack" in lab` matching unchanged (it already
substring-matches, so three `"stack\n..."`-prefixed labels are still colored correctly without
further edits).

- [ ] **Step 9: Copy the deck into this worktree, regenerate, copy back**

```bash
cp /Users/gillenlu/Repositories/fm-pdo-evaluator/docs/prospective_evaluation_harness_overview.pptx \
   docs/prospective_evaluation_harness_overview.pptx
uv run python scripts/plot_generation_eval_summary.py
uv run python scripts/update_harness_overview_slides.py
cp docs/prospective_evaluation_harness_overview.pptx \
   /Users/gillenlu/Repositories/fm-pdo-evaluator/docs/prospective_evaluation_harness_overview.pptx
```

Confirm with Lucas before the final copy-back -- it overwrites a file outside this worktree.

- [ ] **Step 10: Commit the docs changes in this worktree**

The `.pptx` itself is gitignored in both worktrees (nothing to commit there). Commit the results
doc and the two script edits:

```bash
git add docs/tahoe_generation_results.md scripts/update_harness_overview_slides.py scripts/plot_generation_eval_summary.py
git commit -m "docs: add drug-aligned Check-1/Check-2 rows, regenerate the harness-overview deck"
```

---

## Plan exit criteria

- Tasks 1-4's tests all pass; `uv run pytest -v` is green for the full suite.
- `uv run ruff check src tests` and `uv run pyright src tests` are both clean (the one
  pre-existing, unrelated `src/fmharness/deltas.py:194` error is not this plan's to fix).
- `score_generation_eval.py`'s cytokine-aligned, unfiltered Check-2 numbers are unchanged
  before/after Tasks 1-2's refactor (verified in Task 5 Step 5).
- `check2_registry_driver.py`'s leakage filter drops the same (patient, drug) pair from every
  representation, not just `stack` (proven by Task 4's test and re-confirmed on real data in
  Task 5 Step 3 by inspecting `doubly_exposed_frac`).
- `docs/tahoe_generation_results.md` and `docs/prospective_evaluation_harness_overview.pptx`
  both show cytokine-aligned, drug-aligned-unfiltered, and drug-aligned-leak-excluded rows for
  both Check 1 and Check 2.

## Not in this plan

- MOA-level stratification of the selection-gap shortlists (proposed in
  `docs/tahoe_generation_results.md`'s own "Proposed" callouts) -- not built yet anywhere in
  this codebase, separate future work.
- The "gate" (Hallmark-vs-random-genes readout-power check) staying leakage-unfiltered is a
  deliberate design decision (it scores the real Tahoe delta, not any model's output), not an
  oversight -- not revisited here.
- A drug-aligned Stack *embedding* (encoder output, as opposed to generated delta) -- the
  existing `base (embed)`/`aligned (embed)` rows are cytokine-checkpoint-only; producing a
  drug-aligned embedding would need a new Alpine extraction job, out of scope here.
