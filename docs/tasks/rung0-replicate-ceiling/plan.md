# Rung 0 — replicate ceiling: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the rung-0 replicate ceiling — split-half reliability of the Tahoe (line, drug) delta on the declared gene and drug panels — with schema-validated provenance, controls, and power reporting.

**Architecture:** Port the archived lineage's measurement into a testable core (build / score / null functions with known-answer controls), add the MDE helper and the tranche/promotion machinery the new schema requires, verify locally on synthetic fixtures, then run once on Alpine and promote.

**Tech Stack:** Python 3.11, numpy/pandas, DuckDB (in-engine aggregation over parquet), matplotlib (Agg), pydantic v2 schema (`fmharness.schema`), Slurm via `scripts/alpine/ralpine`.

**Spec:** `docs/tasks/rung0-replicate-ceiling/design.md` (this folder). Read it first; every choice below argues from it.

## Global Constraints

- Declared statistic: per-(line, drug) Pearson over the declared panel, **mean over pairs** as the headline; median is descriptive only. Spearman-Brown `2r/(1+r)` stated wherever used. (SPEC rungs 0–1.)
- Every measurement step ships positive and negative controls as known-answer tests importing the real functions; every promoted comparison reports its MDE at α=0.05, power=0.80. (SPEC rule 4.)
- `uv run pytest -q` green before every push; project-rule markers for this task: `-m "step_promote or step_document or step_score or step_null"`. (PROCESS §3.)
- CI also gates on `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pyright` (strict, `include = ["src", "tests"]`). Every task leaves the files it touched clean under all three; run them scoped to tracked dirs locally (`ruff check src tests scripts`, `ruff format --check src tests scripts`) because the working tree carries untracked strays CI never sees. (Added 2026-08-27 during execution — the plan had gated only on pytest.)
- Landed documents reference only landed work; cluster files cited repo-relative plus "on Alpine"; absolute site paths never. (PROCESS §5.)
- Stage files explicitly — **never `git add -A`** (untracked data and archives sit in the working tree).
- Commit messages state the root cause; no result numbers in messages. Every commit ends with the `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer.
- Alpine only through `./scripts/alpine/ralpine`; deployment git-only; verify job inputs exist before submitting; test on synthetic data before cluster time. (PROCESS §2–3.)
- Ported code comes from the archived lineage: `git show rung0-replicate-ceiling-old-lineage:<path>` (branch on origin). Do not hand-retype ported code.

---

### Task 1: Dependencies and the download-script design amendment

**Files:**
- Modify: `pyproject.toml` (dependencies)
- Modify: `docs/tasks/rung0-replicate-ceiling/design.md` (ported-apparatus row + dated entry)
- Modify: `docs/DATA.md` (download attribution wording)

**Interfaces:**
- Produces: pandas/duckdb/matplotlib importable in later tasks; the design amendment later tasks implement (download script extracted; `build_tahoe_pseudobulk_deltas.py` NOT ported).

Why the amendment: the design's ported-apparatus table listed `scripts/build_tahoe_pseudobulk_deltas.py` "unchanged", but that script imports `fmharness.deltas` (rung-1 bundle machinery) at module top — unchanged, it cannot import on this branch. Rung 0's provenance chain needs only its download half.

- [x] **Step 1: Add dependencies**

In `pyproject.toml`, extend `dependencies` (keep the existing comment and entries):

```toml
dependencies = [
    "pydantic>=2.7",
    "numpy>=1.26",
    "pandas>=2.2",
    "duckdb>=1.0",
    "matplotlib>=3.8",
]
```

- [x] **Step 2: Lock and sanity-import**

Run: `uv lock && uv sync --extra dev && uv run python -c "import pandas, duckdb, matplotlib; print('ok')"`
Expected: `ok`

- [x] **Step 3: Amend the design's ported-apparatus table**

In `design.md`, replace the row

`| scripts/build_tahoe_pseudobulk_deltas.py, 00_target_cids.sbatch, 01_pseudobulk_shortcut.sbatch | unchanged (provenance chain; not re-run) |`

with

`| scripts/download_tahoe_pseudobulk_de.py, 00_target_cids.sbatch, 01_pseudobulk_shortcut.sbatch | download logic extracted from the archived lineage's build_tahoe_pseudobulk_deltas.py, whose delta-bundle aggregation imports rung-1 machinery and arrives with rung 1; 01 adapted to call the extracted script; not re-run |`

and append to the decision history:

`- **2026-08-28** — Plan-time amendment: build_tahoe_pseudobulk_deltas.py is not ported — its module-level import of the rung-1 delta-bundle machinery cannot resolve on this branch, and only its download half is in rung 0's provenance chain. That half lands as scripts/download_tahoe_pseudobulk_de.py; the bundle aggregation arrives with rung 1.`

(Use the actual current date if it differs.)

- [x] **Step 4: Align DATA.md**

In `docs/DATA.md`: in the **Download** paragraph, change "by `scripts/alpine/01_pseudobulk_shortcut.sbatch`" to "by `scripts/alpine/01_pseudobulk_shortcut.sbatch` (download logic in `scripts/download_tahoe_pseudobulk_de.py`; the archived lineage's variant of the same pull performed the 2026-07-24 download)". In the **Delta bundle** bullet, change "(`scripts/build_tahoe_pseudobulk_deltas.py`, landing with the rung-0 task as part of this pool's provenance chain)" to "(arrives with rung 1; its download half is landed as `scripts/download_tahoe_pseudobulk_de.py`)".

- [x] **Step 5: Test and commit**

Run: `uv run pytest -q` — expected: all pass.

```bash
git add pyproject.toml uv.lock docs/tasks/rung0-replicate-ceiling/design.md docs/DATA.md
git commit -m "chore: rung-0 dependencies; design amendment extracting the DE download script"
```

---

### Task 2: Statistics — port `bootstrap_aggregate_pvalue`, add `minimum_detectable_aggregate`

**Files:**
- Create: `src/fmharness/statistics.py`
- Create: `tests/test_statistics_known_answers.py`

**Interfaces:**
- Produces: `bootstrap_aggregate_pvalue(observed_agg: float, null_draws: np.ndarray, n_obs: int, *, agg=np.mean, n_boot=2000, seed=0, min_null_draws=10) -> tuple[float, float, float]` (p, ci_lo, ci_hi) and `minimum_detectable_aggregate(observed_draws, null_draws, n_obs, *, agg=np.mean, alpha=0.05, power=0.80, n_boot=2000, seed=0, min_null_draws=10) -> float`. Tasks 3–4 and the promotion pipeline consume both.

- [x] **Step 1: Port the shared helper**

Run: `git show rung0-replicate-ceiling-old-lineage:src/fmharness/statistics.py > src/fmharness/statistics.py`

Then edit its module docstring's second paragraph (which cites unlanded scripts) to:

```
This was first found and fixed for the rung-0 replicate ceiling on the archived lineage
(branch ``rung0-replicate-ceiling-old-lineage``), where it had been independently
reintroduced in several scripts. One shared, tested helper is how it stays fixed
everywhere at once.
```

- [x] **Step 2: Add the MDE helper**

Append to `src/fmharness/statistics.py`:

```python
def minimum_detectable_aggregate(
    observed_draws: np.ndarray,
    null_draws: np.ndarray,
    n_obs: int,
    *,
    agg: Callable[[np.ndarray], float] = np.mean,
    alpha: float = 0.05,
    power: float = 0.80,
    n_boot: int = 2000,
    seed: int = 0,
    min_null_draws: int = 10,
) -> float:
    """Smallest true aggregate detectable against ``null_draws`` at (``alpha``, ``power``).

    The same bootstrap as ``bootstrap_aggregate_pvalue``, read in the other direction.
    The detection threshold is the (1 - alpha) quantile of the null aggregate's sampling
    distribution at ``n_obs``; the MDE is the aggregate value whose own sampling
    distribution -- the observed draws' spread, recentred -- clears that threshold with
    probability ``power``. Valid for shift-equivariant aggregates (mean, median,
    quantiles), where shifting every draw by c shifts the aggregate by c.

    Reported beside every promoted comparison: a null result with no MDE cannot be told
    apart from an underpowered experiment.
    """
    null_draws = np.asarray(null_draws, dtype=np.float64)
    null_draws = null_draws[np.isfinite(null_draws)]
    observed_draws = np.asarray(observed_draws, dtype=np.float64)
    observed_draws = observed_draws[np.isfinite(observed_draws)]
    if null_draws.size < min_null_draws or observed_draws.size < 2 or n_obs < 1:
        return float("nan")
    rng = np.random.default_rng(seed)
    boot_null = np.array(
        [agg(rng.choice(null_draws, size=n_obs, replace=True)) for _ in range(n_boot)]
    )
    crit = float(np.quantile(boot_null, 1.0 - alpha))
    centred = observed_draws - float(np.mean(observed_draws))
    boot_centred = np.array(
        [agg(rng.choice(centred, size=n_obs, replace=True)) for _ in range(n_boot)]
    )
    # P(agg + theta >= crit) >= power  <=>  theta >= crit - Q_{boot_centred}(1 - power)
    return float(crit - np.quantile(boot_centred, 1.0 - power))
```

- [x] **Step 3: Write the known-answer tests**

Create `tests/test_statistics_known_answers.py`:

```python
"""Known-answer validation for the statistics rung 0 reports (SPEC rule 4).

Each test plants an answer and requires the real, shipped function to recover it, and
plants nothing and requires null. This file exists because, on the archived lineage, a
null test compared an aggregate against a distribution of single draws and reported that
a reproducible ceiling had failed; running against a known answer catches that class of
defect immediately. The recurring error is aggregate-vs-per-item, so units are pinned
explicitly throughout.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fmharness.statistics import bootstrap_aggregate_pvalue, minimum_detectable_aggregate

pytestmark = pytest.mark.known_answer


def test_bootstrap_aggregate_pvalue_returns_null_when_there_is_no_signal() -> None:
    rng = np.random.default_rng(1)
    observed = rng.normal(0.14, 0.13, 1600)
    null = rng.normal(0.14, 0.13, 500)
    p, _, _ = bootstrap_aggregate_pvalue(float(np.mean(observed)), null, observed.size)
    assert p > 0.05, f"no-signal case must not be significant, got p={p}"


def test_bootstrap_aggregate_pvalue_recovers_a_planted_difference() -> None:
    rng = np.random.default_rng(2)
    observed = rng.normal(0.30, 0.13, 1600)
    null = rng.normal(0.14, 0.13, 500)
    p, _, _ = bootstrap_aggregate_pvalue(float(np.mean(observed)), null, observed.size)
    assert p < 0.01, f"planted difference must be recovered, got p={p}"


def test_the_wrong_form_is_the_one_that_fails_to_recover_it() -> None:
    # Pins the aggregate-vs-per-item defect: with the SAME planted data, comparing the
    # aggregate to individual null draws looks non-significant where the real function
    # does not.
    rng = np.random.default_rng(2)
    observed = rng.normal(0.30, 0.13, 1600)
    null = rng.normal(0.14, 0.13, 500)
    wrong = float(np.mean(null >= np.mean(observed)))
    correct, _, _ = bootstrap_aggregate_pvalue(float(np.mean(observed)), null, observed.size)
    assert wrong > 0.05, "the defective form should look non-significant on planted signal"
    assert correct < 0.01
    assert wrong > correct * 10, "the defect inflates p by orders of magnitude"


def test_spearman_brown_lifts_half_data_reliability_as_documented() -> None:
    def sb(r: float) -> float:
        return 2 * r / (1 + r)

    assert sb(0.0) == pytest.approx(0.0)
    assert sb(1.0) == pytest.approx(1.0)
    assert sb(0.3) > 0.3, "correcting half-data reliability must raise it"
    assert sb(0.5) > sb(0.3)


def test_minimum_detectable_aggregate_matches_the_normal_closed_form() -> None:
    # Normal null (mu0, sigma) and normal observed spread (sigma), agg = mean over n:
    #   MDE = mu0 + (z_{1-alpha} + z_{power}) * sigma / sqrt(n)
    rng = np.random.default_rng(0)
    mu0, sigma, n = 0.03, 0.10, 400
    null = rng.normal(mu0, sigma, 2000)
    observed = rng.normal(0.20, sigma, 2000)
    mde = minimum_detectable_aggregate(observed, null, n)
    z95, z80 = 1.6449, 0.8416
    expected = mu0 + (z95 + z80) * sigma / math.sqrt(n)
    assert abs(mde - expected) < 0.005, f"mde={mde}, closed form={expected}"


def test_an_aggregate_at_the_mde_clears_the_null() -> None:
    # Self-consistency with the p-value: the MDE sits above the null's critical value,
    # so a result at the MDE must come out significant at the same alpha.
    rng = np.random.default_rng(3)
    null = rng.normal(0.03, 0.10, 2000)
    observed = rng.normal(0.10, 0.10, 2000)
    n = 400
    mde = minimum_detectable_aggregate(observed, null, n)
    p, _, _ = bootstrap_aggregate_pvalue(mde, null, n)
    assert p < 0.05, f"an aggregate at the MDE must be significant, got p={p}"


def test_minimum_detectable_aggregate_returns_nan_on_too_few_null_draws() -> None:
    assert math.isnan(minimum_detectable_aggregate(np.ones(50), np.ones(3), 10))
```

- [x] **Step 4: Run the tests**

Run: `uv run pytest tests/test_statistics_known_answers.py -v`
Expected: all PASS (the ported function must satisfy the ported tests unchanged; the MDE tests exercise the new helper).

- [x] **Step 5: Full suite, commit**

Run: `uv run pytest -q` — expected: pass.

```bash
git add src/fmharness/statistics.py tests/test_statistics_known_answers.py
git commit -m "feat: shared aggregate-vs-null statistics with known answers, plus the MDE the design requires"
```

---

### Task 3: Measurement core — port and refactor `delta_reproducibility.py` with build/score/null controls

**Files:**
- Create: `scripts/delta_reproducibility.py`
- Create: `tests/test_rung0_controls.py`
- Test fixture helper lives inside the test file (no separate fixture module).

**Interfaces:**
- Consumes: `bootstrap_aggregate_pvalue`, `minimum_detectable_aggregate` (Task 2).
- Produces, importable from `scripts/delta_reproducibility.py` (tests add `scripts/` to `sys.path` via `importlib`):
  - `build_split_half_frame(paths: list[str], target_names: list[str], repl_col: str | None, tmp: Path, memory_limit: str = "36GB") -> tuple[pd.DataFrame, str]` — long frame with columns `patient, drug, gene_name, lfc0, lfc1` and the chosen replicate column name.
  - `masked_rowwise_pearson(a: np.ndarray, b: np.ndarray, min_genes: int) -> np.ndarray`
  - `score_split_half(de: pd.DataFrame, panel: set[str], min_genes: int = 50) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]` — (per-pair r, piv0, piv1), pivots indexed by (patient, drug).
  - `stratified_null_draws(piv0, piv1, n_perm: int = 500, seed: int = 0, min_genes: int = 50) -> dict[str, np.ndarray]` — keys `any_pair`, `diff_drug`, `same_drug`.

- [x] **Step 1: Port the script**

Run: `git show rung0-replicate-ceiling-old-lineage:scripts/delta_reproducibility.py > scripts/delta_reproducibility.py`

- [x] **Step 2: Write the failing control tests**

Create `tests/test_rung0_controls.py`:

```python
"""Positive and negative controls for rung 0's build, score, and null steps (SPEC rule 4).

Every test runs the REAL functions from scripts/delta_reproducibility.py on synthetic
replicate pools with planted, known answers -- not reimplementations of their logic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.known_answer

_SPEC = importlib.util.spec_from_file_location(
    "delta_reproducibility",
    Path(__file__).resolve().parents[1] / "scripts" / "delta_reproducibility.py",
)
dr = importlib.util.module_from_spec(_SPEC)
sys.modules["delta_reproducibility"] = dr
_SPEC.loader.exec_module(dr)


def _write_fixture_pool(
    tmp: Path,
    n_lines: int = 4,
    n_drugs: int = 3,
    n_genes: int = 300,
    plates: tuple[str, ...] = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"),
    signal_sd: float = 1.0,
    noise_sd: float = 1.0,
    drug_sd: float = 0.0,
    seed: int = 0,
) -> Path:
    """A synthetic replicate pool in the DE table's own shape, one parquet file.

    Per (line, drug, gene): a fixed pair-specific signal (sd ``signal_sd``), an optional
    drug-shared component (sd ``drug_sd``), plus independent per-plate noise (sd
    ``noise_sd``). Expected split-half r over genes, as plate count grows:
    (signal_sd^2 + drug_sd^2) / (signal_sd^2 + drug_sd^2 + noise_sd^2 / plates_per_half).
    """
    rng = np.random.default_rng(seed)
    lines = [f"L{i}" for i in range(n_lines)]
    drugs = [f"D{j}" for j in range(n_drugs)]
    genes = [f"G{k}" for k in range(n_genes)]
    drug_eff = {d: rng.normal(0.0, drug_sd, n_genes) for d in drugs}
    rows = []
    for li in lines:
        for d in drugs:
            signal = rng.normal(0.0, signal_sd, n_genes) + drug_eff[d]
            for p in plates:
                lfc = signal + rng.normal(0.0, noise_sd, n_genes)
                rows.append(
                    pd.DataFrame(
                        {
                            "Cell_ID_DepMap": li,
                            "drug": d,
                            "gene_name": genes,
                            "log2FoldChange": lfc,
                            "plate": p,
                        }
                    )
                )
    pool_dir = tmp / "pseudobulk_differential_expression"
    pool_dir.mkdir(parents=True)
    out = pool_dir / "train-00000-of-00001.parquet"
    pd.concat(rows, ignore_index=True).to_parquet(out, index=False)
    return out


def test_build_positive_planted_pool_comes_out_with_the_planted_shape(tmp_path) -> None:
    path = _write_fixture_pool(tmp_path)
    de, chosen = dr.build_split_half_frame(
        [str(path)], ["D0", "D1", "D2"], None, tmp_path / "duck", memory_limit="2GB"
    )
    de = de.dropna(subset=["lfc0", "lfc1"])
    assert chosen == "plate"
    pairs = de.groupby(["patient", "drug"]).ngroups
    assert pairs == 12, f"planted 4 lines x 3 drugs, built {pairs} pairs"
    assert de["gene_name"].nunique() == 300


def test_build_negative_no_replication_yields_no_scoreable_pairs(tmp_path) -> None:
    path = _write_fixture_pool(tmp_path, plates=("P1",))  # one plate: one half stays empty
    de, _ = dr.build_split_half_frame(
        [str(path)], ["D0", "D1", "D2"], None, tmp_path / "duck", memory_limit="2GB"
    )
    assert de.dropna(subset=["lfc0", "lfc1"]).empty


def test_score_positive_planted_reliability_is_recovered(tmp_path) -> None:
    # signal_sd = noise_sd = 1, 8 plates -> 4 per half; half-mean noise sd^2 = 1/4.
    # Expected r = 1 / (1 + 0.25) = 0.8.
    path = _write_fixture_pool(tmp_path, n_genes=600, signal_sd=1.0, noise_sd=1.0)
    de, _ = dr.build_split_half_frame(
        [str(path)], ["D0", "D1", "D2"], None, tmp_path / "duck", memory_limit="2GB"
    )
    de = de.dropna(subset=["lfc0", "lfc1"])
    r, _, _ = dr.score_split_half(de, set(de["gene_name"].unique()))
    r = r[np.isfinite(r)]
    assert abs(float(np.mean(r)) - 0.8) < 0.05, f"planted 0.8, recovered {np.mean(r):.3f}"


def test_score_negative_zero_signal_returns_null(tmp_path) -> None:
    path = _write_fixture_pool(tmp_path, signal_sd=0.0, noise_sd=1.0)
    de, _ = dr.build_split_half_frame(
        [str(path)], ["D0", "D1", "D2"], None, tmp_path / "duck", memory_limit="2GB"
    )
    de = de.dropna(subset=["lfc0", "lfc1"])
    r, piv0, piv1 = dr.score_split_half(de, set(de["gene_name"].unique()))
    r = r[np.isfinite(r)]
    nulls = dr.stratified_null_draws(piv0, piv1, n_perm=200, seed=0)
    from fmharness.statistics import bootstrap_aggregate_pvalue

    p, _, _ = bootstrap_aggregate_pvalue(float(np.mean(r)), nulls["diff_drug"], r.size)
    assert p > 0.05, f"no planted signal must not clear the null, got p={p}"


def test_null_positive_planted_components_recover_the_stratum_ordering(tmp_path) -> None:
    # Drug-shared + line-specific components: matched pairs share both, same-drug
    # mismatches share only the drug component, diff-drug mismatches share nothing.
    path = _write_fixture_pool(
        tmp_path, n_lines=6, n_drugs=4, signal_sd=0.7, drug_sd=0.7, noise_sd=0.7
    )
    de, _ = dr.build_split_half_frame(
        [str(path)], ["D0", "D1", "D2", "D3"], None, tmp_path / "duck", memory_limit="2GB"
    )
    de = de.dropna(subset=["lfc0", "lfc1"])
    r, piv0, piv1 = dr.score_split_half(de, set(de["gene_name"].unique()))
    r = r[np.isfinite(r)]
    nulls = dr.stratified_null_draws(piv0, piv1, n_perm=300, seed=0)
    observed, same_d, diff_d = (
        float(np.mean(r)),
        float(np.mean(nulls["same_drug"])),
        float(np.mean(nulls["diff_drug"])),
    )
    assert observed > same_d + 0.05, f"observed {observed:.3f} !> same_drug {same_d:.3f}"
    assert same_d > diff_d + 0.05, f"same_drug {same_d:.3f} !> diff_drug {diff_d:.3f}"


def test_null_negative_signal_free_strata_sit_at_their_floors(tmp_path) -> None:
    path = _write_fixture_pool(tmp_path, signal_sd=0.0, drug_sd=0.0, noise_sd=1.0)
    de, _ = dr.build_split_half_frame(
        [str(path)], ["D0", "D1", "D2"], None, tmp_path / "duck", memory_limit="2GB"
    )
    de = de.dropna(subset=["lfc0", "lfc1"])
    _, piv0, piv1 = dr.score_split_half(de, set(de["gene_name"].unique()))
    nulls = dr.stratified_null_draws(piv0, piv1, n_perm=200, seed=0)
    for stratum, draws in nulls.items():
        assert abs(float(np.mean(draws))) < 0.05, f"{stratum} floor is not ~0 on noise"
```

- [x] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_rung0_controls.py -v`
Expected: FAIL — `build_split_half_frame` etc. do not exist yet (the ported script has `_split_half_deltas` and inline logic).

- [x] **Step 4: Refactor the ported script's core**

In `scripts/delta_reproducibility.py`:

(a) Rename `_split_half_deltas` to `build_split_half_frame`, add the `memory_limit: str = "36GB"` parameter, and use it in `con.execute(f"SET memory_limit='{memory_limit}'")`. Keep the DuckDB query byte-identical otherwise.

(b) Add the vectorized masked correlation (replaces both the `_corr` groupby-apply and the per-draw loop math — no nested Python loops):

```python
def masked_rowwise_pearson(a: np.ndarray, b: np.ndarray, min_genes: int) -> np.ndarray:
    """Pearson r per row between ``a`` and ``b``, over entries finite in both.

    Vectorized across rows; rows with fewer than ``min_genes`` shared finite entries or
    zero variance come back NaN.
    """
    ok = np.isfinite(a) & np.isfinite(b)
    n = ok.sum(axis=1)
    a0 = np.where(ok, a, 0.0)
    b0 = np.where(ok, b, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = a0.sum(axis=1) / n
        mb = b0.sum(axis=1) / n
        ac = np.where(ok, a - ma[:, None], 0.0)
        bc = np.where(ok, b - mb[:, None], 0.0)
        cov = (ac * bc).sum(axis=1)
        va = (ac**2).sum(axis=1)
        vb = (bc**2).sum(axis=1)
        r = cov / np.sqrt(va * vb)
    r[(n < min_genes) | (va <= 0) | (vb <= 0)] = np.nan
    return r
```

(c) Extract scoring:

```python
def score_split_half(
    de: pd.DataFrame, panel: set[str], min_genes: int = 50
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Per-(line, drug) split-half Pearson over the panel genes, plus the half pivots."""
    d = de[de["gene_name"].isin(panel)]
    piv0 = d.pivot_table(index=["patient", "drug"], columns="gene_name", values="lfc0")
    piv1 = d.pivot_table(index=["patient", "drug"], columns="gene_name", values="lfc1")
    common = piv0.index.intersection(piv1.index)
    piv0, piv1 = piv0.loc[common], piv1.loc[common]
    r = masked_rowwise_pearson(
        piv0.to_numpy(dtype=float), piv1.to_numpy(dtype=float), min_genes
    )
    return r, piv0, piv1
```

(d) Extract the stratified null, vectorized over candidate pairs (replaces `_draw`'s rejection loop):

```python
def stratified_null_draws(
    piv0: pd.DataFrame,
    piv1: pd.DataFrame,
    n_perm: int = 500,
    seed: int = 0,
    min_genes: int = 50,
) -> dict[str, np.ndarray]:
    """Mismatched-pair null correlations per stratum.

    any_pair: two different pairs (continuity with the archived lineage's first run).
    diff_drug: different line AND drug -- the generic-structure floor the ceiling clears.
    same_drug: same drug, different line -- the line-specificity floor.
    """
    lines = piv0.index.get_level_values(0).to_numpy(dtype=str)
    drugs = piv0.index.get_level_values(1).to_numpy(dtype=str)
    n = len(piv0)
    ii, jj = np.divmod(np.arange(n * n), n)
    off = ii != jj
    ii, jj = ii[off], jj[off]
    same_drug = drugs[ii] == drugs[jj]
    same_line = lines[ii] == lines[jj]
    strata = {
        "any_pair": np.ones(ii.size, dtype=bool),
        "diff_drug": ~same_drug & ~same_line,
        "same_drug": same_drug & ~same_line,
    }
    rng = np.random.default_rng(seed)
    a, b = piv0.to_numpy(dtype=float), piv1.to_numpy(dtype=float)
    out: dict[str, np.ndarray] = {}
    for name, mask in strata.items():
        avail = np.flatnonzero(mask)
        if avail.size == 0:
            out[name] = np.array([])
            continue
        pick = rng.choice(avail, size=min(n_perm, avail.size), replace=avail.size < n_perm)
        r = masked_rowwise_pearson(a[ii[pick]], b[jj[pick]], min_genes)
        out[name] = r[np.isfinite(r)]
    return out
```

(e) In `main()`, replace the old `_corr`/pivot/`_draw` block with calls to these functions (the summary dict is rebuilt in Task 4 — for now keep the ported summary keys wired to the new functions' outputs so the script still runs end-to-end: `r`, `nulls = stratified_null_draws(...)`, `nl = nulls["diff_drug"] ...` as before).

- [x] **Step 5: Run the control tests to verify they pass**

Run: `uv run pytest tests/test_rung0_controls.py -v`
Expected: all PASS. If `test_build_negative...` fails because DuckDB's `hash()` put the single plate in bucket 0 not 1, the frame still has one all-NaN half — the assertion holds either way; investigate before touching the test.

- [x] **Step 6: Full suite, commit**

Run: `uv run pytest -q` — expected: pass.

```bash
git add scripts/delta_reproducibility.py tests/test_rung0_controls.py
git commit -m "feat: rung-0 measurement core, testable and vectorized, with build/score/null controls"
```

---

### Task 4: Reporting layer — mean headline, MDE, terciles, per-gene diagnostic, pool description, figure

**Files:**
- Modify: `scripts/delta_reproducibility.py`
- Modify: `tests/test_rung0_controls.py` (append tests)

**Interfaces:**
- Consumes: Task 3's core functions; Task 2's statistics.
- Produces:
  - `effect_size_terciles(piv0, piv1, r) -> dict[str, float]` — keys `splithalf_mean_r_tercile1..3` (tercile 1 = smallest effects).
  - `per_gene_reliability(piv0, piv1, min_pairs: int = 20) -> pd.DataFrame` — columns `gene, n_pairs, r`.
  - `summarize(r, nulls, seed) -> dict` — the headline row (columns below).
  - `main()` writing, under `--out-dir`: `rung0_delta_reproducibility.csv`, `rung0_per_gene_reliability.csv`, `rung0_pool_description.csv`, `rung0_ceiling.png`, and the params sidecar.
  - CLI: `--drug-names-file` (one Tahoe drug name per line; bypasses the HuggingFace name lookup so fixtures and offline runs need no `datasets` import).

- [x] **Step 1: Append the failing reporting tests**

Append to `tests/test_rung0_controls.py`:

```python
def test_tercile_control_rises_monotonically_with_planted_effect_size(tmp_path) -> None:
    # Three drugs with graded signal size, same noise: split-half r must rise with
    # effect size, tercile 1 -> 3.
    rng = np.random.default_rng(7)
    lines = [f"L{i}" for i in range(6)]
    genes = [f"G{k}" for k in range(400)]
    plates = tuple(f"P{p}" for p in range(8))
    rows = []
    for d, s in (("D0", 0.3), ("D1", 0.8), ("D2", 2.0)):
        for li in lines:
            signal = rng.normal(0.0, s, len(genes))
            for p in plates:
                rows.append(
                    pd.DataFrame(
                        {
                            "Cell_ID_DepMap": li,
                            "drug": d,
                            "gene_name": genes,
                            "log2FoldChange": signal + rng.normal(0.0, 1.0, len(genes)),
                            "plate": p,
                        }
                    )
                )
    pool_dir = tmp_path / "pseudobulk_differential_expression"
    pool_dir.mkdir(parents=True)
    pd.concat(rows, ignore_index=True).to_parquet(
        pool_dir / "train-00000-of-00001.parquet", index=False
    )
    de, _ = dr.build_split_half_frame(
        [str(pool_dir / "train-00000-of-00001.parquet")],
        ["D0", "D1", "D2"],
        None,
        tmp_path / "duck",
        memory_limit="2GB",
    )
    de = de.dropna(subset=["lfc0", "lfc1"])
    r, piv0, piv1 = dr.score_split_half(de, set(de["gene_name"].unique()))
    terc = dr.effect_size_terciles(piv0, piv1, r)
    assert (
        terc["splithalf_mean_r_tercile1"]
        < terc["splithalf_mean_r_tercile2"]
        < terc["splithalf_mean_r_tercile3"]
    ), f"terciles not monotone: {terc}"


def test_per_gene_reliability_separates_reliable_from_noise_genes(tmp_path) -> None:
    # Half the genes carry pair-specific signal, half are pure noise: the diagnostic
    # must rank the signal genes above the noise genes.
    rng = np.random.default_rng(8)
    lines = [f"L{i}" for i in range(8)]
    drugs = [f"D{j}" for j in range(4)]
    genes = [f"S{k}" for k in range(100)] + [f"N{k}" for k in range(100)]
    plates = tuple(f"P{p}" for p in range(8))
    rows = []
    for li in lines:
        for d in drugs:
            signal = np.concatenate([rng.normal(0.0, 1.5, 100), np.zeros(100)])
            for p in plates:
                rows.append(
                    pd.DataFrame(
                        {
                            "Cell_ID_DepMap": li,
                            "drug": d,
                            "gene_name": genes,
                            "log2FoldChange": signal + rng.normal(0.0, 1.0, 200),
                            "plate": p,
                        }
                    )
                )
    pool_dir = tmp_path / "pseudobulk_differential_expression"
    pool_dir.mkdir(parents=True)
    pd.concat(rows, ignore_index=True).to_parquet(
        pool_dir / "train-00000-of-00001.parquet", index=False
    )
    de, _ = dr.build_split_half_frame(
        [str(pool_dir / "train-00000-of-00001.parquet")],
        drugs,
        None,
        tmp_path / "duck",
        memory_limit="2GB",
    )
    de = de.dropna(subset=["lfc0", "lfc1"])
    _, piv0, piv1 = dr.score_split_half(de, set(de["gene_name"].unique()))
    pg = dr.per_gene_reliability(piv0, piv1, min_pairs=10)
    mean_signal = pg[pg["gene"].str.startswith("S")]["r"].mean()
    mean_noise = pg[pg["gene"].str.startswith("N")]["r"].mean()
    assert mean_signal > mean_noise + 0.3, f"signal {mean_signal:.3f} vs noise {mean_noise:.3f}"


def test_summarize_headlines_the_mean_and_reports_both_mdes() -> None:
    rng = np.random.default_rng(9)
    r = rng.normal(0.14, 0.06, 1600)
    nulls = {
        "any_pair": rng.normal(0.03, 0.05, 500),
        "diff_drug": rng.normal(0.03, 0.05, 500),
        "same_drug": rng.normal(0.07, 0.05, 500),
    }
    s = dr.summarize(r, nulls, seed=0)
    assert s["splithalf_mean_r"] == pytest.approx(float(np.mean(r)), abs=1e-9)
    assert s["spearman_brown_full"] == pytest.approx(
        2 * s["splithalf_mean_r"] / (1 + s["splithalf_mean_r"]), abs=1e-6
    )
    assert s["p_vs_null"] < 0.01 and s["p_vs_same_drug"] < 0.01
    assert 0 < s["mde_80_vs_diff_drug"] < s["splithalf_mean_r"], "trivially powered here"
    assert 0 < s["mde_80_vs_same_drug"] < s["splithalf_mean_r"]
    assert s["splithalf_median_r"] is not None  # descriptive column retained
```

- [x] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_rung0_controls.py -v -k "tercile or per_gene or summarize"`
Expected: FAIL — functions not defined.

- [x] **Step 3: Implement the reporting layer**

Add to `scripts/delta_reproducibility.py`:

```python
def effect_size_terciles(
    piv0: pd.DataFrame, piv1: pd.DataFrame, r: np.ndarray
) -> dict[str, float]:
    """Split-half mean r within terciles of per-pair effect size (mean |delta|).

    The empirical positive control: an assay that cannot find more reproducibility where
    there is more signal is broken. Tercile 1 = smallest effects.
    """
    a, b = piv0.to_numpy(dtype=float), piv1.to_numpy(dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    mean_abs = np.where(ok, np.abs(a + b) / 2.0, 0.0).sum(axis=1) / np.maximum(
        ok.sum(axis=1), 1
    )
    finite = np.isfinite(r)
    edges = np.quantile(mean_abs[finite], [1 / 3, 2 / 3])
    out: dict[str, float] = {}
    for t in (1, 2, 3):
        lo = -np.inf if t == 1 else edges[t - 2]
        hi = np.inf if t == 3 else edges[t - 1]
        sel = finite & (mean_abs > lo) & (mean_abs <= hi)
        out[f"splithalf_mean_r_tercile{t}"] = round(float(np.mean(r[sel])), 3)
    return out


def per_gene_reliability(
    piv0: pd.DataFrame, piv1: pd.DataFrame, min_pairs: int = 20
) -> pd.DataFrame:
    """The transpose diagnostic: each gene's delta correlated across pairs between halves.

    Unpromoted (see design.md): says which panel genes carry reproducible perturbation
    signal, as the evidence base for any future panel restriction.
    """
    a, b = piv0.to_numpy(dtype=float).T, piv1.to_numpy(dtype=float).T
    r = masked_rowwise_pearson(a, b, min_pairs)
    n = (np.isfinite(a) & np.isfinite(b)).sum(axis=1)
    return pd.DataFrame(
        {"gene": piv0.columns.to_numpy(), "n_pairs": n, "r": np.round(r, 4)}
    ).sort_values("r", ascending=False)
```

and

```python
def summarize(r: np.ndarray, nulls: dict[str, np.ndarray], seed: int = 0) -> dict:
    """The headline row: mean-over-pairs Pearson (the declared statistic), its nulls,
    p-values from the bootstrapped null aggregate, and the MDEs (SPEC rule 4)."""
    from fmharness.statistics import bootstrap_aggregate_pvalue, minimum_detectable_aggregate

    r = r[np.isfinite(r)]
    mean = float(np.mean(r))
    nl = nulls["diff_drug"] if nulls["diff_drug"].size else nulls["any_pair"]
    p_boot, ci_lo, ci_hi = bootstrap_aggregate_pvalue(mean, nl, r.size, seed=seed)
    p_same = bootstrap_aggregate_pvalue(mean, nulls["same_drug"], r.size, seed=seed)[0]
    sb = 2 * mean / (1 + mean) if mean > -1 else float("nan")
    return {
        "n_pairs": int(r.size),
        "splithalf_mean_r": round(mean, 3),
        "splithalf_median_r": round(float(np.median(r)), 3),
        "splithalf_q1_r": round(float(np.quantile(r, 0.25)), 3),
        "splithalf_q3_r": round(float(np.quantile(r, 0.75)), 3),
        "spearman_brown_full": round(sb, 3),
        "frac_pos": round(float(np.mean(r > 0)), 3),
        "null_any_pair_mean_r": round(float(np.mean(nulls["any_pair"])), 3),
        "null_diff_drug_mean_r": round(float(np.mean(nulls["diff_drug"])), 3),
        "null_same_drug_mean_r": round(float(np.mean(nulls["same_drug"])), 3),
        "null_n_draws": int(nl.size),
        "p_vs_null": round(p_boot, 4),
        "p_vs_same_drug": round(p_same, 4),
        "null_mean_ci_lo": round(ci_lo, 3),
        "null_mean_ci_hi": round(ci_hi, 3),
        "mde_80_vs_diff_drug": round(
            minimum_detectable_aggregate(r, nl, r.size, seed=seed), 4
        ),
        "mde_80_vs_same_drug": round(
            minimum_detectable_aggregate(r, nulls["same_drug"], r.size, seed=seed), 4
        ),
    }
```

- [x] **Step 4: Pool description and figure**

Add:

```python
DOSE_CANDIDATES = ("dose", "Dose", "drug_dose", "concentration", "dose_uM")


def pool_description(paths: list[str], target_names: list[str], repl: str, tmp: Path) -> pd.DataFrame:
    """Measured composition of the consumed pool (design: 'measured not asserted'):
    per (line, drug) the replicate-row count, distinct plates per half, and dose levels
    when a dose column exists."""
    import duckdb  # type: ignore  # heavy path; imported where used

    con = duckdb.connect()
    con.execute(f"SET temp_directory='{tmp}'")
    cols = list(
        con.execute("DESCRIBE SELECT * FROM read_parquet(?) LIMIT 0", [paths]).df()["column_name"]
    )
    dose = next((c for c in DOSE_CANDIDATES if c in cols), None)
    dose_expr = f"count(DISTINCT {dose})" if dose else "NULL"
    return con.execute(
        f"""SELECT Cell_ID_DepMap AS patient, drug,
                   count(*) AS n_rows,
                   count(DISTINCT {repl}) AS n_plates,
                   count(DISTINCT {repl}) FILTER (WHERE hash({repl}) % 2 = 0) AS n_plates_half0,
                   count(DISTINCT {repl}) FILTER (WHERE hash({repl}) % 2 = 1) AS n_plates_half1,
                   {dose_expr} AS n_dose_levels
            FROM read_parquet(?)
            WHERE drug IN (SELECT unnest(?)) AND {repl} IS NOT NULL
            GROUP BY Cell_ID_DepMap, drug ORDER BY patient, drug""",
        [paths, target_names],
    ).df()


def write_figure(r: np.ndarray, nulls: dict[str, np.ndarray], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(-0.3, 0.8, 56)
    ax.hist(r[np.isfinite(r)], bins=bins, density=True, alpha=0.65, label="matched pairs")
    ax.hist(nulls["diff_drug"], bins=bins, density=True, alpha=0.45, label="diff-drug null")
    ax.hist(nulls["same_drug"], bins=bins, density=True, alpha=0.45, label="same-drug null")
    ax.axvline(float(np.nanmean(r)), color="k", lw=1.5, label="mean (headline)")
    ax.set_xlabel("split-half Pearson r per (line, drug) pair")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
```

- [x] **Step 5: Rewire `main()`**

In `main()`:
- Add `--drug-names-file` (default None): when given, `names = sorted({ln.strip() for ln in Path(...).read_text().splitlines() if ln.strip()})` and `_target_names` is not called (no HuggingFace import).
- Replace `--out` with `--out-dir` (default `rung0_outputs`); write `rung0_delta_reproducibility.csv` (one row: `{"replicate_col": repl, "n_genes": len(hvg), **summarize(r, nulls, seed), **effect_size_terciles(piv0, piv1, r)}`), `rung0_per_gene_reliability.csv`, `rung0_pool_description.csv`, `rung0_ceiling.png`; keep `_write_params_sidecar` (now beside the summary CSV).
- Delete the superseded summary-dict block, the `_corr` closure, and `_draw` (Task 3 already replaced their logic).
- Keep the panel handling (`--panel-file` pins; top-HVG fallback prints its warning) exactly as ported.

- [x] **Step 6: Run the reporting tests, then an end-to-end fixture run**

Run: `uv run pytest tests/test_rung0_controls.py -v`
Expected: all PASS.

End-to-end through the real CLI (PROCESS §3's synthetic gate), from the repo root:

```bash
uv run python - <<'PY'
import subprocess, sys
sys.path.insert(0, "tests")
from pathlib import Path
from test_rung0_controls import _write_fixture_pool
tmp = Path("/tmp/rung0_e2e"); tmp.exists() and __import__("shutil").rmtree(tmp)
p = _write_fixture_pool(tmp)
(tmp / "names.txt").write_text("D0\nD1\nD2\n")
(tmp / "panel.txt").write_text("\n".join(f"G{k}" for k in range(300)) + "\n")
subprocess.run([sys.executable, "scripts/delta_reproducibility.py",
    "--local-dir", str(tmp), "--drug-names-file", str(tmp / "names.txt"),
    "--panel-file", str(tmp / "panel.txt"), "--n-perm", "200",
    "--out-dir", str(tmp / "out")], check=True)
print(sorted(x.name for x in (tmp / "out").iterdir()))
PY
```

Expected: exits 0; prints the four output files plus the params sidecar; the printed summary's `splithalf_mean_r` is ≈0.8 (the fixture's planted reliability) and `p_vs_null` ≤ 0.005. (Use the scratchpad path of your session if `/tmp` is disallowed.)

- [x] **Step 7: Full suite, commit**

Run: `uv run pytest -q` — expected: pass.

```bash
git add scripts/delta_reproducibility.py tests/test_rung0_controls.py
git commit -m "feat: rung-0 reporting -- mean headline with MDEs, tercile control, per-gene diagnostic, measured pool description"
```

---

### Task 5: `promote_result.py` — promotion that emits a schema-valid `PromotedResult`

**Files:**
- Create: `scripts/promote_result.py`
- Create: `tests/test_promote_result.py`

**Interfaces:**
- Consumes: `fmharness.schema.PromotedResult`, `EnvironmentSnapshot`.
- Produces: CLI `python scripts/promote_result.py --task <slug> --result <path> --script <repo-relative> --input <path> [--input ...] --seed 0 --data-commit <tranche-content-hash> --arg key=value [--log <path>] [--job-id N] [--repo <path>]`, and importable `promote(...) -> Path` (used by tests).

- [x] **Step 1: Write the failing tests**

Create `tests/test_promote_result.py`:

```python
"""The promotion gate: a result becomes evidence only with a valid record beside it,
and an artifact cannot change under its claim (SPEC rule 1; design 'Run and promotion')."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from fmharness.schema import PromotedResult

_SPEC = importlib.util.spec_from_file_location(
    "promote_result", Path(__file__).resolve().parents[1] / "scripts" / "promote_result.py"
)
pr = importlib.util.module_from_spec(_SPEC)
sys.modules["promote_result"] = pr
_SPEC.loader.exec_module(pr)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "made_it.py").write_text("print('x')\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "input.txt").write_text("input-bytes\n")
    (tmp_path / "result.csv").write_text("a,b\n1,2\n")
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t",
         "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "init"], check=True)
    return tmp_path


def _promote(repo: Path, **kw) -> Path:
    defaults = dict(
        task="rung0-replicate-ceiling",
        result=repo / "result.csv",
        script="scripts/made_it.py",
        inputs=[repo / "docs" / "input.txt"],
        seed=0,
        data_commit="c" * 64,
        args={"tranche_id": "tahoe100m-pseudobulk-de.v1"},
        job_id="123",
        log=None,
        repo=repo,
    )
    defaults.update(kw)
    return pr.promote(**defaults)


def test_promotion_writes_a_schema_valid_record_beside_the_result(repo: Path) -> None:
    record_path = _promote(repo)
    promoted = repo / "results" / "rung0-replicate-ceiling" / "result.csv"
    assert promoted.exists()
    record = PromotedResult.model_validate_json(record_path.read_text())
    assert record.result_sha256 == pr.sha256_of(promoted)
    assert record.clean_tree is True
    assert record.environment.cuda_deterministic is False
    assert record.environment.data_commit == "c" * 64
    assert record.args["tranche_id"] == "tahoe100m-pseudobulk-de.v1"


def test_promotion_refuses_when_the_promoted_copy_differs(repo: Path) -> None:
    _promote(repo)
    (repo / "result.csv").write_text("a,b\n9,9\n")  # task-side copy changed
    with pytest.raises(SystemExit, match="differ"):
        _promote(repo)


def test_promotion_refuses_a_result_with_no_inputs(repo: Path) -> None:
    with pytest.raises(SystemExit, match="input"):
        _promote(repo, inputs=[])


def test_promotion_records_a_dirty_tree_honestly(repo: Path) -> None:
    (repo / "scripts" / "made_it.py").write_text("print('changed')\n")
    record_path = _promote(repo)
    assert PromotedResult.model_validate_json(record_path.read_text()).clean_tree is False


def test_promotion_refuses_a_script_not_in_the_repo(repo: Path) -> None:
    with pytest.raises(SystemExit, match="script"):
        _promote(repo, script="scripts/never_existed.py")
```

- [x] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_promote_result.py -v`
Expected: FAIL — `scripts/promote_result.py` does not exist.

- [x] **Step 3: Implement**

Create `scripts/promote_result.py`:

```python
"""Promote a run's output into committed evidence (SPEC rule 1; PROCESS §1 'Promote').

A log is not evidence; a result becomes evidence when a claim cites it, and citing means
a ``results/<task-slug>/`` copy with a schema-validated ``PromotedResult`` beside it.
Three fields cannot be reconstructed later and are written here: whether the tree was
clean, the producing commit, and the artifact's checksum. Promotion REFUSES when the
task-side copy and an existing promoted copy differ -- an artifact must not change under
its claim.

Usage (rung 0):
    uv run python scripts/promote_result.py \
        --task rung0-replicate-ceiling \
        --result docs/tasks/rung0-replicate-ceiling/rung0_delta_reproducibility.csv \
        --script scripts/delta_reproducibility.py \
        --input <panel file copy> --input <cid file> \
        --seed 0 --data-commit <tranche content_hash> \
        --arg tranche_id=tahoe100m-pseudobulk-de.v1 --job-id <slurm id> \
        --log results/rung0-replicate-ceiling/<job log>
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fmharness.schema import EnvironmentSnapshot, PromotedResult  # noqa: E402


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def promote(
    *,
    task: str,
    result: Path,
    script: str,
    inputs: list[Path],
    seed: int,
    data_commit: str,
    args: dict[str, str],
    job_id: str | None,
    log: Path | None,
    repo: Path,
) -> Path:
    repo = repo.resolve()
    if not (repo / script).exists():
        raise SystemExit(
            f"--script {script} is not in the repo; a result whose producing script "
            "cannot be found cannot be regenerated"
        )
    if not inputs:
        raise SystemExit(
            "at least one --input is required; a result with no recorded inputs cannot "
            "be checked against a rerun"
        )
    missing = [p for p in inputs if not p.exists()]
    if missing:
        raise SystemExit(f"declared inputs not found: {[str(p) for p in missing]}")

    out_dir = repo / "results" / task
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / result.name
    src_hash = sha256_of(result)
    if dest.exists() and sha256_of(dest) != src_hash:
        raise SystemExit(
            f"refusing: {dest.relative_to(repo)} exists and its checksum differs from "
            f"{result} -- the promoted copy and the task-side copy differ"
        )
    dest.write_bytes(result.read_bytes())

    record = PromotedResult(
        result=str(dest.relative_to(repo)),
        result_sha256=sha256_of(dest),
        task=task,
        script=script,
        args={k: str(v) for k, v in args.items()},
        inputs={str(p): sha256_of(p) for p in inputs},
        log=str(log) if log else None,
        log_sha256=sha256_of(log) if log else None,
        job_id=job_id,
        clean_tree=_git(repo, "status", "--porcelain") == "",
        environment=EnvironmentSnapshot(
            code_commit=_git(repo, "rev-parse", "HEAD"),
            python_version=platform.python_version(),
            seed=seed,
            cuda_deterministic=False,
            data_commit=data_commit,
        ),
        promoted_at=datetime.now(UTC),
    )
    record_path = dest.with_suffix(".provenance.json")
    record_path.write_text(record.model_dump_json(indent=2) + "\n")
    print(f"promoted -> {dest.relative_to(repo)}")
    print(f"           {record_path.relative_to(repo)}")
    return record_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--result", required=True, type=Path)
    ap.add_argument("--script", required=True)
    ap.add_argument("--input", action="append", default=[], type=Path, dest="inputs")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-commit", required=True)
    ap.add_argument("--arg", action="append", default=[], help="key=value, repeatable")
    ap.add_argument("--job-id", default=None)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ns = ap.parse_args()
    promote(
        task=ns.task,
        result=ns.result,
        script=ns.script,
        inputs=ns.inputs,
        seed=ns.seed,
        data_commit=ns.data_commit,
        args=dict(kv.split("=", 1) for kv in ns.arg),
        job_id=ns.job_id,
        log=ns.log,
        repo=ns.repo,
    )


if __name__ == "__main__":
    main()
```

Note on the test fixture's `git add -A`: it runs inside a throwaway `tmp_path` repo the test itself creates — the never-`add -A` constraint protects this repository's working tree, which that fixture never touches.

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_promote_result.py -v`
Expected: all PASS. `test_rule_01_*` in `tests/test_project_rules.py` still skip (nothing promoted in this repo yet) — that is correct.

- [x] **Step 5: Full suite, commit**

```bash
uv run pytest -q
git add scripts/promote_result.py tests/test_promote_result.py
git commit -m "feat: promotion emits a schema-validated PromotedResult and refuses a changed artifact"
```

---

### Task 6: `register_tranche.py` — ingest the DE pool as an immutable tranche

**Files:**
- Create: `scripts/register_tranche.py`
- Create: `tests/test_register_tranche.py`

**Interfaces:**
- Consumes: `fmharness.schema.Tranche`; the HuggingFace download-cache metadata layout (`<data-dir>/.cache/huggingface/download/metadata/**/<shard>.metadata`, line 1 = dataset revision at download time, line 2 = per-file sha256 etag).
- Produces: CLI writing `data/tranches/<tranche-id>.json` (a `Tranche`) plus `<tranche-id>.manifest.txt` beside it; importable `shard_manifest`, `content_hash`, `read_download_metadata`, `register`.

- [x] **Step 1: Write the failing tests**

Create `tests/test_register_tranche.py`:

```python
"""Tranche ingestion controls: a stable content hash, corruption detection against the
download-time etags, and refusal to overwrite (a tranche is ingested once, then
immutable -- SPEC vocabulary; design 'Data and inputs')."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from fmharness.schema import Tranche

_SPEC = importlib.util.spec_from_file_location(
    "register_tranche", Path(__file__).resolve().parents[1] / "scripts" / "register_tranche.py"
)
rt = importlib.util.module_from_spec(_SPEC)
sys.modules["register_tranche"] = rt
_SPEC.loader.exec_module(rt)

CONFIG = "pseudobulk_differential_expression"


def _fixture_pool(tmp: Path, contents: dict[str, bytes]) -> Path:
    data = tmp / "pool"
    shard_dir = data / CONFIG
    meta_dir = data / ".cache" / "huggingface" / "download" / "metadata" / CONFIG
    shard_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    for name, blob in contents.items():
        (shard_dir / name).write_bytes(blob)
        etag = hashlib.sha256(blob).hexdigest()
        (meta_dir / f"{name}.metadata").write_text(f"deadbeef01\n{etag}\n1700000000.0\n")
    return data


def test_content_hash_is_stable_and_sensitive_to_content(tmp_path: Path) -> None:
    d1 = _fixture_pool(tmp_path / "a", {"s1.parquet": b"AAA", "s2.parquet": b"BBB"})
    d2 = _fixture_pool(tmp_path / "b", {"s1.parquet": b"AAA", "s2.parquet": b"BBB"})
    d3 = _fixture_pool(tmp_path / "c", {"s1.parquet": b"AAA", "s2.parquet": b"XXX"})
    h = rt.content_hash(rt.shard_manifest(d1, CONFIG))
    assert h == rt.content_hash(rt.shard_manifest(d2, CONFIG)), "same bytes, same hash"
    assert h != rt.content_hash(rt.shard_manifest(d3, CONFIG)), "changed bytes, changed hash"


def test_registration_cross_checks_the_download_etags(tmp_path: Path) -> None:
    data = _fixture_pool(tmp_path, {"s1.parquet": b"AAA"})
    (data / CONFIG / "s1.parquet").write_bytes(b"CORRUPTED")  # bytes drift after download
    with pytest.raises(SystemExit, match="etag"):
        rt.register(
            data_dir=data, config=CONFIG, tranche_id="t.v1", source="src",
            ingestion_date="2026-07-24", patient_count=0, sample_count=50,
            drug_count=32, description="d", out=tmp_path / "t.v1.json",
        )


def test_registration_writes_a_valid_tranche_and_refuses_overwrite(tmp_path: Path) -> None:
    data = _fixture_pool(tmp_path, {"s1.parquet": b"AAA", "s2.parquet": b"BBB"})
    out = tmp_path / "t.v1.json"
    rt.register(
        data_dir=data, config=CONFIG, tranche_id="t.v1", source="src",
        ingestion_date="2026-07-24", patient_count=0, sample_count=50,
        drug_count=32, description="d", out=out,
    )
    tr = Tranche.model_validate_json(out.read_text())
    assert tr.version == "deadbeef01", "version comes from the download-time revision"
    assert tr.content_hash == rt.content_hash(rt.shard_manifest(data, CONFIG))
    assert out.with_suffix(".manifest.txt").exists()
    with pytest.raises(SystemExit, match="immutable"):
        rt.register(
            data_dir=data, config=CONFIG, tranche_id="t.v1", source="src",
            ingestion_date="2026-07-24", patient_count=0, sample_count=50,
            drug_count=32, description="d", out=out,
        )
```

- [x] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_register_tranche.py -v` — expected: FAIL (script absent).

- [x] **Step 3: Implement**

Create `scripts/register_tranche.py`:

```python
"""Ingest a downloaded dataset directory as an immutable, content-hashed Tranche.

Re-hashes every shard and CROSS-CHECKS each against the sha256 etag the HuggingFace
download recorded at pull time, so corruption since download fails registration instead
of becoming provenance. The tranche's version is the dataset revision recorded at
download time (line 1 of any download-cache metadata file); the content hash is the
sha256 of the sorted "relpath\tsize\tsha256" manifest, written beside the record.

Ingested once, then immutable: registration refuses an existing record.

Alpine usage (rung 0), from the repo root:
    python scripts/register_tranche.py \
        --data-dir /scratch/alpine/$USER/tahoe_pseudobulk_de \
        --tranche-id tahoe100m-pseudobulk-de.v1 \
        --source tahoebio/Tahoe-100M:pseudobulk_differential_expression \
        --ingestion-date 2026-07-24 --sample-count 50 --drug-count 32 \
        --description "Tahoe-100M pseudobulk DE shards; see docs/DATA.md" \
        --out data/tranches/tahoe100m-pseudobulk-de.v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fmharness.schema import Tranche  # noqa: E402

META_SUBDIR = Path(".cache/huggingface/download/metadata")


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def shard_manifest(data_dir: Path, config: str) -> list[tuple[str, int, str]]:
    """(relative path, size, sha256) per shard, sorted by path; hashes computed now."""
    shards = sorted(p for p in data_dir.rglob("*.parquet") if config in str(p)
                    and META_SUBDIR.parts[0] not in p.parts)
    if not shards:
        raise SystemExit(f"no {config} parquet under {data_dir}")
    return [(str(p.relative_to(data_dir)), p.stat().st_size, sha256_of(p)) for p in shards]


def content_hash(manifest: list[tuple[str, int, str]]) -> str:
    text = "".join(f"{rel}\t{size}\t{sha}\n" for rel, size, sha in manifest)
    return hashlib.sha256(text.encode()).hexdigest()


def read_download_metadata(data_dir: Path, config: str) -> tuple[str, dict[str, str]]:
    """(download-time dataset revision, {shard filename: etag sha256})."""
    meta_files = sorted((data_dir / META_SUBDIR).rglob("*.metadata"))
    meta_files = [m for m in meta_files if config in str(m)]
    if not meta_files:
        raise SystemExit(f"no download metadata under {data_dir / META_SUBDIR}")
    revisions, etags = set(), {}
    for m in meta_files:
        lines = m.read_text().splitlines()
        revisions.add(lines[0].strip())
        etags[m.name.removesuffix(".metadata")] = lines[1].strip()
    if len(revisions) != 1:
        raise SystemExit(f"shards from more than one dataset revision: {sorted(revisions)}")
    return revisions.pop(), etags


def register(
    *,
    data_dir: Path,
    config: str,
    tranche_id: str,
    source: str,
    ingestion_date: str,
    patient_count: int,
    sample_count: int,
    drug_count: int,
    description: str,
    out: Path,
) -> Path:
    if out.exists():
        raise SystemExit(f"{out} exists; a tranche is ingested once, then immutable")
    version, etags = read_download_metadata(data_dir, config)
    manifest = shard_manifest(data_dir, config)
    mismatched = [
        rel for rel, _, sha in manifest
        if Path(rel).name in etags and etags[Path(rel).name] != sha
    ]
    if mismatched:
        raise SystemExit(
            f"{len(mismatched)} shard(s) no longer match their download-time etag "
            f"(first: {mismatched[0]}) -- refusing to register corrupted data"
        )
    tranche = Tranche(
        tranche_id=tranche_id,
        source=source,
        version=version,
        ingestion_date=date.fromisoformat(ingestion_date),
        patient_count=patient_count,
        sample_count=sample_count,
        drug_count=drug_count,
        content_hash=content_hash(manifest),
        description=description,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".manifest.txt").write_text(
        "".join(f"{rel}\t{size}\t{sha}\n" for rel, size, sha in manifest)
    )
    out.write_text(tranche.model_dump_json(indent=2) + "\n")
    print(f"registered {tranche_id}: {len(manifest)} shards, version {version}")
    print(f"content_hash {tranche.content_hash}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--config", default="pseudobulk_differential_expression")
    ap.add_argument("--tranche-id", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--ingestion-date", required=True, help="YYYY-MM-DD of the download")
    ap.add_argument("--patient-count", type=int, default=0)
    ap.add_argument("--sample-count", type=int, required=True)
    ap.add_argument("--drug-count", type=int, required=True)
    ap.add_argument("--description", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ns = ap.parse_args()
    register(
        data_dir=ns.data_dir, config=ns.config, tranche_id=ns.tranche_id,
        source=ns.source, ingestion_date=ns.ingestion_date,
        patient_count=ns.patient_count, sample_count=ns.sample_count,
        drug_count=ns.drug_count, description=ns.description, out=ns.out,
    )


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_register_tranche.py -v` — expected: all PASS.

- [x] **Step 5: Full suite, commit**

```bash
uv run pytest -q
git add scripts/register_tranche.py tests/test_register_tranche.py
git commit -m "feat: tranche ingestion with etag cross-check and immutability refusal"
```

---

### Task 7: Alpine plumbing — ralpine with `switch`, the download script, and the sbatch files

**Files:**
- Create: `scripts/alpine/ralpine` (ported + one new verb), `scripts/download_tahoe_pseudobulk_de.py`, `scripts/alpine/00_target_cids.sbatch` (ported verbatim), `scripts/alpine/01_pseudobulk_shortcut.sbatch` (ported, one command changed), `scripts/alpine/register_tranche.sbatch` (new), `scripts/alpine/delta_reproducibility.sbatch` (ported, adapted)

**Interfaces:**
- Produces: `ralpine switch <branch>` (fetch, `git switch`, ff to upstream); job scripts Task 8 submits. All sbatch scripts assume repo root as cwd and the proven PATH-based `stack` env activation (NOT `module load anaconda` — see the ported comments for the evidence).

- [x] **Step 1: Port ralpine and add `switch`**

```bash
mkdir -p scripts/alpine
git show origin/worktree-modular-harness-core:scripts/alpine/ralpine > scripts/alpine/ralpine
chmod +x scripts/alpine/ralpine
```

Insert a new case **immediately before** the `update)` case:

```bash
  switch)
    # Move the Alpine checkout to another origin branch, then fast-forward it. Added
    # 2026-08-28 for branch-per-task (PROCESS §4): `update` only fast-forwards the
    # CURRENT branch, so a new task's first deployment needs this once. `git switch`
    # refuses rather than clobbering local changes, same safety property as update;
    # --guess creates the local tracking branch from origin/<branch> when absent.
    [[ $# -eq 1 ]] || die "usage: ralpine switch <branch>"
    [[ "$1" =~ ^[A-Za-z0-9._/-]+$ ]] || die "refusing '$1': not a plausible branch name"
    remote_fixed "git -C $(printf '%q' "$REMOTE_ROOT") fetch --quiet origin && \
                  git -C $(printf '%q' "$REMOTE_ROOT") switch --guess $(printf '%q' "$1") && \
                  git -C $(printf '%q' "$REMOTE_ROOT") merge --ff-only @{u} && \
                  git -C $(printf '%q' "$REMOTE_ROOT") rev-parse --short HEAD"
    ;;
```

Also add `switch` to the usage block in the header comment (one line under `ralpine update`):
`#   ralpine switch <branch>        # git switch the Alpine checkout to an origin branch`

Run: `bash -n scripts/alpine/ralpine` — expected: no output (syntax ok).

- [x] **Step 2: Extract the download script**

Create `scripts/download_tahoe_pseudobulk_de.py`:

```python
"""One authenticated bulk download of the Tahoe-100M pseudobulk DE config to scratch.

Extracted from the archived lineage's ``build_tahoe_pseudobulk_deltas.py`` (branch
``rung0-replicate-ceiling-old-lineage`` on origin), whose delta-bundle aggregation
imports rung-1 machinery and arrives with rung 1. This half is rung 0's provenance
chain: it reproduces the 2026-07-24 pull recorded in docs/DATA.md.

The table is a flat 1,026-file shard set with no drug partition; run as ONE process and
authenticate first (``hf auth login``) so the pull is not rate-limited.

    python scripts/download_tahoe_pseudobulk_de.py --local-dir /scratch/alpine/$USER/tahoe_pseudobulk_de
"""

from __future__ import annotations

import argparse
from pathlib import Path

TAHOE = "tahoebio/Tahoe-100M"
DE = "pseudobulk_differential_expression"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local-dir", required=True, type=Path)
    args = ap.parse_args()

    local = args.local_dir
    existing = [p for p in local.rglob("*.parquet") if DE in str(p)] if local.exists() else []
    if existing:
        print(f"{len(existing)} {DE} parquet already under {local}; nothing to do")
        return
    from huggingface_hub import snapshot_download  # type: ignore  # Alpine-only

    print(f"downloading the {DE} config to {local} (one-time, authenticated) ...")
    snapshot_download(TAHOE, repo_type="dataset", allow_patterns=[f"*{DE}*"], local_dir=str(local))
    got = [p for p in local.rglob("*.parquet") if DE in str(p)]
    print(f"downloaded {len(got)} parquet shards")


if __name__ == "__main__":
    main()
```

- [x] **Step 3: Port 00 and 01**

```bash
git show rung0-replicate-ceiling-old-lineage:scripts/alpine/00_target_cids.sbatch > scripts/alpine/00_target_cids.sbatch
git show rung0-replicate-ceiling-old-lineage:scripts/alpine/01_pseudobulk_shortcut.sbatch > scripts/alpine/01_pseudobulk_shortcut.sbatch
```

In `01_pseudobulk_shortcut.sbatch`, replace the final command block

```bash
python scripts/build_tahoe_pseudobulk_deltas.py \
    --drugs-cid-file data/static/tahoe_target_cids.txt \
    --local-dir "/scratch/alpine/$USER/tahoe_pseudobulk_de" \
    --out-dir tahoe_deltas
```

with

```bash
python scripts/download_tahoe_pseudobulk_de.py \
    --local-dir "/scratch/alpine/$USER/tahoe_pseudobulk_de"
```

and add one line to its header comment: `# The archived lineage's variant also built the rung-1 delta bundle; that half arrives with rung 1.`

Note: `00_target_cids.sbatch` reads `data/static/gdsc2_auc_pubchem_cids.txt`, which is untracked on Alpine and not landed here. Neither 00 nor 01 is re-run by this task (their outputs exist and are pinned by hash/tranche); they land as the provenance chain. The GDSC2 CID list becomes a tracked input when rung 4 registers GDSC2 — note this in Task 9's verification.md.

- [x] **Step 4: The registration job**

Create `scripts/alpine/register_tranche.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=register-tranche
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# One-time: ingest the Tahoe pseudobulk DE pool as tranche tahoe100m-pseudobulk-de.v1.
# Re-hashes all 1,026 shards (~83G, IO-bound) and cross-checks each against the sha256
# etag recorded at download time, so corruption since 2026-07-24 fails loudly here.
# The record is pulled back and committed by hand -- see the task plan.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

# PATH-based env activation; `module load anaconda` fails in batch jobs (cause not
# established -- see ralpine's submit comment; do not assert one here).
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1

python -c "import pydantic" 2>/dev/null || pip install -q "pydantic>=2.7"

python scripts/register_tranche.py \
    --data-dir "/scratch/alpine/$USER/tahoe_pseudobulk_de" \
    --tranche-id tahoe100m-pseudobulk-de.v1 \
    --source "tahoebio/Tahoe-100M:pseudobulk_differential_expression" \
    --ingestion-date 2026-07-24 \
    --sample-count 50 --drug-count 32 \
    --description "Tahoe-100M pseudobulk DE shards on scratch; sample_count = cell lines (no patients in a cell-line corpus); drug_count = the declared drug panel. See docs/DATA.md." \
    --out data/tranches/tahoe100m-pseudobulk-de.v1.json
```

- [x] **Step 5: The measurement job**

```bash
git show rung0-replicate-ceiling-old-lineage:scripts/alpine/delta_reproducibility.sbatch > scripts/alpine/delta_reproducibility.sbatch
```

Edit the final command block to match the Task 4 CLI and add matplotlib to the dependency check line:

```bash
python -c "import duckdb, matplotlib" 2>/dev/null || pip install -q duckdb matplotlib

python scripts/delta_reproducibility.py \
    --local-dir "/scratch/alpine/$USER/tahoe_pseudobulk_de" \
    --drugs-cid-file data/static/tahoe_target_cids.txt \
    --panel-file "${PANEL_FILE:-results/rung1_panel/common_panel.txt}" \
    --out-dir rung0_outputs
```

(Everything else in the ported sbatch — partition/QoS, PATH-based env, `PYTHONUNBUFFERED`, HF token plumbing for the drug-name lookup — stays as ported; those comments carry evidence from real failures.)

- [x] **Step 6: Commit**

```bash
uv run pytest -q
git add scripts/alpine/ralpine scripts/download_tahoe_pseudobulk_de.py scripts/alpine/00_target_cids.sbatch scripts/alpine/01_pseudobulk_shortcut.sbatch scripts/alpine/register_tranche.sbatch scripts/alpine/delta_reproducibility.sbatch
git commit -m "feat: Alpine plumbing -- ralpine switch verb, extracted DE download, tranche and ceiling jobs"
```

---

### Task 8: Alpine execution — register the tranche, run the ceiling

This task runs against the cluster; each step verifies before acting (PROCESS §2). Standing permissions cover `ralpine submit/cancel/update` and commit/push on this branch.

- [x] **Step 1: Deploy, and bind the remote paths once**

```bash
git push
./scripts/alpine/ralpine switch rung0-replicate-ceiling
./scripts/alpine/ralpine rev     # expect: new branch HEAD == local HEAD, tree section may list untracked data

# The remote repo root (ralpine's REMOTE_ROOT) and the scratch root, bound once and
# reused below. `ralpine run` executes from the SSH session's home, and a `$USER` in a
# locally-built argument would expand to the LOCAL username, so both are resolved
# explicitly:
ROOT=$(awk -F'"' '/^REMOTE_ROOT=/{print $2}' scripts/alpine/ralpine | sed 's/^\${ALPINE_ROOT:-//; s/}$//')
SCRATCH="/scratch/alpine/$(./scripts/alpine/ralpine run whoami | tr -d '[:space:]')"
echo "ROOT=$ROOT"; echo "SCRATCH=$SCRATCH"
```

- [x] **Step 2: Verify job inputs exist**

```bash
./scripts/alpine/ralpine run ls "$SCRATCH/tahoe_pseudobulk_de/pseudobulk_differential_expression" | head -3
./scripts/alpine/ralpine run wc -l "$ROOT/results/rung1_panel/common_panel.txt" "$ROOT/data/static/tahoe_target_cids.txt"
```

Expected: parquet shards listed; panel 14121 lines; CID file 32 lines. Any of the three missing is a stop — rebuild via `01_pseudobulk_shortcut.sbatch` (pool) or stop and report (panel/CIDs), never submit into missing inputs.

- [x] **Step 3: Register the tranche**

```bash
./scripts/alpine/ralpine submit scripts/alpine/register_tranche.sbatch
./scripts/alpine/ralpine sq                      # watch until COMPLETED (~30-60 min, IO-bound)
./scripts/alpine/ralpine log register-tranche    # expect: "registered ... 1026 shards, version <hash>" and the content_hash
```

If registration fails on an etag mismatch: the pool is corrupt — stop, report, and re-download via `01_pseudobulk_shortcut.sbatch` before re-registering. Do not register a pool that fails its own integrity check.

- [x] **Step 4: Pull and commit the tranche record**

```bash
./scripts/alpine/ralpine pull "$ROOT/data/tranches/tahoe100m-pseudobulk-de.v1.json" data/tranches/tahoe100m-pseudobulk-de.v1.json
./scripts/alpine/ralpine pull "$ROOT/data/tranches/tahoe100m-pseudobulk-de.v1.manifest.txt" data/tranches/tahoe100m-pseudobulk-de.v1.manifest.txt
uv run python -c "from pathlib import Path; from fmharness.schema import Tranche; print(Tranche.model_validate_json(Path('data/tranches/tahoe100m-pseudobulk-de.v1.json').read_text()).content_hash)"
git add data/tranches/tahoe100m-pseudobulk-de.v1.json data/tranches/tahoe100m-pseudobulk-de.v1.manifest.txt
git commit -m "data: register the Tahoe pseudobulk DE pool as tranche tahoe100m-pseudobulk-de.v1"
git push
./scripts/alpine/ralpine update
```

(`$ROOT` is bound in Task 8 Step 1; the pull verb takes the full remote path.)

- [x] **Step 5: Submit the measurement**

```bash
./scripts/alpine/ralpine submit scripts/alpine/delta_reproducibility.sbatch
./scripts/alpine/ralpine sq
./scripts/alpine/ralpine log delta-repro         # once running: panel line count, pair count, then the summary block
```

Expected wall time ~2h. Sanity against the design's Expected result: `splithalf_mean_r` near 0.135, `null_diff_drug_mean_r` near 0.03, `p_vs_null` ≤ 0.001, n_pairs 1,600, ~13,886 panel genes present. A large departure is a stop-and-investigate, not a promote.

- [x] **Step 6: Pull outputs and log**

```bash
for f in rung0_delta_reproducibility.csv rung0_delta_reproducibility.params.json rung0_per_gene_reliability.csv rung0_pool_description.csv rung0_ceiling.png; do
  ./scripts/alpine/ralpine pull "$ROOT/rung0_outputs/$f" "docs/tasks/rung0-replicate-ceiling/$f"
done
mkdir -p results/rung0-replicate-ceiling
./scripts/alpine/ralpine pull "$ROOT/logs/delta-repro-<JOBID>.out" "results/rung0-replicate-ceiling/delta-repro-<JOBID>.out"
```

(The log lands in `results/<task-slug>/` per PROCESS §2.)

- [x] **Step 7: Commit the run outputs**

```bash
git add docs/tasks/rung0-replicate-ceiling/rung0_delta_reproducibility.csv docs/tasks/rung0-replicate-ceiling/rung0_delta_reproducibility.params.json docs/tasks/rung0-replicate-ceiling/rung0_per_gene_reliability.csv docs/tasks/rung0-replicate-ceiling/rung0_pool_description.csv docs/tasks/rung0-replicate-ceiling/rung0_ceiling.png results/rung0-replicate-ceiling/delta-repro-<JOBID>.out
git commit -m "run: rung-0 ceiling outputs and job log, job <JOBID>"
```

---

### Task 9: Promote, and move the documents that report the number

**Files:**
- Modify: `docs/STATE.md` (ladder row 0 with its three links), `README.md` (status), `docs/tasks/rung0-replicate-ceiling/design.md` (status/current-result note)
- Create: `results/rung0-replicate-ceiling/rung0_delta_reproducibility.csv` + `.provenance.json` (via the promote script), `docs/tasks/rung0-replicate-ceiling/verification.md`

- [x] **Step 1: Stage the promotion inputs**

The panel file is referenced on Alpine; promotion records its hash. Pull a temporary copy to hash it (not committed):

```bash
./scripts/alpine/ralpine pull "$ROOT/results/rung1_panel/common_panel.txt" /tmp/common_panel.txt
./scripts/alpine/ralpine pull "$ROOT/data/static/tahoe_target_cids.txt" /tmp/tahoe_target_cids.txt
```

(Use the session scratchpad instead of `/tmp` when available.)

- [x] **Step 2: Promote**

```bash
uv run python scripts/promote_result.py \
  --task rung0-replicate-ceiling \
  --result docs/tasks/rung0-replicate-ceiling/rung0_delta_reproducibility.csv \
  --script scripts/delta_reproducibility.py \
  --input /tmp/common_panel.txt --input /tmp/tahoe_target_cids.txt \
  --seed 0 \
  --data-commit "$(uv run python -c "from pathlib import Path; from fmharness.schema import Tranche; print(Tranche.model_validate_json(Path('data/tranches/tahoe100m-pseudobulk-de.v1.json').read_text()).content_hash)")" \
  --arg tranche_id=tahoe100m-pseudobulk-de.v1 \
  --arg panel_file="results/rung1_panel/common_panel.txt on Alpine" \
  --job-id <JOBID> \
  --log results/rung0-replicate-ceiling/delta-repro-<JOBID>.out
```

Expected: `promoted -> results/rung0-replicate-ceiling/rung0_delta_reproducibility.csv` plus the record path. Note: `clean_tree` will be `False` if any working-tree edit is pending — promote from a clean tree (commit doc edits first, promote, then commit the promotion).

- [x] **Step 3: Run the project-rule tests — they now bind**

Run: `uv run pytest tests/test_project_rules.py -v -m "step_promote or step_score or step_null or step_document"`
Expected: rule-1 tests now RUN (not skip) and PASS against the real record; rule-4 edge test now RUNS and PASSES (known-answer-marked tests exist).

- [x] **Step 4: Write `verification.md`**

Create `docs/tasks/rung0-replicate-ceiling/verification.md` with: the exact commands of Tasks 8–9 as actually run, the job id, the tail of the job log showing the summary block, the pool-description highlights (pairs scored, plates per half), the tercile-control values and whether they rose monotonically, both MDE columns with one sentence reading them, and the note that `data/static/gdsc2_auc_pubchem_cids.txt` (00's input) becomes a tracked input when rung 4 registers GDSC2. Every number cited must be copied from the pulled CSV, not from memory.

- [x] **Step 5: Move STATE, README, and the design status**

- `docs/STATE.md` rung-0 row → done, with the three links: spec (`docs/tasks/rung0-replicate-ceiling/design.md`), code (the branch's commits), outputs (`results/rung0-replicate-ceiling/rung0_delta_reproducibility.csv` + its provenance record). Update the "No results/ directory exists" line — it is no longer true.
- `README.md` Status section: rung 0 landed, the ceiling exists, higher rungs read against it. No result numbers in commit messages — numbers live in STATE/the CSV.
- `design.md`: add a dated "Current result" note pointing at the promoted artifact (numbers by reference, not restated).

- [x] **Step 6: Full suite, commit, push**

```bash
uv run pytest -q
git add results/rung0-replicate-ceiling/rung0_delta_reproducibility.csv results/rung0-replicate-ceiling/rung0_delta_reproducibility.provenance.json docs/tasks/rung0-replicate-ceiling/verification.md docs/STATE.md README.md docs/tasks/rung0-replicate-ceiling/design.md
git commit -m "promote: rung-0 replicate ceiling with provenance; STATE and README move with it"
git push
```

---

### Task 10: Review and close out

- [x] **Step 1: Self-review the diff** — run `/code-review` (or the superpowers:requesting-code-review flow) on the branch diff vs `project-docs`; record each finding and its disposition in `docs/tasks/rung0-replicate-ceiling/review.md`; fix what review surfaces, re-running the suite.
- [x] **Step 2: Verification gate** — superpowers:verification-before-completion: every "done" claim in `verification.md` backed by command output.
- [x] **Step 3: Open the PR** — superpowers:finishing-a-development-branch. Target the trunk (`project-docs` if still open, else `main`), title "Rung 0 — the replicate ceiling", body linking design.md, verification.md, and the promoted artifact. One functional area; CodeRabbit + one lab-member approval per Greene Lab standard. Confirm the target with Lucas before opening.

---

## Self-review (completed at plan time)

- **Spec coverage:** declared statistic (T3–T4), controls per step (T3–T4 tests), MDE (T2, T4), tranche + `data_commit` chain (T6, T8–T9), promotion refusal (T5), pool description measured (T4, T8), per-gene diagnostic unpromoted (T4, T8 outputs; not promoted in T9), panel pinned by hash at promotion (T9), provenance-chain scripts on branch (T1, T7), landed-references (T1 amendment; ported docstrings rewritten), STATE/README move with the number (T9).
- **Known deviation recorded:** `build_tahoe_pseudobulk_deltas.py` not ported (T1 amendment with dated design entry).
- **Type consistency:** `build_split_half_frame` / `score_split_half` / `stratified_null_draws` / `masked_rowwise_pearson` signatures match between Task 3's definitions and Task 4/8 consumers; `promote()` and `register()` keyword signatures match their tests.

## Post-hoc reconciliation (2026-08-28, drift audit)

All checkboxes above are ticked: every task in this plan ran. Six execution rulings and one
dependency addition landed during implementation without a dated note against this plan's text
(drift-audit finding 3 + 12); recorded here, each against what the plan showed, what shipped, and
why:

- **`spearman_brown`** — the plan's statistics section (Task 2) shows the Spearman-Brown lift
  computed inline as `2*r/(1+r)`; shipped `src/fmharness/statistics.py::spearman_brown` as a
  third, independently tested helper, consumed by `summarize()` (commit `5a4d038`) — so the lift
  is the same tested code wherever it is used, not reimplemented per call site.
- **Null sampler `replace=False`** — plan:543 shows `rng.choice(..., replace=avail.size < n_perm)`
  (falls back to sampling with replacement once a stratum runs short); shipped samples without
  replacement whenever `avail.size >= n_perm` regardless (commit `9b24cc0`) — drawing the same
  mismatched pair twice would inflate apparent null precision without adding information.
- **Column intersection in `score_split_half`** — plan:498 intersects only the pivot INDEX
  (patient, drug) across halves; shipped also intersects the pivot COLUMNS, i.e. the gene sets
  (commit `9b24cc0`) — `pivot_table` drops all-NaN columns per half independently, so two halves
  can carry different gene sets even after the row intersection, and a caller that skipped it
  could silently correlate misaligned genes.
- **`clean_tree` redefinition** — plan:1100 shows plain `git status --porcelain`, captured after
  the result is copied into `results/`; shipped uses `--porcelain -uno` (tracked files only),
  captured BEFORE the copy (commit `441b238`) — untracked task-folder scratch sits in the working
  tree throughout this task, so counting it would make `clean_tree` permanently `False`, and
  capturing after the write would count the promotion's own output as a dirty-tree signal. This
  changes what the promoted record's `clean_tree` field means relative to the plan's text.
- **Unverified-shard refusal in `register()`** — plan:1358 refuses only shards whose hash
  MISMATCHES a download-time etag; shipped also refuses shards with NO etag entry at all (commit
  `61060f3`) — an unlisted shard is not verified as unmodified, so silently registering it would
  be a weaker guarantee than the plan's text describes.
- **`--input LABEL=PATH`** — plan:1122 shows `--input` taking a bare path (`type=Path`); shipped
  parses `LABEL=PATH` so the promoted record's `inputs` dict is keyed by a durable label rather
  than an ephemeral scratch path (commit `4211e30`) — already noted forward-looking in
  `verification.md`.
- **`pyarrow>=15`** — the plan's Task 1 dependency block (`pyproject.toml`) lists `pydantic`,
  `numpy`, `pandas`, `duckdb`, `matplotlib`; the shipped `pyproject.toml` also carries
  `pyarrow>=15`, needed transitively for the parquet-writing path the fixture-based tests exercise
  — never added to the plan's text.

Per project rule 2, this is the record; the superseded code blocks above are left as originally
written rather than rewritten to match what shipped.
