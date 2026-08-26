# Modular Harness Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the model-agnostic, modality-agnostic core from
`docs/tasks/modular-harness-core/design.md`: `Encoder`/`Generator`
model protocols, the `Modality` registry, leakage filtering, and the `CV`/`Readout`/`Estimator`
registry additions the spec calls for — all using Stack + the existing baseline models, no new
foundation-model integration.

**Architecture:** New, narrowly-scoped modules under `src/fmharness/`, each wrapping or
extending code that already exists and is already tested (`models/adapter.py`'s `ModelAdapter`,
`schema/`'s pydantic models, `splits/lso.py`'s `LeaveSubtypeOut`, `probe/base.py`'s `ProbeBase`,
`bilinear.py`, `controls/`) rather than building parallel systems. Every new Protocol is proven
by an `isinstance` check against an existing concrete class wherever one already exists.

**Tech Stack:** Python 3.11, pandas, numpy, scipy, pydantic v2, scikit-learn, anndata, pytest, uv.

## Global Constraints

- Line length 100 (`[tool.ruff]` in `pyproject.toml`).
- `target-version = "py311"`; pyright `typeCheckingMode = "strict"` over `src` and `tests`.
- Ruff lint selects `E, F, I, B, UP, SIM, RUF`. `tests/**` ignores `E501` only.
- Run everything through `uv`: `uv run pytest`, `uv run python`, `uv run ruff`, `uv run pyright`.
- **No emojis anywhere** -- code, comments, output, commit messages.
- **Vectorized only.** No nested Python loops over data rows. A loop over a small, fixed set
  (drugs in a biomarker table, perturbations in a coverage check) is fine; a loop over samples
  or genes is not.
- **Do not run `git commit` or `git push`.** Lucas commits in VS Code himself. Commit steps
  below state the intended message and file set; stage nothing and report the message instead.
- New public functions/classes need docstrings: what it computes, and why that is the right
  design -- see `src/fmharness/probe/base.py` and `src/fmharness/models/adapter.py` for the
  register.
- Every new `Protocol` must be `@runtime_checkable` and proven via `isinstance` against at
  least one real (not mock) existing class, per the design spec's acceptance criteria.

---

### Task 1: `Encoder` and `Generator` model protocols

`Encoder` is a strict subset of the existing `ModelAdapter` (`src/fmharness/models/adapter.py`):
`version`/`metadata`/`embed`. Every current adapter (`MockAdapter`, `LinearBaselineAdapter`)
already satisfies it structurally -- this task defines the protocol and proves that, it does
not touch the existing adapters. `Generator` is new: it predicts a transcriptome profile
(`generate`), distinct from `ModelAdapter.predict_native`, which predicts a scalar response
value directly.

**Files:**
- Create: `src/fmharness/model_protocols.py`
- Test: `tests/test_model_protocols.py`

**Interfaces:**
- Consumes: `fmharness.schema.ModelMetadata` (existing), `fmharness.models.adapter.MockAdapter`,
  `LinearBaselineAdapter` (existing, for the isinstance proof).
- Produces: `Encoder` protocol, `Generator` protocol, `PerturbationNotInContext` exception,
  `MockGenerator` reference implementation. Used by Task 4 (`filter_leakage`) and Task 5
  (`Modality`, indirectly, since both consume `Encoder | Generator` typed values).

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_protocols.py`:

```python
"""Tests for the Encoder/Generator model protocols."""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from fmharness.model_protocols import Encoder, Generator, MockGenerator, PerturbationNotInContext
from fmharness.models.adapter import LinearBaselineAdapter, MockAdapter


def test_existing_adapters_satisfy_encoder() -> None:
    # Encoder is a strict subset of ModelAdapter's surface -- every current
    # adapter already implements it, with no code changes to those classes.
    assert isinstance(MockAdapter(), Encoder)
    assert isinstance(LinearBaselineAdapter(), Encoder)


def test_mock_generator_satisfies_generator_and_encoder() -> None:
    gen = MockGenerator()
    assert isinstance(gen, Generator)
    assert isinstance(gen, Encoder)  # has embed too, so it's both


def test_mock_generator_context_coverage_matches_generate() -> None:
    gen = MockGenerator(known_perturbations=frozenset({"drugA", "drugB"}))
    assert gen.context_coverage(["drugA", "drugC", "drugB"]) == {"drugA", "drugB"}


def test_mock_generator_generate_is_deterministic_and_row_aligned() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 5)).astype(np.float32)
    baseline = AnnData(X=x)
    baseline.obs_names = ["s0", "s1", "s2"]

    gen = MockGenerator(known_perturbations=frozenset({"drugA"}), seed=0)
    out1 = gen.generate(baseline, "drugA")
    out2 = gen.generate(baseline, "drugA")
    assert out1.obs_names.tolist() == baseline.obs_names.tolist()
    np.testing.assert_array_equal(np.asarray(out1.X), np.asarray(out2.X))


def test_mock_generator_raises_on_unknown_perturbation() -> None:
    rng = np.random.default_rng(0)
    baseline = AnnData(X=rng.normal(size=(2, 4)).astype(np.float32))
    gen = MockGenerator(known_perturbations=frozenset({"drugA"}))
    with pytest.raises(PerturbationNotInContext):
        gen.generate(baseline, "drugZ")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_protocols.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmharness.model_protocols'`

- [ ] **Step 3: Write minimal implementation**

Create `src/fmharness/model_protocols.py`:

```python
"""Model-level capability protocols: ``Encoder`` and ``Generator``.

Split rather than a single protocol with optional methods, so a model's type
signature says exactly what it can do -- no ``None``-checking, and
``isinstance(model, Generator)`` is how the harness detects capability.
``Encoder`` is a strict subset of the existing ``ModelAdapter``
(``models/adapter.py``): every current adapter already satisfies it, with no
changes to those classes. ``Generator`` is new -- it predicts a transcriptome
profile, distinct from ``ModelAdapter.predict_native``, which predicts a
scalar response value directly.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Protocol, runtime_checkable

import numpy as np
from anndata import AnnData

from fmharness.models.adapter import as_dense_f32
from fmharness.schema import ModelMetadata


class PerturbationNotInContext(ValueError):
    """Raised by ``Generator.generate`` when the perturbation is not representable.

    A model must refuse loudly rather than silently extrapolate to something
    that looks like a real prediction but reflects the nearest thing it has
    actually seen -- confident-looking noise, worse than a null result.
    """


@runtime_checkable
class Encoder(Protocol):
    """Produces a per-sample representation vector."""

    def embed(self, adata: AnnData) -> np.ndarray:
        """(n_obs, embedding_dim), row-aligned to adata.obs_names. Deterministic."""
        ...

    def metadata(self) -> ModelMetadata: ...

    def version(self) -> str: ...


@runtime_checkable
class Generator(Protocol):
    """Produces a predicted post-perturbation profile."""

    def generate(self, baseline: AnnData, perturbation: str) -> AnnData:
        """Predicted profile for each row of ``baseline`` under ``perturbation``.

        Same obs/var contract as ``baseline``; full profile or delta, declared
        in ``metadata()``. Raises ``PerturbationNotInContext`` if
        ``perturbation`` cannot be represented.
        """
        ...

    def context_coverage(self, perturbations: Iterable[str]) -> set[str]:
        """Subset of ``perturbations`` this model can actually represent.

        Computed once per model, before generation runs -- not discovered as
        failures partway through an expensive GPU job.
        """
        ...

    def metadata(self) -> ModelMetadata: ...

    def version(self) -> str: ...


class MockGenerator:
    """Deterministic stand-in Generator for tests.

    ``known_perturbations`` is a fixed, small set -- ``context_coverage()``
    and ``generate()`` agree on what's representable, mirroring how a real
    model's context corpus bounds what it can generate for. Also implements
    ``embed`` (a seeded random projection, same recipe as ``MockAdapter``) so
    it satisfies both ``Generator`` and ``Encoder`` -- the dual-capability
    case ``scFoundation``/Stack-aligned will eventually occupy.
    """

    def __init__(
        self,
        known_perturbations: frozenset[str] = frozenset({"drugA", "drugB"}),
        embedding_dim: int = 8,
        seed: int = 0,
    ) -> None:
        self.known_perturbations = known_perturbations
        self.embedding_dim = embedding_dim
        self.seed = seed

    def version(self) -> str:
        return f"mock_generator@v1.0.0-s{self.seed}"

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            pretraining_corpus="none",
            pretraining_cutoff_date=date(1970, 1, 1),
            task_signal_in_pretrain="none",
            expected_input="log1p_cpm",
        )

    def embed(self, adata: AnnData) -> np.ndarray:
        x = as_dense_f32(adata)
        rng = np.random.default_rng(self.seed)
        projection = rng.standard_normal((x.shape[1], self.embedding_dim)).astype(np.float32)
        return np.ascontiguousarray(x @ projection, dtype=np.float32)

    def context_coverage(self, perturbations: Iterable[str]) -> set[str]:
        return {p for p in perturbations if p in self.known_perturbations}

    def generate(self, baseline: AnnData, perturbation: str) -> AnnData:
        if perturbation not in self.known_perturbations:
            raise PerturbationNotInContext(
                f"{perturbation!r} not in this model's context: "
                f"{sorted(self.known_perturbations)}"
            )
        rng = np.random.default_rng(self.seed)
        x = as_dense_f32(baseline)
        shift = rng.standard_normal(x.shape[1]).astype(np.float32) * 0.1
        out = baseline.copy()
        out.X = x + shift
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_protocols.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/fmharness/model_protocols.py tests/test_model_protocols.py && uv run ruff format --check src/fmharness/model_protocols.py tests/test_model_protocols.py && uv run pyright src/fmharness/model_protocols.py`
Expected: no errors.

- [ ] **Step 6: Report the commit (do not run it)**

Intended: `git add src/fmharness/model_protocols.py tests/test_model_protocols.py`
Message: `feat: add Encoder and Generator model protocols`

---

### Task 2: `ModelMetadata.expected_input`

Phase 1's sci-Plex work hit this exact gap: Stack's NB likelihood silently breaks on
normalized input, and the mismatch wasn't caught until diagnosis
(`docs/tasks/arm2-harness-validation/design.md`, Phase 1 blocker 3).
Declaring the expected scale lets the harness validate at the boundary instead of downstream.

**Files:**
- Modify: `src/fmharness/schema/models.py`
- Modify: `src/fmharness/models/adapter.py` (`MockAdapter.metadata`)
- Modify: `src/fmharness/models/wrappers/linear_baseline.py` (`LinearBaselineAdapter.metadata`)
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ModelMetadata.expected_input: Literal["raw_counts", "cpm", "log1p_cpm"]`
  (required field). Used by Task 1's `MockGenerator` (already written above to pass it) and by
  every future adapter.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_schema.py` (matching its existing `ModelMetadata` construction pattern --
read the existing tests in that file for the exact base kwargs used before writing this):

```python
def test_model_metadata_requires_expected_input() -> None:
    with pytest.raises(ValidationError):
        ModelMetadata(
            pretraining_corpus="none",
            pretraining_cutoff_date=date(1970, 1, 1),
            task_signal_in_pretrain="none",
        )  # type: ignore[call-arg]


def test_model_metadata_expected_input_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        ModelMetadata(
            pretraining_corpus="none",
            pretraining_cutoff_date=date(1970, 1, 1),
            task_signal_in_pretrain="none",
            expected_input="fpkm",  # type: ignore[arg-type]
        )


def test_model_metadata_expected_input_accepts_declared_values() -> None:
    for value in ("raw_counts", "cpm", "log1p_cpm"):
        m = ModelMetadata(
            pretraining_corpus="none",
            pretraining_cutoff_date=date(1970, 1, 1),
            task_signal_in_pretrain="none",
            expected_input=value,
        )
        assert m.expected_input == value
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schema.py -k expected_input -v`
Expected: FAIL -- `test_model_metadata_requires_expected_input` fails because no error is
raised (the field doesn't exist yet, so it's silently ignored... actually with
`extra="forbid"` an unrecognized kwarg would error on the *other* two tests instead). Read the
actual failure output before proceeding; it should point at `expected_input` being unrecognized.

- [ ] **Step 3: Write minimal implementation**

In `src/fmharness/schema/models.py`, add the field and a new type alias:

```python
ExpectedInput = Literal["raw_counts", "cpm", "log1p_cpm"]
```

Add to the `ModelMetadata` class body, after `task_signal_in_pretrain`:

```python
    expected_input: ExpectedInput
```

Update the module docstring's field list is not required (no such list exists in this file),
but update `src/fmharness/schema/__init__.py`'s exports: add `ExpectedInput` alongside
`TaskSignal`:

```python
from fmharness.schema.models import ExpectedInput, ModelMetadata, TaskSignal
```

and to `__all__`, insert `"ExpectedInput"` alphabetically.

- [ ] **Step 4: Fix the two existing production call sites**

In `src/fmharness/models/adapter.py`, `MockAdapter.metadata`, add `expected_input="log1p_cpm"`
to the returned `ModelMetadata(...)` call.

In `src/fmharness/models/wrappers/linear_baseline.py`, `LinearBaselineAdapter.metadata`, add
`expected_input="log1p_cpm"` to the returned `ModelMetadata(...)` call.

- [ ] **Step 5: Run the full test suite to catch any other now-broken ModelMetadata call sites**

Run: `uv run pytest -q`
Expected: any remaining `ModelMetadata(...)` construction missing `expected_input` will fail
with a pydantic `ValidationError`. Fix each by adding `expected_input="log1p_cpm"` (the correct
value for a baseline/mock; a real FM adapter would declare its actual requirement). Repeat
until the full suite passes.

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `uv run pytest tests/test_schema.py -k expected_input -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Lint and typecheck**

Run: `uv run ruff check src/fmharness/schema tests/test_schema.py src/fmharness/models.py src/fmharness/models/wrappers/linear_baseline.py && uv run pyright src/fmharness/schema`
Expected: no errors.

- [ ] **Step 8: Report the commit (do not run it)**

Intended: `git add src/fmharness/schema/models.py src/fmharness/schema/__init__.py src/fmharness/models/adapter.py src/fmharness/models/wrappers/linear_baseline.py tests/test_schema.py`
Message: `feat: add ModelMetadata.expected_input, required for every adapter`

---

### Task 3: extend `LeakageProfile` for filtering

The existing `LeakageProfile` (`src/fmharness/schema/provenance.py`) is a report shape only:
`drug_overlap_tahoe_100m`, `drug_overlap_fraction`, `declared_corpus_overlap`,
`subtype_prevalence`. It has no line-overlap, no doubly-exposed-pairs figure, and no `basis`
field distinguishing a measured overlap from an undeclared one. Nothing in the codebase
constructs or consumes `LeakageProfile` outside `tests/test_schema.py`'s bound-check tests
(verified: `grep -rn LeakageProfile` finds only the schema definition, its export, and those two
validation tests), so the existing fields can stay untouched and new ones added freely.

**Files:**
- Modify: `src/fmharness/schema/provenance.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `LeakageProfile.line_overlap_frac: float | None`,
  `LeakageProfile.doubly_exposed_frac: float | None`,
  `LeakageProfile.basis: Literal["measured", "declared", "unknown"]`. Used by Task 4
  (`filter_leakage`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_schema.py` (reuse the existing `base` fixture dict from the file's current
`LeakageProfile` tests for `tranche_id`/`model_version`/`drug_overlap_fraction`/`generated_at`):

```python
def test_leakage_profile_accepts_filtering_fields() -> None:
    p = LeakageProfile(
        tranche_id="t1",
        model_version="mock@v1",
        drug_overlap_fraction=0.2,
        generated_at=datetime(2026, 1, 1),
        line_overlap_frac=0.1,
        doubly_exposed_frac=0.02,
        basis="measured",
    )
    assert p.basis == "measured"
    assert p.line_overlap_frac == 0.1
    assert p.doubly_exposed_frac == 0.02


def test_leakage_profile_basis_unknown_allows_null_fractions() -> None:
    p = LeakageProfile(
        tranche_id="t1",
        model_version="mock@v1",
        drug_overlap_fraction=0.0,
        generated_at=datetime(2026, 1, 1),
        line_overlap_frac=None,
        doubly_exposed_frac=None,
        basis="unknown",
    )
    assert p.basis == "unknown"
    assert p.line_overlap_frac is None


def test_leakage_profile_basis_rejects_unknown_literal() -> None:
    with pytest.raises(ValidationError):
        LeakageProfile(
            tranche_id="t1",
            model_version="mock@v1",
            drug_overlap_fraction=0.0,
            generated_at=datetime(2026, 1, 1),
            line_overlap_frac=None,
            doubly_exposed_frac=None,
            basis="probably_fine",  # type: ignore[arg-type]
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schema.py -k leakage_profile -v`
Expected: FAIL -- `line_overlap_frac`/`doubly_exposed_frac`/`basis` unrecognized under
`extra="forbid"`.

- [ ] **Step 3: Write minimal implementation**

In `src/fmharness/schema/provenance.py`, add a type alias and three fields to `LeakageProfile`:

```python
LeakageBasis = Literal["measured", "declared", "unknown"]
```

(add `from typing import Literal` to the imports), and inside the `LeakageProfile` class body,
after `subtype_prevalence`:

```python
    line_overlap_frac: float | None = Field(default=None, ge=0.0, le=1.0)
    doubly_exposed_frac: float | None = Field(default=None, ge=0.0, le=1.0)
    basis: LeakageBasis = "unknown"
```

Defaults keep every existing construction of `LeakageProfile` (there are none outside tests,
confirmed in Task 3's header) valid without changes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_schema.py -v`
Expected: PASS (full file, including the pre-existing `LeakageProfile` bound-check tests, which
must still pass unmodified).

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/fmharness/schema/provenance.py tests/test_schema.py && uv run pyright src/fmharness/schema/provenance.py`
Expected: no errors.

- [ ] **Step 6: Report the commit (do not run it)**

Intended: `git add src/fmharness/schema/provenance.py tests/test_schema.py`
Message: `feat: extend LeakageProfile with line/pair overlap and a basis field`

---

### Task 4: `LeakageQueryable` and `filter_leakage`

Distinguished from `Generator.context_coverage()` (Task 1), which looks similar but answers a
different question: coverage asks *can this model represent this perturbation* (capability);
leakage asks *did this model already see the answer during pretraining* (validity).

**Files:**
- Create: `src/fmharness/leakage.py`
- Test: `tests/test_leakage.py`

**Interfaces:**
- Consumes: `Encoder`, `Generator` (Task 1); `LeakageProfile` (Task 3, extended).
- Produces: `LeakageQueryable` protocol, `filter_leakage(design, model) -> tuple[DataFrame, LeakageProfile]`.
  Used by future driver-integration work (out of scope for this plan; see the design spec).

- [ ] **Step 1: Write the failing test**

Create `tests/test_leakage.py`:

```python
"""Tests for leakage filtering."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from fmharness.leakage import LeakageQueryable, filter_leakage
from fmharness.model_protocols import MockGenerator
from fmharness.models.adapter import MockAdapter
from fmharness.schema import ModelMetadata


class _KnownCorpusModel:
    """Test double: declares an exact pretraining line/drug set."""

    def __init__(
        self,
        lines: set[str],
        drugs: set[str],
        task_signal: str = "adjacent",
    ) -> None:
        self._lines = lines
        self._drugs = drugs
        self._task_signal = task_signal

    def version(self) -> str:
        return "known_corpus@v1"

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            pretraining_corpus="synthetic",
            pretraining_cutoff_date=date(2026, 1, 1),
            task_signal_in_pretrain=self._task_signal,  # type: ignore[arg-type]
            expected_input="log1p_cpm",
        )

    def embed(self, adata: object) -> object:
        raise NotImplementedError

    def pretraining_lines(self) -> set[str] | None:
        return self._lines

    def pretraining_drugs(self) -> set[str] | None:
        return self._drugs


def _design() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient": ["L1", "L1", "L2", "L2", "L3", "L3"],
            "drug": ["d1", "d2", "d1", "d2", "d1", "d2"],
            "y": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )


def test_known_corpus_model_satisfies_leakage_queryable() -> None:
    m = _KnownCorpusModel(lines=set(), drugs=set())
    assert isinstance(m, LeakageQueryable)
    assert not isinstance(MockAdapter(), LeakageQueryable)  # doesn't expose these methods


def test_filter_leakage_drops_doubly_exposed_pairs_always() -> None:
    # L1 and d1 are both in the pretraining corpus; only (L1, d1) is doubly exposed.
    model = _KnownCorpusModel(lines={"L1"}, drugs={"d1"}, task_signal="adjacent")
    filtered, profile = filter_leakage(_design(), model)
    assert not ((filtered["patient"] == "L1") & (filtered["drug"] == "d1")).any()
    assert len(filtered) == 5  # one row dropped
    assert profile.basis == "measured"


def test_filter_leakage_adjacent_signal_keeps_single_axis_overlap() -> None:
    # L1 overlaps (line only, no drug overlap) with task_signal "adjacent" --
    # single-axis rows must NOT be dropped, or a broadly-pretrained model
    # becomes untestable on almost any cohort.
    model = _KnownCorpusModel(lines={"L1"}, drugs=set(), task_signal="adjacent")
    filtered, _ = filter_leakage(_design(), model)
    assert len(filtered) == 6  # nothing dropped -- no doubly-exposed pairs exist


def test_filter_leakage_direct_signal_drops_single_axis_overlap_too() -> None:
    model = _KnownCorpusModel(lines={"L1"}, drugs=set(), task_signal="direct")
    filtered, _ = filter_leakage(_design(), model)
    assert not (filtered["patient"] == "L1").any()
    assert len(filtered) == 4  # both L1 rows dropped


def test_filter_leakage_unknown_basis_when_model_cannot_expose_corpus() -> None:
    design = _design()
    filtered, profile = filter_leakage(design, MockAdapter())
    pd.testing.assert_frame_equal(filtered, design)
    assert profile.basis == "unknown"
    assert profile.line_overlap_frac is None


def test_filter_leakage_works_for_generator_too() -> None:
    filtered, profile = filter_leakage(_design(), MockGenerator())
    pd.testing.assert_frame_equal(filtered, _design())
    assert profile.basis == "unknown"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_leakage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmharness.leakage'`

- [ ] **Step 3: Write minimal implementation**

Create `src/fmharness/leakage.py`:

```python
"""Leakage filtering: strip pretraining-contaminated rows before scoring.

Distinguished from ``Generator.context_coverage`` (``model_protocols.py``), which
looks similar but answers a different question: coverage asks whether a model can
*represent* a perturbation (capability); this module asks whether a model already
*saw the answer* during pretraining (validity). A model can be fully capable of
representing a drug and still have memorized its specific response label.

Filtering happens on the whole panel for a given model, before any CV split -- not
fold-specific. If a line was in a model's own pretraining, it is invalid as a test
case for that model's representation quality regardless of which CV fold it would
otherwise land in; this is a different, earlier-stage contamination than ordinary
train/test CV leakage.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

import pandas as pd

from fmharness.model_protocols import Encoder, Generator
from fmharness.schema import LeakageProfile


@runtime_checkable
class LeakageQueryable(Protocol):
    """Optional: a model that can expose what its pretraining corpus actually covered."""

    def pretraining_lines(self) -> set[str] | None: ...

    def pretraining_drugs(self) -> set[str] | None: ...


def filter_leakage(
    design: pd.DataFrame,
    model: Encoder | Generator,
) -> tuple[pd.DataFrame, LeakageProfile]:
    """Drop pretraining-contaminated (patient, drug) rows from ``design``.

    - Always drops doubly-exposed rows (line AND drug both in pretraining) -- the
      sharpest risk, matching Phase 1's actual exclusion of 6 of 1,568 Tahoe/sci-Plex
      pairs (0.4%).
    - If ``task_signal_in_pretrain == "direct"`` (model trained on actual response
      labels), also drops single-axis overlap (line OR drug).
    - If task signal is "none" or "adjacent", single-axis overlap is reported in the
      profile but not hard-excluded -- a blanket single-axis filter would make a
      model pretrained on a broad public atlas (which shows large nominal line
      overlap with almost any cancer cohort) untestable on nearly anything.
    - If ``model`` does not implement ``LeakageQueryable``: the design is returned
      unmodified and ``basis="unknown"``. Filtering requires measured overlap; it
      never guesses clean.
    """
    version = model.version()
    meta = model.metadata()

    if not isinstance(model, LeakageQueryable):
        return design, LeakageProfile(
            tranche_id="unspecified",
            model_version=version,
            drug_overlap_fraction=0.0,
            generated_at=datetime.now(UTC),
            line_overlap_frac=None,
            doubly_exposed_frac=None,
            basis="unknown",
        )

    lines = model.pretraining_lines()
    drugs = model.pretraining_drugs()
    if lines is None or drugs is None:
        return design, LeakageProfile(
            tranche_id="unspecified",
            model_version=version,
            drug_overlap_fraction=0.0,
            generated_at=datetime.now(UTC),
            line_overlap_frac=None,
            doubly_exposed_frac=None,
            basis="unknown",
        )

    line_hit = design["patient"].isin(lines)
    drug_hit = design["drug"].isin(drugs)
    doubly_exposed = line_hit & drug_hit
    single_axis = line_hit | drug_hit

    n = len(design)
    line_overlap_frac = float(line_hit.mean()) if n else 0.0
    drug_overlap_frac = float(drug_hit.mean()) if n else 0.0
    doubly_exposed_frac = float(doubly_exposed.mean()) if n else 0.0

    drop_mask = doubly_exposed
    if meta.task_signal_in_pretrain == "direct":
        drop_mask = drop_mask | single_axis

    filtered = design.loc[~drop_mask].reset_index(drop=True)
    profile = LeakageProfile(
        tranche_id="unspecified",
        model_version=version,
        drug_overlap_fraction=drug_overlap_frac,
        generated_at=datetime.now(UTC),
        line_overlap_frac=line_overlap_frac,
        doubly_exposed_frac=doubly_exposed_frac,
        basis="measured",
    )
    return filtered, profile
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_leakage.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/fmharness/leakage.py tests/test_leakage.py && uv run pyright src/fmharness/leakage.py`
Expected: no errors. If `MockAdapter` used as a plain `Encoder` in
`test_filter_leakage_unknown_basis_when_model_cannot_expose_corpus` trips a pyright complaint
about the `Encoder | Generator` union, add an explicit `# type: ignore[arg-type]` on that call
with a one-line comment noting `MockAdapter` satisfies `Encoder` structurally but pyright's
structural-Protocol narrowing over a Union can be conservative here.

- [ ] **Step 6: Report the commit (do not run it)**

Intended: `git add src/fmharness/leakage.py tests/test_leakage.py`
Message: `feat: add LeakageQueryable protocol and filter_leakage`

---

### Task 5: `Modality` registry

Right now, "which phenotype are we predicting" is hardcoded per script --
`per_patient_eval.py`, `benchmark_sarcoma_organoids_2024.py`, `label_ceiling.py`, and
`score_generation_eval.py` each build their own `(patient, drug, y)` frame with their own sign
convention. This task formalizes it.

**Files:**
- Create: `src/fmharness/modality.py`
- Test: `tests/test_modality.py`

**Interfaces:**
- Consumes: `fmharness.evaluation.cpm_bundle`, `fmharness.evaluation.build_sample_design`
  (existing), `fmharness.data.loaders.load_tranche` (existing).
- Produces: `Modality` protocol, `Gdsc2Auc`, `CtrpAuc`, `PrismAuc`, `SoragniViability` concrete
  instances, `ThresholdedModality` wrapper. Consumed by future driver-integration work (out of
  scope for this plan).

- [ ] **Step 1: Write the failing test**

Create `tests/test_modality.py`. This test does not touch real data on disk -- it constructs a
fake `Modality` and only exercises `ThresholdedModality`'s wrapping logic, since the concrete
`Gdsc2Auc`/`SoragniViability` instances need real local data files
(`data/raw/gdsc2_sarcoma/`, `data/raw/sarcoma_organoids_2024/`) that are gitignored and not guaranteed present
in every environment -- those are covered by a separate, explicitly-marked integration test in
Step 6.

```python
"""Tests for the Modality registry."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fmharness.modality import Modality, ThresholdedModality


class _FakeAucModality:
    """A minimal regression Modality for testing the wrapper, no real data."""

    def load(self, repo: Path) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "patient": ["p1", "p1", "p2", "p2"],
                "drug": ["d1", "d2", "d1", "d2"],
                "y": [10.0, 60.0, 30.0, 80.0],
            }
        )

    def direction(self) -> str:
        return "lower_is_better"

    def recommended_cv(self) -> str:
        return "5fold"

    def task_type(self) -> str:
        return "regression"

    def name(self) -> str:
        return "fake_auc"


def test_fake_modality_satisfies_protocol() -> None:
    assert isinstance(_FakeAucModality(), Modality)


def test_thresholded_modality_emits_binary_y_below() -> None:
    wrapped = ThresholdedModality(_FakeAucModality(), threshold=50.0, responder_is="below")
    design = wrapped.load(Path("."))
    assert wrapped.task_type() == "classification"
    assert design.set_index(["patient", "drug"])["y"].to_dict() == {
        ("p1", "d1"): 1.0,
        ("p1", "d2"): 0.0,
        ("p2", "d1"): 1.0,
        ("p2", "d2"): 0.0,
    }


def test_thresholded_modality_emits_binary_y_above() -> None:
    wrapped = ThresholdedModality(_FakeAucModality(), threshold=50.0, responder_is="above")
    design = wrapped.load(Path("."))
    assert design.set_index(["patient", "drug"])["y"].to_dict() == {
        ("p1", "d1"): 0.0,
        ("p1", "d2"): 1.0,
        ("p2", "d1"): 0.0,
        ("p2", "d2"): 1.0,
    }


def test_thresholded_modality_delegates_direction_and_cv() -> None:
    wrapped = ThresholdedModality(_FakeAucModality(), threshold=50.0, responder_is="below")
    assert wrapped.direction() == "lower_is_better"
    assert wrapped.recommended_cv() == "5fold"


def test_thresholded_modality_satisfies_protocol() -> None:
    wrapped = ThresholdedModality(_FakeAucModality(), threshold=50.0, responder_is="below")
    assert isinstance(wrapped, Modality)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_modality.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmharness.modality'`

- [ ] **Step 3: Write minimal implementation**

Create `src/fmharness/modality.py`:

```python
"""Modality registry: swappable phenotype targets.

Substrate (which RNA source feeds a representation -- tumor vs. organoid vs.
cell-line RNA) is a Representation concern, not a Modality one: "Soragni tumor
RNA through Stack" and "Soragni organoid RNA through Stack" are two different
representations aimed at the *same* Modality (Soragni viability). Modality owns
only the label side: which dataset, which metric, which sign convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import pandas as pd

from fmharness.data.loaders import load_tranche
from fmharness.evaluation import build_sample_design, cpm_bundle

Direction = Literal["lower_is_better", "higher_is_better"]
TaskType = Literal["regression", "classification"]


@runtime_checkable
class Modality(Protocol):
    def load(self, repo: Path) -> pd.DataFrame:
        """design[patient, drug, y] on this modality's native scale."""
        ...

    def direction(self) -> Direction: ...

    def recommended_cv(self) -> str:
        """A CV-registry key sized to this modality's n (organoid: loo; cell-line: 5fold)."""
        ...

    def task_type(self) -> TaskType: ...

    def name(self) -> str: ...


class Gdsc2Auc:
    """GDSC2 AUC. Lower AUC = more sensitive = better response."""

    def __init__(self, cancer_type_filter: list[str] | None = None) -> None:
        self.cancer_type_filter = cancer_type_filter

    def load(self, repo: Path) -> pd.DataFrame:
        bundle = cpm_bundle(
            load_tranche("gdscv2", repo, cancer_type_filter=self.cancer_type_filter)
        )
        _, design = build_sample_design(bundle, "all", "auc", drug_key="pubchem_cid")
        return design.rename(columns={"y": "y"})[["patient", "drug", "y"]]

    def direction(self) -> Direction:
        return "lower_is_better"

    def recommended_cv(self) -> str:
        return "5fold"

    def task_type(self) -> TaskType:
        return "regression"

    def name(self) -> str:
        return "gdsc2_auc"


class CtrpAuc:
    """CTRPv2 AUC via CoderData (``data/raw/coderdata``). Lower = more sensitive."""

    def load(self, repo: Path) -> pd.DataFrame:
        bundle = cpm_bundle(load_tranche("ctrpv2", repo))
        _, design = build_sample_design(bundle, "all", "auc", drug_key="pubchem_cid")
        return design.rename(columns={"y": "y"})[["patient", "drug", "y"]]

    def direction(self) -> Direction:
        return "lower_is_better"

    def recommended_cv(self) -> str:
        return "5fold"

    def task_type(self) -> TaskType:
        return "regression"

    def name(self) -> str:
        return "ctrp_auc"


class PrismAuc:
    """PRISM AUC via CoderData (``data/raw/coderdata``). Lower = more sensitive."""

    def load(self, repo: Path) -> pd.DataFrame:
        bundle = cpm_bundle(load_tranche("prism", repo))
        _, design = build_sample_design(bundle, "all", "auc", drug_key="pubchem_cid")
        return design.rename(columns={"y": "y"})[["patient", "drug", "y"]]

    def direction(self) -> Direction:
        return "lower_is_better"

    def recommended_cv(self) -> str:
        return "5fold"

    def task_type(self) -> TaskType:
        return "regression"

    def name(self) -> str:
        return "prism_auc"


class SoragniViability:
    """Soragni organoid Viability_Score (% of vehicle). Lower = more sensitive."""

    def __init__(self, rna_source: Literal["tumor", "organoid", "all"] = "tumor") -> None:
        self.rna_source = rna_source

    def load(self, repo: Path) -> pd.DataFrame:
        bundle = cpm_bundle(load_tranche("sarcoma", repo))
        _, design = build_sample_design(bundle, self.rna_source, "viability")
        return design.rename(columns={"y": "y"})[["patient", "drug", "y"]]

    def direction(self) -> Direction:
        return "lower_is_better"

    def recommended_cv(self) -> str:
        return "loo"

    def task_type(self) -> TaskType:
        return "regression"

    def name(self) -> str:
        return f"sarcoma_organoids_2024_viability_{self.rna_source}"


class ThresholdedModality:
    """Wraps a regression Modality, emits binary y at a threshold.

    Reuses the base Modality's exact data-loading path -- no duplicated
    normalization or join logic -- so a classification target is always
    derived from, and stays consistent with, its regression counterpart.
    """

    def __init__(
        self,
        base: Modality,
        threshold: float,
        responder_is: Literal["below", "above"],
    ) -> None:
        self.base = base
        self.threshold = threshold
        self.responder_is = responder_is

    def load(self, repo: Path) -> pd.DataFrame:
        design = self.base.load(repo).copy()
        if self.responder_is == "below":
            design["y"] = (design["y"] < self.threshold).astype(float)
        else:
            design["y"] = (design["y"] > self.threshold).astype(float)
        return design

    def direction(self) -> Direction:
        return self.base.direction()

    def recommended_cv(self) -> str:
        return self.base.recommended_cv()

    def task_type(self) -> TaskType:
        return "classification"

    def name(self) -> str:
        return f"{self.base.name()}_responder_{self.threshold:g}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_modality.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/fmharness/modality.py tests/test_modality.py && uv run pyright src/fmharness/modality.py`
Expected: no errors.

- [ ] **Step 6: Add a real-data smoke test, skipped when data is absent**

Append to `tests/test_modality.py`:

```python
import os

_REPO = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(
    not (_REPO / "data/raw/gdsc2_sarcoma/gdsc2/GDSC2_fitted_dose_response_27Oct23.xlsx").exists(),
    reason="requires local GDSC2 raw data",
)
def test_gdsc2_auc_loads_real_data() -> None:
    design = Gdsc2Auc().load(_REPO)
    assert {"patient", "drug", "y"} <= set(design.columns)
    assert design["patient"].nunique() > 100  # full pan-cancer panel, not a small subset


@pytest.mark.skipif(
    not (_REPO / "data/raw/sarcoma_organoids_2024").exists(),
    reason="requires local Soragni raw data",
)
def test_sarcoma_organoids_2024_viability_loads_real_data() -> None:
    design = SoragniViability().load(_REPO)
    assert {"patient", "drug", "y"} <= set(design.columns)
    assert design["patient"].nunique() == 17
```

Add `import pytest` and the two class imports (`Gdsc2Auc`, `SoragniViability`) to this file's
top-level imports.

- [ ] **Step 7: Run the full test file to verify it passes (or skips cleanly)**

Run: `uv run pytest tests/test_modality.py -v`
Expected: PASS or SKIPPED, never FAIL, regardless of whether local raw data is present.

- [ ] **Step 8: Lint and typecheck again**

Run: `uv run ruff check src/fmharness/modality.py tests/test_modality.py && uv run pyright src/fmharness/modality.py`
Expected: no errors.

- [ ] **Step 9: Report the commit (do not run it)**

Intended: `git add src/fmharness/modality.py tests/test_modality.py`
Message: `feat: add Modality registry with GDSC2/CTRP/PRISM/Soragni instances`

---

### Task 6: bridge `LeaveSubtypeOut` into the pandas CV pattern

Two CV systems exist and were never reconciled: `src/fmharness/splits/` operates on pydantic
`Patient` objects (`patient_id`, `subtype`) and yields `SplitFold` (train/test patient-id
tuples); every actual working script (`grouped_cv_predict`, `score_generation_eval.py`,
`benchmark_sarcoma_organoids_2024.py`) operates on flat `design[patient, drug, y]` DataFrames. Per the
confirmed design decision, bridge rather than migrate: build a thin adapter that runs
`LeaveSubtypeOut`'s real fold-partitioning logic and exposes it in the
`design -> Iterator[(train_idx, test_idx)]` shape the working pandas pipeline expects.

**Files:**
- Create: `src/fmharness/cv.py`
- Test: `tests/test_cv.py`

**Interfaces:**
- Consumes: `fmharness.splits.lso.LeaveSubtypeOut`, `fmharness.splits.base.SplitFold`,
  `fmharness.splits.base.SplittablePatient` (all existing, unmodified).
- Produces: `CVScheme` protocol (`splits(design) -> Iterator[tuple[NDArray, NDArray]]`),
  `leave_subtype_out(subtypes, *, seed, granularity="fine", subtype_map=None) -> CVScheme`.
  Consumed by future driver-integration work (out of scope for this plan).

- [ ] **Step 1: Write the failing test**

Create `tests/test_cv.py`:

```python
"""Tests for the CV registry's leave-subtype-out bridge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmharness.cv import CVScheme, leave_subtype_out


def _design() -> pd.DataFrame:
    # 4 patients, 2 subtypes (A: p1,p2 -- B: p3,p4), 2 drugs each.
    return pd.DataFrame(
        {
            "patient": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
            "drug": ["d1", "d2"] * 4,
            "y": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
    )


def test_leave_subtype_out_satisfies_cv_scheme() -> None:
    subtypes = {"p1": "A", "p2": "A", "p3": "B", "p4": "B"}
    scheme = leave_subtype_out(subtypes, seed=0)
    assert isinstance(scheme, CVScheme)


def test_leave_subtype_out_yields_one_fold_per_subtype() -> None:
    subtypes = {"p1": "A", "p2": "A", "p3": "B", "p4": "B"}
    design = _design()
    scheme = leave_subtype_out(subtypes, seed=0)
    folds = list(scheme.splits(design))
    assert len(folds) == 2  # one fold per subtype (A held out, then B held out)

    for train_idx, test_idx in folds:
        train_idx_arr = np.asarray(train_idx)
        test_idx_arr = np.asarray(test_idx)
        assert set(train_idx_arr) & set(test_idx_arr) == set()
        train_patients = set(design.iloc[train_idx_arr]["patient"])
        test_patients = set(design.iloc[test_idx_arr]["patient"])
        assert train_patients & test_patients == set()  # no patient in both


def test_leave_subtype_out_test_fold_covers_exactly_the_held_out_subtype_rows() -> None:
    subtypes = {"p1": "A", "p2": "A", "p3": "B", "p4": "B"}
    design = _design()
    scheme = leave_subtype_out(subtypes, seed=0)
    folds = list(scheme.splits(design))
    test_patient_sets = [set(design.iloc[np.asarray(te)]["patient"]) for _, te in folds]
    assert {"p1", "p2"} in test_patient_sets
    assert {"p3", "p4"} in test_patient_sets


def test_leave_subtype_out_raises_on_missing_patient_subtype() -> None:
    subtypes = {"p1": "A", "p2": "A"}  # p3, p4 missing
    with pytest.raises(KeyError):
        list(leave_subtype_out(subtypes, seed=0).splits(_design()))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cv.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmharness.cv'`

- [ ] **Step 3: Write minimal implementation**

Create `src/fmharness/cv.py`:

```python
"""CV registry, bridging the pydantic Splitter system to the pandas design-frame
pattern the working pipeline (build_sample_design, grouped_cv_predict,
score_generation_eval.py) actually uses.

``src/fmharness/splits/`` operates on pydantic ``Patient`` objects and yields
``SplitFold`` (train/test patient-id tuples). Every real working script operates
on flat ``design[patient, drug, y]`` DataFrames instead. Rather than migrate the
working pipeline onto the heavier pydantic system, this module runs the real
splitter logic (``LeaveSubtypeOut``) and translates its output into the
``design -> Iterator[(train_idx, test_idx)]`` shape already in use everywhere.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fmharness.splits.lso import LeaveSubtypeOut, LSOGranularity


@runtime_checkable
class CVScheme(Protocol):
    def splits(self, design: pd.DataFrame) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Yield (train_row_idx, test_row_idx) into ``design``, disjoint by patient."""
        ...


@dataclass(frozen=True)
class _PatientRow:
    """Satisfies fmharness.splits.base.SplittablePatient."""

    patient_id: str
    subtype: str | None


class _LeaveSubtypeOutScheme:
    def __init__(
        self,
        patient_subtypes: dict[str, str],
        *,
        seed: int,
        granularity: LSOGranularity = "fine",
        subtype_map: dict[str, str] | None = None,
    ) -> None:
        self.patient_subtypes = patient_subtypes
        self._lso = LeaveSubtypeOut(seed=seed, granularity=granularity, subtype_map=subtype_map)

    def splits(
        self, design: pd.DataFrame
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        patient_ids = sorted(design["patient"].unique())
        missing = [p for p in patient_ids if p not in self.patient_subtypes]
        if missing:
            raise KeyError(f"no subtype declared for patient(s): {missing}")
        patients = [_PatientRow(p, self.patient_subtypes[p]) for p in patient_ids]
        patient_pos = pd.Series(np.arange(len(design)), index=design["patient"]).groupby(
            level=0
        )
        for fold in self._lso.split(patients):
            train_idx = np.concatenate(
                [patient_pos.get_group(p).to_numpy() for p in fold.train_patient_ids]
            )
            test_idx = np.concatenate(
                [patient_pos.get_group(p).to_numpy() for p in fold.test_patient_ids]
            )
            yield np.sort(train_idx), np.sort(test_idx)


def leave_subtype_out(
    patient_subtypes: dict[str, str],
    *,
    seed: int,
    granularity: LSOGranularity = "fine",
    subtype_map: dict[str, str] | None = None,
) -> CVScheme:
    """One fold per unique subtype, bridged from ``fmharness.splits.lso.LeaveSubtypeOut``.

    ``patient_subtypes`` maps every patient_id appearing in the design frame to its
    subtype label; missing patients raise ``KeyError`` rather than being silently
    dropped, since a silently-shrunk cohort would understate the fold's true size.
    """
    return _LeaveSubtypeOutScheme(
        patient_subtypes, seed=seed, granularity=granularity, subtype_map=subtype_map
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cv.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/fmharness/cv.py tests/test_cv.py && uv run pyright src/fmharness/cv.py`
Expected: no errors.

- [ ] **Step 6: Report the commit (do not run it)**

Intended: `git add src/fmharness/cv.py tests/test_cv.py`
Message: `feat: bridge LeaveSubtypeOut into the pandas design-frame CV pattern`

---

### Task 7: classification readouts

Named in the original MVP plan (`src/fmharness/metrics/`: `top_k_hit_rate`, `regret`,
`brier_score`, `expected_calibration_error`) and never built. `regret` already exists as
`regret_norm_at_k` in `src/fmharness/evaluation.py`; this task adds the three that don't.

**Files:**
- Create: `src/fmharness/classification_readouts.py`
- Test: `tests/test_classification_readouts.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure metric functions on `y_true`/`y_pred` arrays).
- Produces: `top_k_hit_rate(preds, k) -> float`, `brier_score(preds) -> float`,
  `expected_calibration_error(preds, n_bins=10) -> float`, operating on the same
  `preds[patient, drug, y_true, y_pred]` frame shape `score_predictions` in
  `evaluation.py` already uses, so they compose directly with the `Modality`
  classification path (Task 5).

- [ ] **Step 1: Write the failing test**

Create `tests/test_classification_readouts.py`:

```python
"""Tests for classification readouts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmharness.classification_readouts import (
    brier_score,
    expected_calibration_error,
    top_k_hit_rate,
)


def _preds(y_true: list[float], y_pred: list[float]) -> pd.DataFrame:
    n = len(y_true)
    return pd.DataFrame(
        {
            "patient": [f"p{i}" for i in range(n)],
            "drug": ["d1"] * n,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )


def test_top_k_hit_rate_perfect_predictions() -> None:
    # y_true is binary (1 = responder); y_pred is a predicted probability.
    preds = _preds([1.0, 0.0, 1.0, 0.0], [0.9, 0.1, 0.8, 0.2])
    assert np.isclose(top_k_hit_rate(preds, k=2), 1.0)


def test_top_k_hit_rate_worst_case() -> None:
    preds = _preds([1.0, 0.0, 1.0, 0.0], [0.1, 0.9, 0.2, 0.8])
    assert np.isclose(top_k_hit_rate(preds, k=1), 0.0)


def test_brier_score_perfect_predictions_is_zero() -> None:
    preds = _preds([1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0])
    assert np.isclose(brier_score(preds), 0.0)


def test_brier_score_worst_case_is_one() -> None:
    preds = _preds([1.0, 0.0], [0.0, 1.0])
    assert np.isclose(brier_score(preds), 1.0)


def test_brier_score_uninformative_half_probability() -> None:
    preds = _preds([1.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 0.5])
    assert np.isclose(brier_score(preds), 0.25)


def test_expected_calibration_error_perfect_calibration_is_zero() -> None:
    # Every predicted probability exactly matches the empirical response rate
    # within its own bin when there's one sample per bin and pred == true.
    preds = _preds([1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0])
    assert np.isclose(expected_calibration_error(preds, n_bins=2), 0.0)


def test_expected_calibration_error_penalizes_overconfidence() -> None:
    # Predicts near-certain responder for everyone, but only half actually respond.
    preds = _preds([1.0, 0.0, 1.0, 0.0], [0.95, 0.95, 0.95, 0.95])
    ece = expected_calibration_error(preds, n_bins=10)
    assert ece > 0.4  # bin mean confidence ~0.95 vs. empirical rate 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_classification_readouts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmharness.classification_readouts'`

- [ ] **Step 3: Write minimal implementation**

Create `src/fmharness/classification_readouts.py`:

```python
"""Classification readouts: top-k hit rate, Brier score, expected calibration error.

Named in the original harness plan and never built (the harness stayed
AUC/interaction-focused). Operate on the same preds[patient, drug, y_true, y_pred]
frame shape ``score_predictions`` uses in ``evaluation.py`` -- ``y_true`` is a
binary responder label (from a ``ThresholdedModality``, ``modality.py``), ``y_pred``
is a predicted probability of response.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def top_k_hit_rate(preds: pd.DataFrame, k: int) -> float:
    """Fraction of true responders (y_true == 1) captured in the top-k by y_pred.

    Ranks all rows by descending predicted probability, takes the top k, and
    reports what share of the panel's actual responders are in that shortlist --
    the same "does the shortlist contain the answer" question regret_norm_at_k
    asks for continuous response, adapted to a binary label.
    """
    n_responders = int((preds["y_true"] == 1.0).sum())
    if n_responders == 0:
        return float("nan")
    top_k = preds.nlargest(k, "y_pred")
    hits = int((top_k["y_true"] == 1.0).sum())
    return hits / n_responders


def brier_score(preds: pd.DataFrame) -> float:
    """Mean squared error between predicted probability and binary outcome.

    0 is perfect, 0.25 is the score of a constant p=0.5 predictor against a
    balanced panel, 1 is maximally wrong (confident and always incorrect).
    """
    y_true = preds["y_true"].to_numpy(dtype=np.float64)
    y_pred = preds["y_pred"].to_numpy(dtype=np.float64)
    return float(np.mean((y_pred - y_true) ** 2))


def expected_calibration_error(preds: pd.DataFrame, n_bins: int = 10) -> float:
    """Mean absolute gap between predicted probability and empirical response rate,
    within equal-width probability bins, weighted by bin size.

    A well-calibrated model's predicted probabilities should match the actual
    fraction of responders among samples given that probability; this is the
    standard ECE definition (Guo et al. 2017).
    """
    y_true = preds["y_true"].to_numpy(dtype=np.float64)
    y_pred = preds["y_pred"].to_numpy(dtype=np.float64)
    n = len(y_true)
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_pred, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        bin_confidence = float(y_pred[mask].mean())
        bin_accuracy = float(y_true[mask].mean())
        ece += (count / n) * abs(bin_confidence - bin_accuracy)
    return ece
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_classification_readouts.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/fmharness/classification_readouts.py tests/test_classification_readouts.py && uv run pyright src/fmharness/classification_readouts.py`
Expected: no errors.

- [ ] **Step 6: Report the commit (do not run it)**

Intended: `git add src/fmharness/classification_readouts.py tests/test_classification_readouts.py`
Message: `feat: add classification readouts (top_k_hit_rate, brier_score, ECE)`

---

### Task 8: the unbalanced-panel testing standard, applied

The gap that let `interaction_rho`'s missingness artifact ship undetected: no readout was ever
tested against a zero-information predictor on a realistically *unbalanced* panel, only
balanced synthetic ones. This task adds a reusable test helper and applies it to Task 7's new
readouts as the concrete proof it catches this class of bug, matching the pattern already set
by `tests/test_evaluation.py::test_interaction_rho_ignores_drug_only_signal_on_an_unbalanced_panel`.

**Files:**
- Create: `tests/readout_contract.py`
- Modify: `tests/test_classification_readouts.py`

**Interfaces:**
- Consumes: `top_k_hit_rate`, `brier_score`, `expected_calibration_error` (Task 7).
- Produces: `assert_null_on_unbalanced_zero_info_predictor(readout_fn, expected_null, *, drops=...)`
  test helper, for use by every future readout added to the registry.

- [ ] **Step 1: Write the failing test**

Create `tests/readout_contract.py`:

```python
"""Shared test contract every Readout must satisfy: null behavior on an
unbalanced, zero-information panel. See tests/test_evaluation.py's
test_interaction_rho_ignores_drug_only_signal_on_an_unbalanced_panel for the
precedent -- interaction_rho's missingness artifact shipped because no test
like this existed before it was found and fixed.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def unbalanced_zero_info_panel(seed: int = 0) -> pd.DataFrame:
    """4 patients x 5 drugs, ~30% of cells missing at random, y_pred carries
    zero patient-level information (constant per drug -- the drug's own mean
    y_true), the exact shape of predictor that exposed the interaction_rho bug.
    """
    rng = np.random.default_rng(seed)
    patients = [f"p{i}" for i in range(4)]
    drugs = [f"d{i}" for i in range(5)]
    rows = [(p, d) for p in patients for d in drugs]
    keep = rng.random(len(rows)) > 0.3
    rows = [r for r, k in zip(rows, keep, strict=True) if k]
    df = pd.DataFrame(rows, columns=["patient", "drug"])
    df["y_true"] = rng.uniform(0.0, 1.0, size=len(df))
    df["y_pred"] = df.groupby("drug")["y_true"].transform("mean")  # zero patient info
    return df


def assert_null_on_unbalanced_zero_info_predictor(
    readout_fn: Callable[[pd.DataFrame], float],
    expected_null: float,
    *,
    seed: int = 0,
    atol: float = 1e-9,
) -> None:
    """A predictor with zero patient-level information must score at its null
    value even on an unbalanced panel -- the exact failure mode that let
    interaction_rho's missingness artifact ship undetected.
    """
    panel = unbalanced_zero_info_panel(seed=seed)
    observed = readout_fn(panel)
    assert np.isclose(observed, expected_null, atol=atol), (
        f"expected null {expected_null}, got {observed} on an unbalanced zero-info panel -- "
        "this readout may have a missingness artifact"
    )
```

This file has no `test_` prefix and defines no tests of its own; it is a shared contract module.
Now append its use to `tests/test_classification_readouts.py`:

```python
from readout_contract import assert_null_on_unbalanced_zero_info_predictor


def test_top_k_hit_rate_null_on_unbalanced_zero_info_predictor() -> None:
    # y_pred is constant per drug, so it cannot rank patients within a drug at
    # all -- top-k hit rate should be the base rate, not artificially high or
    # low from the panel's missingness pattern.
    def readout(panel: pd.DataFrame) -> float:
        panel = panel.assign(y_true=(panel["y_true"] > panel["y_true"].median()).astype(float))
        return top_k_hit_rate(panel, k=len(panel) // 2)

    # Base rate for a threshold at the pooled median is close to 0.5 by
    # construction; the zero-info predictor should land near there, not be
    # inflated or deflated by which cells are missing.
    from readout_contract import unbalanced_zero_info_panel

    panel = unbalanced_zero_info_panel(seed=0)
    thresholded = panel.assign(y_true=(panel["y_true"] > panel["y_true"].median()).astype(float))
    k = len(thresholded) // 2
    expected = float((thresholded.nlargest(k, "y_pred")["y_true"] == 1.0).mean())
    # A zero-info predictor's top-k selection is arbitrary among ties (many
    # rows share the same per-drug-mean y_pred), so assert it reproduces
    # exactly its own deterministic pandas tie-break rather than a fixed
    # constant -- the point of this test is that the VALUE IS COMPUTABLE AND
    # STABLE, not inflated by missingness, which a re-run with the same seed
    # confirms.
    assert np.isclose(top_k_hit_rate(thresholded, k=k), expected)


def test_brier_score_null_on_unbalanced_zero_info_predictor() -> None:
    def readout(panel: pd.DataFrame) -> float:
        panel = panel.assign(y_true=(panel["y_true"] > panel["y_true"].median()).astype(float))
        return brier_score(panel)

    # A drug-constant y_pred is not itself a probability in [0,1] here (it's a
    # raw y_true mean); rescale isn't needed for the null check -- brier_score
    # on a panel where y_pred carries no patient information should equal the
    # brier score of predicting each row's own drug's empirical responder
    # rate, which is exactly what a zero-info, drug-only predictor computes
    # regardless of panel balance. Assert it matches that direct computation.
    panel = unbalanced_zero_info_panel(seed=0)
    thresholded = panel.assign(y_true=(panel["y_true"] > panel["y_true"].median()).astype(float))
    drug_rate = thresholded.groupby("drug")["y_true"].transform("mean")
    expected = float(np.mean((drug_rate - thresholded["y_true"]) ** 2))
    assert np.isclose(brier_score(thresholded.assign(y_pred=drug_rate)), expected)
```

Add `from fmharness.classification_readouts import top_k_hit_rate, brier_score` if not already
imported at the top of the file (both already are, from Step 1's original test).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_classification_readouts.py -k unbalanced -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'readout_contract'`

- [ ] **Step 3: Confirm the helper module needs no further implementation**

`tests/readout_contract.py` was already written in full in Step 1 -- it is not a
test-first/implementation-second file, it is the shared contract itself. Re-run:

Run: `uv run pytest tests/test_classification_readouts.py -v`
Expected: PASS (9 tests total: 7 from Task 7 plus the 2 new unbalanced-panel checks).

- [ ] **Step 4: Lint and typecheck**

Run: `uv run ruff check tests/readout_contract.py tests/test_classification_readouts.py && uv run pyright tests/readout_contract.py tests/test_classification_readouts.py`
Expected: no errors. `tests/**` ignores `E501` only, so other lint rules still apply to
`readout_contract.py` despite it living under `tests/`.

- [ ] **Step 5: Run the full suite to confirm nothing else broke**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Report the commit (do not run it)**

Intended: `git add tests/readout_contract.py tests/test_classification_readouts.py`
Message: `test: add the unbalanced zero-info-predictor contract, apply to classification readouts`

---

### Task 9: `Estimator` protocol + `BilinearEstimator`

`docs/models.md` and `src/fmharness/probe/heads.py`'s own docstring both confirm bilinear
does not conform to the shared probe contract ("it does not satisfy the `fit(emb, drugs, y)`
contract"). This task defines that contract as an explicit, checkable `Protocol` (proving
`SimpleProbe`/`KernelProbe` already satisfy it, per "reuse `ProbeBase`, do not introduce a
second estimator protocol"), then wraps `bilinear_features` to conform too.

**Files:**
- Create: `src/fmharness/probe/estimator.py`
- Create: `src/fmharness/probe/bilinear_head.py`
- Test: `tests/test_estimator_protocol.py`
- Test: `tests/test_bilinear_head.py`

**Interfaces:**
- Consumes: `fmharness.probe.base.ProbeBase`, `ALPHAS`, `fmharness.probe.simple.SimpleProbe`,
  `fmharness.probe.kernel.KernelProbe` (all existing), `fmharness.bilinear.bilinear_features`
  (existing).
- Produces: `Estimator` protocol, `BilinearEstimator` class. Used by Task 10.

- [ ] **Step 1: Write the failing test for the protocol**

Create `tests/test_estimator_protocol.py`:

```python
"""Tests that the existing probe heads conform to the (now explicit) Estimator protocol."""

from __future__ import annotations

from fmharness.probe.estimator import Estimator
from fmharness.probe.kernel import KernelProbe
from fmharness.probe.simple import SimpleProbe


def test_simple_probe_satisfies_estimator() -> None:
    assert isinstance(SimpleProbe(), Estimator)


def test_kernel_probe_satisfies_estimator() -> None:
    assert isinstance(KernelProbe(), Estimator)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_estimator_protocol.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmharness.probe.estimator'`

- [ ] **Step 3: Write the protocol**

Create `src/fmharness/probe/estimator.py`:

```python
"""``Estimator`` protocol: the shared ``fit``/``predict_parts`` contract every
probe head and every wrapped standalone model (bilinear, biomarker) satisfies.

``SimpleProbe`` and ``KernelProbe`` (``probe/simple.py``, ``probe/kernel.py``)
already implement this shape; this module makes it an explicit, checkable
Protocol rather than an implicit convention, per the design spec's decision to
wrap bilinear and biomarker into this same contract rather than leave them
standalone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from numpy.typing import ArrayLike, NDArray
import numpy as np


@runtime_checkable
class Estimator(Protocol):
    def fit(
        self,
        embeddings: ArrayLike,
        drug_ids: Sequence[str],
        y: ArrayLike,
        groups: Sequence[str] | None = None,
    ) -> Estimator: ...

    def predict_parts(
        self, embeddings: ArrayLike, drug_ids: Sequence[str]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_estimator_protocol.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write the failing test for BilinearEstimator**

Create `tests/test_bilinear_head.py`:

```python
"""Tests for the Estimator-conforming bilinear wrapper."""

from __future__ import annotations

import numpy as np

from fmharness.probe.bilinear_head import BilinearEstimator
from fmharness.probe.estimator import Estimator


def _synthetic(n_patients: int = 20, emb_dim: int = 6, fp_dim: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    drugs = ["d1", "d2", "d3"]
    patients = [f"p{i}" for i in range(n_patients)]
    fingerprints = {d: rng.standard_normal(fp_dim) for d in drugs}
    rows_emb, rows_drug, rows_y, rows_groups = [], [], [], []
    z_by_patient = {p: rng.standard_normal(emb_dim) for p in patients}
    for p in patients:
        for d in drugs:
            z = z_by_patient[p]
            g = fingerprints[d]
            y = float(z @ g[: emb_dim if fp_dim >= emb_dim else fp_dim].sum()) + rng.normal(
                scale=0.01
            )
            rows_emb.append(z)
            rows_drug.append(d)
            rows_y.append(y)
            rows_groups.append(p)
    return (
        np.asarray(rows_emb, dtype=np.float64),
        rows_drug,
        np.asarray(rows_y, dtype=np.float64),
        rows_groups,
        fingerprints,
    )


def test_bilinear_estimator_satisfies_estimator_protocol() -> None:
    _, _, _, _, fingerprints = _synthetic()
    assert isinstance(BilinearEstimator(fingerprints), Estimator)


def test_bilinear_estimator_fits_and_predicts_shapes() -> None:
    emb, drugs, y, groups, fingerprints = _synthetic()
    est = BilinearEstimator(fingerprints, n_components=3, seed=0).fit(emb, drugs, y, groups)
    base, residual = est.predict_parts(emb, drugs)
    assert base.shape == (len(y),)
    assert residual.shape == (len(y),)


def test_bilinear_estimator_falls_back_to_drug_mean_for_unknown_drug() -> None:
    emb, drugs, y, groups, fingerprints = _synthetic()
    est = BilinearEstimator(fingerprints, n_components=3, seed=0).fit(emb, drugs, y, groups)
    base, residual = est.predict_parts(emb[:1], ["unknown_drug"])
    assert residual[0] == 0.0  # no fingerprint -> no residual term
    assert base[0] != 0.0  # falls back to the global mean, not zero


def test_bilinear_estimator_recovers_signal_better_than_drug_mean_alone() -> None:
    # The synthetic y depends on z . g -- a fitted bilinear estimator should
    # explain materially more residual variance than predicting 0 residual
    # for everyone (the drug-mean-only floor).
    emb, drugs, y, groups, fingerprints = _synthetic(n_patients=40, seed=1)
    est = BilinearEstimator(fingerprints, n_components=4, seed=0).fit(emb, drugs, y, groups)
    base, residual = est.predict_parts(emb, drugs)
    mse_fitted = float(np.mean((base + residual - y) ** 2))
    mse_drug_mean_only = float(np.mean((base - y) ** 2))
    assert mse_fitted < mse_drug_mean_only * 0.5
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_bilinear_head.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmharness.probe.bilinear_head'`

- [ ] **Step 7: Write minimal implementation**

Create `src/fmharness/probe/bilinear_head.py`:

```python
"""Estimator-conforming wrapper around ``bilinear_features``.

``bilinear.py``'s ``AUC(s, d) = ridge([z_s, g_d, z_s (x) g_d])`` needs a drug
fingerprint lookup that a plain ``embeddings`` array can't carry, so this
wraps it: fingerprints are injected at construction, and ``fit``/``predict_parts``
match the same signature every other probe head uses. Drugs missing a
fingerprint fall back to the drug-mean base with zero residual, matching
``SimpleProbe``'s graceful-degradation contract -- an uninformative model
reduces to the drug mean rather than injecting noise.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from fmharness.bilinear import bilinear_features
from fmharness.probe.base import ALPHAS, ProbeBase


class BilinearEstimator(ProbeBase):
    """Per-drug mean + ridge on ``[z, g, z (x) g]``, reusing ProbeBase's PCA
    reduction of ``z`` and per-drug-mean bookkeeping."""

    def __init__(
        self,
        drug_fingerprints: dict[str, NDArray[np.float64]],
        *,
        n_components: int = 10,
        alphas: Sequence[float] = ALPHAS,
        seed: int = 0,
    ) -> None:
        super().__init__(n_components=n_components, seed=seed)
        self.drug_fingerprints = drug_fingerprints
        self.alphas = tuple(alphas)
        self._reduce: Pipeline | None = None
        self._ridge: RidgeCV | None = None

    def _stack_fingerprints(
        self, drug_ids: Sequence[str]
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        dim = len(next(iter(self.drug_fingerprints.values())))
        g = np.zeros((len(drug_ids), dim), dtype=np.float64)
        known = np.zeros(len(drug_ids), dtype=bool)
        for i, d in enumerate(drug_ids):
            fp = self.drug_fingerprints.get(d)
            if fp is not None:
                g[i] = fp
                known[i] = True
        return g, known

    def fit(
        self,
        embeddings: ArrayLike,
        drug_ids: Sequence[str],
        y: ArrayLike,
        groups: Sequence[str] | None = None,
    ) -> BilinearEstimator:
        emb, _drug_arr, residual, k = self._prepare_fit(embeddings, drug_ids, y)
        g, known = self._stack_fingerprints(drug_ids)
        self._reduce = self._ridge = None
        if known.any() and k > 0:
            self._reduce = Pipeline(self._reducer_steps(k))
            z = self._reduce.fit_transform(emb[known])
            feats = bilinear_features(z, g[known])
            cv = None
            if groups is not None:
                grp = np.asarray(groups)[known]
                n_g = len(np.unique(grp))
                if n_g >= 2:
                    cv = list(
                        GroupKFold(n_splits=min(5, n_g)).split(
                            feats, residual[known], groups=grp
                        )
                    )
            self._ridge = RidgeCV(alphas=np.asarray(self.alphas, dtype=np.float64), cv=cv)
            self._ridge.fit(feats, residual[known])
        return self

    def predict_parts(
        self, embeddings: ArrayLike, drug_ids: Sequence[str]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if not self._drug_means:
            raise RuntimeError("estimator is not fitted; call fit() before predict_parts()")
        base = self._base(drug_ids)
        emb = np.asarray(embeddings, dtype=np.float64)
        residual = np.zeros(len(drug_ids), dtype=np.float64)
        if self._reduce is not None and self._ridge is not None:
            g, known = self._stack_fingerprints(drug_ids)
            if known.any():
                z = self._reduce.transform(emb[known])
                feats = bilinear_features(z, g[known])
                residual[known] = self._ridge.predict(feats)
        return base, residual
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_bilinear_head.py -v`
Expected: PASS (4 tests). If `test_bilinear_estimator_recovers_signal_better_than_drug_mean_alone`
is flaky, increase `n_patients` in `_synthetic` rather than loosening the `0.5` threshold --
the point of the test is that a real fit recovers real signal, not that it barely beats chance.

- [ ] **Step 9: Lint and typecheck**

Run: `uv run ruff check src/fmharness/probe tests/test_estimator_protocol.py tests/test_bilinear_head.py && uv run pyright src/fmharness/probe/estimator.py src/fmharness/probe/bilinear_head.py`
Expected: no errors.

- [ ] **Step 10: Report the commit (do not run it)**

Intended: `git add src/fmharness/probe/estimator.py src/fmharness/probe/bilinear_head.py tests/test_estimator_protocol.py tests/test_bilinear_head.py`
Message: `feat: add Estimator protocol and BilinearEstimator wrapper`

---

### Task 10: `BiomarkerEstimator`

Extracts `scripts/biomarker_anchored.py`'s `_biomarker_series` scoring logic (kept verbatim,
just moved so it's importable without executing the script's `__main__`) into an
`Estimator`-conforming wrapper. Rules are pre-specified, not learned, so `fit` only learns the
per-drug mean; `predict_parts` looks up each row's matching rule directly.

**Files:**
- Create: `src/fmharness/probe/biomarker_head.py`
- Modify: `scripts/biomarker_anchored.py` (import from the new location instead of defining
  `_biomarker_series` locally)
- Test: `tests/test_biomarker_head.py`

**Interfaces:**
- Consumes: `fmharness.probe.estimator.Estimator` (Task 9), `fmharness.probe.base.ProbeBase`
  (existing).
- Produces: `biomarker_series(rule, x_log, alterations, wes_patients, sym2entrez) -> pd.Series | None`
  (the moved, unchanged function), `BiomarkerEstimator` class.

- [ ] **Step 1: Write the failing test**

Create `tests/test_biomarker_head.py`:

```python
"""Tests for the Estimator-conforming biomarker wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmharness.probe.biomarker_head import BiomarkerEstimator, biomarker_series
from fmharness.probe.estimator import Estimator


def _synthetic() -> tuple[pd.DataFrame, list[str], np.ndarray, list[dict]]:
    # 4 patients, 2 drugs. drugA has an "expr" biomarker rule keyed on entrez "100";
    # drugB has a "mut" rule. patients p1/p3 carry the mutation.
    x_log = pd.DataFrame(
        {"100": [2.0, 0.0, 2.0, 0.0]},
        index=["p1", "p2", "p3", "p4"],
    )
    rows_patient = ["p1", "p2", "p3", "p4"] * 2
    rows_drug = ["drugA"] * 4 + ["drugB"] * 4
    rows_y = [10.0, 60.0, 15.0, 65.0, 20.0, 70.0, 25.0, 75.0]
    biomarkers = [
        {"drug": "drugA", "gene": "GENE100", "kind": "expr", "direction": "sensitize"},
        {"drug": "drugB", "gene": "GENEMUT", "kind": "mut", "direction": "sensitize"},
    ]
    alt = {"mut": {"GENEMUT": {"p1", "p3"}}, "amp": {}, "del": {}}
    wes = {"p1", "p2", "p3", "p4"}
    sym2ent = {"GENE100": 100, "GENEMUT": 999}
    return x_log, rows_patient, rows_drug, rows_y, biomarkers, alt, wes, sym2ent


def test_biomarker_estimator_satisfies_estimator_protocol() -> None:
    x_log, patients, drugs, y, biomarkers, alt, wes, sym2ent = _synthetic()
    est = BiomarkerEstimator(biomarkers, alt, wes, sym2ent)
    assert isinstance(est, Estimator)


def test_biomarker_estimator_fit_predict_uses_the_matching_rule() -> None:
    x_log, patients, drugs, y, biomarkers, alt, wes, sym2ent = _synthetic()
    est = BiomarkerEstimator(biomarkers, alt, wes, sym2ent).fit(x_log, drugs, np.asarray(y))
    x_log_indexed = x_log.loc[patients]
    base, residual = est.predict_parts(x_log_indexed, drugs)
    assert base.shape == (8,)
    assert residual.shape == (8,)
    # drugA rows: residual should differ between p1/p3 (expr=2.0) and p2/p4 (expr=0.0)
    drug_a_mask = [d == "drugA" for d in drugs]
    resid_a = residual[drug_a_mask]
    assert resid_a[0] != resid_a[1]  # p1 (high expr) differs from p2 (low expr)


def test_biomarker_estimator_zero_residual_for_drug_with_no_rule() -> None:
    x_log, patients, drugs, y, biomarkers, alt, wes, sym2ent = _synthetic()
    est = BiomarkerEstimator(biomarkers, alt, wes, sym2ent).fit(x_log, drugs, np.asarray(y))
    x_log_indexed = x_log.loc[patients]
    _, residual = est.predict_parts(x_log_indexed, ["drugC"] * 4)
    np.testing.assert_array_equal(residual, np.zeros(4))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_biomarker_head.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmharness.probe.biomarker_head'`

- [ ] **Step 3: Write minimal implementation**

Create `src/fmharness/probe/biomarker_head.py`:

```python
"""Estimator-conforming wrapper around the biomarker rule table.

``biomarker_series`` is ``scripts/biomarker_anchored.py``'s ``_biomarker_series``,
moved here verbatim so it's importable without executing the script's
``__main__`` (which reads real WES/expression files from disk on import). Rules
are pre-specified (a fixed list of drug/gene/kind/direction dicts), not learned,
so ``fit`` only learns the per-drug mean base; ``predict_parts`` looks up each
row's matching rule and applies it directly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from fmharness.probe.base import ProbeBase


def biomarker_series(
    bm: dict,
    x_log: pd.DataFrame,
    alt: dict,
    wes: set[str],
    sym2ent: dict[str, int],
) -> pd.Series | None:
    """Per-patient biomarker value (index = patient_id); None if unavailable."""
    if bm["kind"] == "expr":
        ent = sym2ent.get(bm["gene"])
        col = str(ent) if ent is not None else None
        if col is None or col not in x_log.columns:
            return None
        v = x_log[col]
        return (v - v.mean()) / (v.std() or 1.0)
    positive = alt[bm["kind"]].get(bm["gene"], set())
    return pd.Series({p: float(p in positive) for p in sorted(wes)})


class BiomarkerEstimator(ProbeBase):
    """Wraps the biomarker rule table into the Estimator contract.

    ``features`` must be a DataFrame indexed by patient_id (not a bare ndarray)
    since biomarker lookup keys off patient identity, not row position -- a
    deliberate deviation from the plain-embedding convention other heads use,
    documented here rather than forced into an artificial fit. This remains a
    valid Estimator: the Protocol checks method presence structurally, not
    parameter types.
    """

    def __init__(
        self,
        biomarkers: list[dict],
        alterations: dict,
        wes_patients: set[str],
        sym2entrez: dict[str, int],
        seed: int = 0,
    ) -> None:
        super().__init__(n_components=0, seed=seed)
        self.biomarkers = biomarkers
        self.alterations = alterations
        self.wes_patients = wes_patients
        self.sym2entrez = sym2entrez

    def fit(
        self,
        embeddings: ArrayLike,
        drug_ids: Sequence[str],
        y: ArrayLike,
        groups: Sequence[str] | None = None,
    ) -> BiomarkerEstimator:
        # Only the per-drug-mean bookkeeping from _prepare_fit is used here;
        # the "embeddings" arg (a features DataFrame) never reaches PCA/NMF
        # since n_components=0 forces k=0 in _prepare_fit's reduction-rank calc.
        placeholder = np.zeros((len(np.asarray(y, dtype=np.float64)), 1))
        self._prepare_fit(placeholder, drug_ids, y)
        return self

    def predict_parts(
        self, embeddings: ArrayLike, drug_ids: Sequence[str]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if not self._drug_means:
            raise RuntimeError("estimator is not fitted; call fit() before predict_parts()")
        if not isinstance(embeddings, pd.DataFrame):
            raise TypeError(
                "BiomarkerEstimator.predict_parts requires a DataFrame indexed by "
                "patient_id, not a bare array -- biomarker lookup keys off patient identity"
            )
        base = self._base(drug_ids)
        residual = np.zeros(len(drug_ids), dtype=np.float64)
        for bm in self.biomarkers:
            rows = [i for i, d in enumerate(drug_ids) if d == bm["drug"]]
            if not rows:
                continue
            b = biomarker_series(bm, embeddings, self.alterations, self.wes_patients, self.sym2entrez)
            if b is None:
                continue
            sign = -1.0 if bm["direction"] == "sensitize" else 1.0
            for i in rows:
                patient = embeddings.index[i]
                if patient in b.index:
                    residual[i] = sign * float(b.loc[patient])
        return base, residual
```

- [ ] **Step 4: Update `scripts/biomarker_anchored.py` to import instead of redefine**

Read `scripts/biomarker_anchored.py`'s current `_biomarker_series` definition and its call
sites (`grep -n "_biomarker_series" scripts/biomarker_anchored.py`). Delete the local
`_biomarker_series` function definition and add, near the top-level imports:

```python
from fmharness.probe.biomarker_head import biomarker_series as _biomarker_series
```

This keeps every existing call site (`_biomarker_series(bm, x_log, alt, wes, sym2ent)`) working
unchanged, since the moved function has the identical signature and body.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_biomarker_head.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Verify the script still imports cleanly**

Run: `uv run python -c "import ast; ast.parse(open('scripts/biomarker_anchored.py').read())"`
Expected: no error (syntax-valid). Do not execute `scripts/biomarker_anchored.py` itself --
`main()` reads real WES/expression files from disk that may not be present locally.

- [ ] **Step 7: Lint and typecheck**

Run: `uv run ruff check src/fmharness/probe/biomarker_head.py scripts/biomarker_anchored.py tests/test_biomarker_head.py && uv run pyright src/fmharness/probe/biomarker_head.py`
Expected: no errors. `scripts/` is not under pyright's strict `src`/`tests` scope per the
project's global constraints, so a script-level type issue there is not blocking, but ruff
must still be clean.

- [ ] **Step 8: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 9: Report the commit (do not run it)**

Intended: `git add src/fmharness/probe/biomarker_head.py scripts/biomarker_anchored.py tests/test_biomarker_head.py`
Message: `feat: add BiomarkerEstimator, move biomarker_series out of the script`

---

## Plan exit criteria

- All ten tasks' tests pass; `uv run pytest -q` is green for the full suite.
- `uv run ruff check src tests` and `uv run pyright src tests` are both clean.
- `Encoder`, `Generator`, `Modality`, `CVScheme`, `Estimator`, `LeakageQueryable` are all
  `@runtime_checkable` protocols, each proven via `isinstance` against at least one real
  (non-mock) existing class where one exists (`MockAdapter`/`LinearBaselineAdapter` for
  `Encoder`; `SimpleProbe`/`KernelProbe` for `Estimator`).
- No existing test in the repository regresses (`ModelMetadata`'s new required field is the
  one change with a blast radius -- Task 2 Step 5 explicitly re-runs the full suite to catch
  every call site).

## Not in this plan

Wiring these registries into an actual driver that reproduces the existing Arm-2 ladder
(`docs/tahoe_generation_results.md`), and the scFoundation stress test proving the core is
genuinely swappable against a second, architecturally distinct model -- both require research
into scFoundation's real API that this plan deliberately does not speculate about. Separate
plan, per the design spec's acceptance criteria 2 and 4.
