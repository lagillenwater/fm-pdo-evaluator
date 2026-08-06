# Phase 0 — Close the Loop on Tahoe: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the Arm-2 selection metric (`gap@k` in raw AUC) is valid, by scoring
the potency prior against every representation on identical folds and measuring how concentrated
each representation's shortlists are.

**Architecture:** All testable logic lands in `src/fmharness/` (a new `selection.py` and a new
`moa.py`); `scripts/score_generation_eval.py` gains only a per-pair prediction dump; a new thin
driver `scripts/check2_selection_audit.py` reads that dump and prints the tables. This mirrors the
existing split -- pyright and ruff cover `src` and `tests` only, and no test imports from
`scripts/`, so anything that needs a test must live in `src/fmharness/`.

**Tech Stack:** Python 3.11, pandas, numpy, scipy, pyarrow (parquet), pytest, uv.

## Global Constraints

- Line length 100 (`[tool.ruff]` in `pyproject.toml`).
- `target-version = "py311"`; pyright `typeCheckingMode = "strict"` over `src` and `tests`.
- Ruff lint selects `E, F, I, B, UP, SIM, RUF`. `tests/**` ignores `E501` only.
- Run everything through `uv`: `uv run pytest`, `uv run python`, `uv run ruff`, `uv run pyright`.
- **No emojis anywhere** -- code, comments, output, commit messages.
- **Vectorized only.** No nested Python loops over data. Use pandas groupby / numpy. A
  comprehension over a 3-element `ks` tuple is fine; a loop over rows is not.
- **Do not run `git commit` or `git push`.** Lucas commits in VS Code himself. Commit steps below
  state the intended message and file set; stage nothing and report the message instead.
- New public functions need docstrings in the surrounding style: what it computes, and why that
  is the right quantity -- see `src/fmharness/evaluation.py:203-212` for the register.

---

### Task 1: Within-drug percentile transform

The doc's fix for a concentrated panel (`docs/tahoe_generation_results.md:186-192`): score
selection in each drug's rank among lines rather than raw AUC, so a pan-cytotoxic compound carries
no advantage. Making every drug's marginal uniform is the whole mechanism.

**Files:**
- Create: `src/fmharness/selection.py`
- Test: `tests/test_selection.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `within_drug_percentile(preds: pd.DataFrame, cols: tuple[str, ...] = ("y_true", "y_pred")) -> pd.DataFrame`
  -- returns a copy of `preds` with each named column replaced by its within-drug percentile rank
  in `(0, 1]`. Used by Task 7 for the `pct_gap@k` columns.

- [ ] **Step 1: Write the failing test**

Create `tests/test_selection.py`:

```python
"""Tests for selection-metric machinery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmharness.selection import within_drug_percentile


def test_within_drug_percentile_makes_each_drug_uniform() -> None:
    # d_toxic is on a completely different scale from d_mild. In raw AUC, d_toxic dominates
    # any cross-drug comparison; after the transform each drug spans the same (0, 1] ranks,
    # so scale carries no advantage.
    preds = pd.DataFrame(
        {
            "patient": ["A", "B", "C", "D"] * 2,
            "drug": ["d_toxic"] * 4 + ["d_mild"] * 4,
            "y_true": [0.01, 0.02, 0.03, 0.04, 0.90, 0.92, 0.94, 0.96],
            "y_pred": [0.04, 0.03, 0.02, 0.01, 0.96, 0.94, 0.92, 0.90],
        }
    )
    out = within_drug_percentile(preds)
    for drug in ("d_toxic", "d_mild"):
        g = out.loc[out["drug"] == drug, "y_true"].to_numpy()
        assert np.allclose(np.sort(g), [0.25, 0.5, 0.75, 1.0])
    # Order within a drug is preserved; only the scale changes.
    toxic = out.loc[out["drug"] == "d_toxic", "y_true"].to_numpy()
    assert np.allclose(toxic, [0.25, 0.5, 0.75, 1.0])
    # The input frame is untouched.
    assert np.isclose(preds["y_true"].iloc[0], 0.01)


def test_within_drug_percentile_only_transforms_named_columns() -> None:
    preds = pd.DataFrame(
        {
            "patient": ["A", "B"],
            "drug": ["d1", "d1"],
            "y_true": [10.0, 20.0],
            "y_pred": [1.0, 2.0],
            "y_prior": [5.0, 5.0],
        }
    )
    out = within_drug_percentile(preds, cols=("y_true",))
    assert np.allclose(out["y_true"].to_numpy(), [0.5, 1.0])
    assert np.allclose(out["y_pred"].to_numpy(), [1.0, 2.0])
    assert np.allclose(out["y_prior"].to_numpy(), [5.0, 5.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_selection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmharness.selection'`

- [ ] **Step 3: Write minimal implementation**

Create `src/fmharness/selection.py`:

```python
"""Selection-metric machinery for the check-2 shortlist audit.

`gap@k` in raw AUC rewards ranking broadly-potent compounds, because a pan-cytotoxic drug is
close to the best drug for most cell lines. Scoring in each drug's rank among lines instead
makes every drug's marginal uniform, so breadth of potency carries no advantage and only
line-specific ordering can score. The concentration summary answers the prior question --
whether a representation is producing line-specific shortlists at all, or re-picking the same
few toxic compounds for everyone.
"""

from __future__ import annotations

import pandas as pd


def within_drug_percentile(
    preds: pd.DataFrame, cols: tuple[str, ...] = ("y_true", "y_pred")
) -> pd.DataFrame:
    """Replace each named column with its within-drug percentile rank in ``(0, 1]``.

    Ranking inside each drug removes that drug's location and scale, so a compound that is
    potent on every line no longer sits closer to the per-line optimum than a selective one.
    Order within a drug is preserved. ``preds`` is not modified.
    """
    out = preds.copy()
    for col in cols:
        out[col] = out.groupby("drug")[col].rank(pct=True)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_selection.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/fmharness/selection.py tests/test_selection.py && uv run ruff format --check src/fmharness/selection.py tests/test_selection.py && uv run pyright src/fmharness/selection.py`
Expected: no errors.

- [ ] **Step 6: Report the commit (do not run it)**

Intended: `git add src/fmharness/selection.py tests/test_selection.py`
Message: `feat: add within-drug percentile transform for selection scoring`

---

### Task 2: Shortlist concentration summary

Answers `docs/tahoe_generation_results.md:142-165`: does a representation pick only 1-2 distinct
drugs across all lines, or does its pick distribution resemble the observed one? The trap recorded
there is that the truth is itself concentrated -- across 955 GDSC2 lines the observed best drug is
one of only 13 compounds, Staurosporine for 69% -- so the question is whether models are *more*
concentrated than the truth, which requires reporting the observed row as the reference.

**Files:**
- Modify: `src/fmharness/selection.py`
- Test: `tests/test_selection.py`

**Interfaces:**
- Consumes: `src/fmharness/selection.py` from Task 1.
- Produces:
  - `broadly_active_drugs(preds: pd.DataFrame, frac: float = 0.5) -> set[str]` -- drugs whose
    `y_true` is below the line's own median for more than `frac` of lines.
  - `shortlist_concentration(preds: pd.DataFrame, pred_col: str = "y_pred") -> dict[str, float | str]`
    -- keys `distinct`, `modal_drug`, `modal_share`, `broadly_active_share`, `n_lines`.
    Passing `pred_col="y_true"` yields the observed reference row.
    Used by Task 7.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_selection.py`:

```python
from fmharness.selection import broadly_active_drugs, shortlist_concentration


def _panel() -> pd.DataFrame:
    # Four lines x three drugs. d_tox is below each line's median everywhere (broadly active);
    # d_sel is the true best for exactly one line; d_weak is never good.
    return pd.DataFrame(
        {
            "patient": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D"],
            "drug": ["d_tox", "d_sel", "d_weak"] * 4,
            "y_true": [
                0.20, 0.10, 0.90,  # A: d_sel best
                0.20, 0.50, 0.90,  # B: d_tox best
                0.20, 0.60, 0.90,  # C: d_tox best
                0.20, 0.70, 0.90,  # D: d_tox best
            ],
        }
    )


def test_broadly_active_drugs_flags_the_pan_potent_compound() -> None:
    preds = _panel()
    assert broadly_active_drugs(preds) == {"d_tox"}


def test_shortlist_concentration_collapsed_model() -> None:
    # A model that ignores the line and always ranks d_tox first: 1 distinct pick, modal
    # share 1.0, and every pick is a broadly-active compound.
    preds = _panel()
    preds["y_pred"] = preds["drug"].map({"d_tox": 0.0, "d_sel": 1.0, "d_weak": 2.0})
    c = shortlist_concentration(preds)
    assert c["distinct"] == 1.0
    assert c["modal_drug"] == "d_tox"
    assert np.isclose(float(c["modal_share"]), 1.0)
    assert np.isclose(float(c["broadly_active_share"]), 1.0)
    assert c["n_lines"] == 4.0


def test_shortlist_concentration_observed_reference() -> None:
    # Scoring y_true against itself gives the observed reference row: d_sel wins for A,
    # d_tox for B/C/D -> 2 distinct, modal share 3/4, and 3 of 4 picks broadly active.
    c = shortlist_concentration(_panel(), pred_col="y_true")
    assert c["distinct"] == 2.0
    assert c["modal_drug"] == "d_tox"
    assert np.isclose(float(c["modal_share"]), 0.75)
    assert np.isclose(float(c["broadly_active_share"]), 0.75)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_selection.py -v`
Expected: FAIL with `ImportError: cannot import name 'broadly_active_drugs'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/fmharness/selection.py`:

```python
def broadly_active_drugs(preds: pd.DataFrame, frac: float = 0.5) -> set[str]:
    """Drugs below the line's own median response for more than ``frac`` of lines.

    ``y_true`` is AUC-like (lower is more sensitive), so these are the compounds that work on
    nearly everything. Picking them is partly correct behaviour -- the observed best drug is
    usually one of them -- which is why the shortlist audit reports the observed row as its
    reference rather than treating any such pick as a failure.
    """
    median = preds.groupby("patient")["y_true"].transform("median")
    share = (preds["y_true"] < median).groupby(preds["drug"]).mean()
    return set(share.index[share > frac].astype(str))


def shortlist_concentration(
    preds: pd.DataFrame, pred_col: str = "y_pred"
) -> dict[str, float | str]:
    """How concentrated a representation's top-1 picks are across lines.

    Returns the number of distinct drugs ever ranked first, the most-picked drug and its share
    of lines, and the share of top-1 picks that are broadly active. Call with
    ``pred_col="y_true"`` for the observed reference: the truth is itself concentrated, so a
    model is only collapsed if it is *more* concentrated than that row.
    """
    top1 = preds.loc[preds.groupby("patient")[pred_col].idxmin(), ["patient", "drug"]]
    counts = top1["drug"].value_counts()
    active = broadly_active_drugs(preds)
    return {
        "distinct": float(counts.size),
        "modal_drug": str(counts.index[0]),
        "modal_share": float(counts.iloc[0] / len(top1)),
        "broadly_active_share": float(top1["drug"].isin(active).mean()),
        "n_lines": float(len(top1)),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_selection.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/fmharness/selection.py tests/test_selection.py && uv run pyright src/fmharness/selection.py`
Expected: no errors.

- [ ] **Step 6: Report the commit (do not run it)**

Intended: `git add src/fmharness/selection.py tests/test_selection.py`
Message: `feat: add shortlist concentration summary with observed reference`

---

### Task 3: Emit the potency prior and dump per-pair predictions

`_penalized_preds` builds the per-(line, drug) prediction frame, scores it, and discards it
(`scripts/score_generation_eval.py:465-469`), so the picks are unrecoverable and the prior has
never been scored. The prior is already inside the fitted model: `StandardScaler` centres the
training features and `fit_intercept=True`, so `model.intercept_` is the training-fold mean AUC
for that drug (`:178-182`). Emitting it as a column is equivalent and avoids depending on the
estimator's internals.

**Files:**
- Modify: `scripts/score_generation_eval.py:183-187` (row construction and column list)
- Modify: `scripts/score_generation_eval.py:463-470` (accumulate frames, dump parquet)
- Modify: `scripts/score_generation_eval.py` argument parser (add `--preds-out`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `results/check2_preds.parquet` with columns
  `source, method, patient, drug, y_true, y_pred, y_prior`. Read by Task 7.

No unit test: `_penalized_preds` is script-local and the repo's pyright/ruff config covers `src`
and `tests` only. The verification is the end-to-end run in Step 4, which asserts against the
already-published table.

- [ ] **Step 1: Add `y_prior` to the row construction**

In `scripts/score_generation_eval.py`, replace:

```python
            pred = model.predict(sc.transform(fdf.loc[te].to_numpy(dtype=np.float64)))
            rows.extend(
                (ln, drug, float(auc[ln]), float(p)) for ln, p in zip(te, pred, strict=False)
            )
    cols = ["patient", "drug", "y_true", "y_pred"]
```

with:

```python
            pred = model.predict(sc.transform(fdf.loc[te].to_numpy(dtype=np.float64)))
            # The potency prior: this drug's mean AUC over the training-fold lines, i.e. the
            # same fitted model with its coefficients zeroed. Ranking by it ignores the cell
            # line entirely, so it is the floor any line-specific claim has to clear.
            prior = float(np.mean([auc[ln] for ln in tr]))
            rows.extend(
                (ln, drug, float(auc[ln]), float(p), prior)
                for ln, p in zip(te, pred, strict=False)
            )
    cols = ["patient", "drug", "y_true", "y_pred", "y_prior"]
```

Also update the `rows` declaration a few lines above from
`rows: list[tuple[str, str, float, float]] = []` to
`rows: list[tuple[str, str, float, float, float]] = []`.

- [ ] **Step 2: Add the `--preds-out` argument**

In the argument parser, immediately after the `--folds` argument block, add:

```python
    ap.add_argument(
        "--preds-out",
        default="results/check2_preds.parquet",
        help="per-(line, drug) check-2 predictions dump; enables the selection audit",
    )
```

- [ ] **Step 3: Accumulate and dump the frames**

Replace the check-2 representation loop:

```python
    for repr_name, feat in representations.items():
        for pen in penalties:
            preds = _penalized_preds(feat, design_target, fold_of, n_folds, uniq_lines, pen)
            if preds.empty:
                continue
            s = score_predictions(preds, n_perm=args.n_permutations)
            out.append({"source": repr_name, "method": pen, **_row(s)})
```

with:

```python
    pred_frames: list[pd.DataFrame] = []
    for repr_name, feat in representations.items():
        for pen in penalties:
            preds = _penalized_preds(feat, design_target, fold_of, n_folds, uniq_lines, pen)
            if preds.empty:
                continue
            pred_frames.append(preds.assign(source=repr_name, method=pen))
            s = score_predictions(preds, n_perm=args.n_permutations)
            out.append({"source": repr_name, "method": pen, **_row(s)})

    if pred_frames:
        dest = _rel(repo, args.preds_out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        cols = ["source", "method", "patient", "drug", "y_true", "y_pred", "y_prior"]
        pd.concat(pred_frames, ignore_index=True)[cols].to_parquet(dest, index=False)
        print(f"wrote {dest} ({sum(len(f) for f in pred_frames)} rows)")
```

- [ ] **Step 4: Re-run check 2 and verify nothing moved**

This runs on Alpine, where the Tahoe inputs live. Author the change locally, then have Lucas
submit `scripts/alpine/07_stack_emb_score.sbatch` with the updated checkout.

Verify: the printed check-2 table still matches `docs/tahoe_generation_results.md:78-88` --
in particular `base (embed)` at L2 global `0.644` / interaction `+0.119`. `y_prior` is a new
column and must not change any existing number. If a number moves, stop and diagnose; this
step is additive.

Then confirm the dump: `./scripts/alpine/ralpine ls <repo>/results/check2_preds.parquet`

- [ ] **Step 5: Pull the dump locally**

Run: `./scripts/alpine/ralpine pull /projects/lgillenwater@xsede.org/repositories/fm-pdo-evaluator/results/check2_preds.parquet results/check2_preds.parquet`
Expected: file present, and
`uv run python -c "import pandas as pd; d=pd.read_parquet('results/check2_preds.parquet'); print(d.shape, sorted(d['source'].unique()))"`
lists every representation from the ladder.

- [ ] **Step 6: Report the commit (do not run it)**

Intended: `git add scripts/score_generation_eval.py`
Message: `feat: emit potency prior and dump per-pair check-2 predictions`

---

### Task 4: MOA join from the GDSC compound table

`docs/tahoe_generation_results.md:139-140` records the MOA annotation as "not in the current
context map". It is present locally at
`data/raw/gdsc2_sarcoma/gdsc2/screened_compounds_rel_8.5.csv` -- 621 compounds -- under the
current GDSC column names `TARGET` and `TARGET_PATHWAY`, not the `PUTATIVE_TARGET` /
`PATHWAY_NAME` the doc cites. Tahoe drug names vary in case and hyphenation (`crizotinib`,
`5-Fluorouracil`, `AZD-8055`), so the join needs normalization.

**Files:**
- Create: `src/fmharness/moa.py`
- Test: `tests/test_moa.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `normalize_drug(name: str) -> str` -- lowercase, strip every non-alphanumeric character.
  - `load_moa(path: Path) -> pd.DataFrame` -- indexed by normalized drug key, columns
    `drug_name`, `target`, `target_pathway`.
  - `pathway_map(moa: pd.DataFrame, drugs: Iterable[str]) -> dict[str, str]` -- maps each input
    drug name (as written) to its pathway, omitting unmatched drugs.
    Used by Tasks 5, 6 and 7.

- [ ] **Step 1: Write the failing test**

Create `tests/test_moa.py`:

```python
"""Tests for the GDSC mechanism-of-action join."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fmharness.moa import load_moa, normalize_drug, pathway_map


def test_normalize_drug_strips_case_and_punctuation() -> None:
    assert normalize_drug("AZD-8055") == "azd8055"
    assert normalize_drug("crizotinib") == "crizotinib"
    assert normalize_drug("5-Fluorouracil") == "5fluorouracil"
    assert normalize_drug("Trametinib (DMSO_TF solvate)") == "trametinibdmsotfsolvate"


def test_load_moa_and_pathway_map(tmp_path: Path) -> None:
    src = tmp_path / "compounds.csv"
    pd.DataFrame(
        {
            "DRUG_ID": [1, 2, 3],
            "SCREENING_SITE": ["MGH", "MGH", "WTSI"],
            "DRUG_NAME": ["Crizotinib", "AZD8055", "Trametinib"],
            "SYNONYMS": ["PF-02341066", "-", "GSK1120212"],
            "TARGET": ["MET, ALK", "mTORC1, mTORC2", "MEK1, MEK2"],
            "TARGET_PATHWAY": ["RTK signalling", "PI3K/MTOR signalling", "ERK MAPK signalling"],
        }
    ).to_csv(src, index=False)

    moa = load_moa(src)
    assert moa.loc["crizotinib", "target_pathway"] == "RTK signalling"
    assert moa.loc["azd8055", "target"] == "mTORC1, mTORC2"

    # Tahoe writes these names differently; the normalized join still lands them.
    pw = pathway_map(moa, ["crizotinib", "AZD-8055", "Trametinib", "LJI308"])
    assert pw["crizotinib"] == "RTK signalling"
    assert pw["AZD-8055"] == "PI3K/MTOR signalling"
    assert pw["Trametinib"] == "ERK MAPK signalling"
    assert "LJI308" not in pw  # unmatched drugs are omitted, not mapped to a sentinel
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_moa.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmharness.moa'`

- [ ] **Step 3: Write minimal implementation**

Create `src/fmharness/moa.py`:

```python
"""Mechanism-of-action annotation for GDSC compounds.

Selection gap@k is drug-level and mechanism-blind: two representations can post the same
delta-AUC while shortlisting mechanistically different compounds. Joining each drug to its
target pathway lets the audit ask the clinical question -- did the shortlist contain the right
pathway, not the right molecule -- and lets the interaction be split by class, since targeted
agents are line-specific by biology and broad cytotoxics are not.

Source: ``data/raw/gdsc2_sarcoma/gdsc2/screened_compounds_rel_8.5.csv`` (GDSC release 8.5,
621 compounds), columns ``TARGET`` and ``TARGET_PATHWAY``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def normalize_drug(name: str) -> str:
    """Lowercase and strip every non-alphanumeric character.

    Tahoe, GDSC and sci-Plex spell the same compound differently (``crizotinib`` vs
    ``Crizotinib``, ``AZD-8055`` vs ``AZD8055``), so joins key on this instead of the raw name.
    """
    return _NON_ALNUM.sub("", str(name).lower())


def load_moa(path: Path) -> pd.DataFrame:
    """Load the GDSC screened-compounds table, indexed by normalized drug key.

    Duplicate keys (the same compound screened at more than one site) collapse to the first
    row; the target annotation does not vary by site.
    """
    raw = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "drug_name": raw["DRUG_NAME"].astype(str),
            "target": raw["TARGET"].astype(str),
            "target_pathway": raw["TARGET_PATHWAY"].astype(str),
        }
    )
    out.index = pd.Index(out["drug_name"].map(normalize_drug), name="key")
    return out[~out.index.duplicated(keep="first")]


def pathway_map(moa: pd.DataFrame, drugs: Iterable[str]) -> dict[str, str]:
    """Map each drug name, as written by the caller, to its target pathway.

    Unmatched drugs are omitted rather than mapped to a sentinel, so a caller counting
    coverage sees the true join rate.
    """
    lookup = moa["target_pathway"].to_dict()
    pairs = ((d, lookup.get(normalize_drug(d))) for d in drugs)
    return {d: pw for d, pw in pairs if pw is not None}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_moa.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Check the real join rate against the 33 Tahoe drugs**

Run:

```bash
uv run python -c "
from pathlib import Path
from fmharness.moa import load_moa, pathway_map
moa = load_moa(Path('data/raw/gdsc2_sarcoma/gdsc2/screened_compounds_rel_8.5.csv'))
drugs = [l.split(chr(9))[0] for l in Path('data/static/tahoe_pert_to_cid.tsv').read_text().splitlines() if l.strip()]
pw = pathway_map(moa, drugs)
print(f'{len(pw)}/{len(drugs)} Tahoe drugs joined')
print('unmatched:', sorted(set(drugs) - set(pw)))
"
```

This needs `context_by_drug/pert_to_cid.tsv` from Alpine first:
`./scripts/alpine/ralpine pull /projects/lgillenwater@xsede.org/repositories/fm-pdo-evaluator/context_by_drug/pert_to_cid.tsv data/static/tahoe_pert_to_cid.tsv`

Expected: a high join rate. Record the unmatched names -- `Trametinib (DMSO_TF solvate)` will
not match and should be folded onto `Trametinib` (both are CID 11707110) in Task 6.

- [ ] **Step 6: Report the commit (do not run it)**

Intended: `git add src/fmharness/moa.py tests/test_moa.py data/static/tahoe_pert_to_cid.tsv`
Message: `feat: add GDSC mechanism-of-action join`

---

### Task 5: MOA hit-rate@k

The clinical question is right pathway, not right compound: me-too compounds collapse, and if the
base embedding's positive interaction is real it should hit the correct pathway more often than
PCA at equal delta-AUC (`docs/tahoe_generation_results.md:131-134`).

**Files:**
- Modify: `src/fmharness/moa.py`
- Test: `tests/test_moa.py`

**Interfaces:**
- Consumes: `pathway_map` from Task 4.
- Produces: `moa_hit_rate_at_k(preds: pd.DataFrame, pathway: dict[str, str], ks: tuple[int, ...] = (1, 3, 5)) -> dict[int, float]`
  -- share of lines whose top-k shortlist (ranked by ascending `y_pred`) contains a drug sharing
  the pathway of that line's true-best drug. Lines whose true-best drug has no pathway
  annotation are skipped. Used by Tasks 6 and 7.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_moa.py`:

```python
import numpy as np

from fmharness.moa import moa_hit_rate_at_k


def test_moa_hit_rate_at_k_counts_pathway_not_compound() -> None:
    # Line A's true best is d_mek1 (MEK). The model ranks d_mek2 first -- wrong compound,
    # right pathway -- so it is a hit at k=1. Line B's true best is d_mek1 too, but the
    # model ranks the RTK compound first, so B only hits once k reaches 2.
    preds = pd.DataFrame(
        {
            "patient": ["A", "A", "A", "B", "B", "B"],
            "drug": ["d_mek1", "d_mek2", "d_rtk"] * 2,
            "y_true": [0.1, 0.5, 0.9, 0.1, 0.5, 0.9],
            "y_pred": [0.5, 0.1, 0.9, 0.5, 0.9, 0.1],
        }
    )
    pathway = {"d_mek1": "ERK MAPK", "d_mek2": "ERK MAPK", "d_rtk": "RTK signalling"}
    hits = moa_hit_rate_at_k(preds, pathway, ks=(1, 2))
    assert np.isclose(hits[1], 0.5)  # A hits, B misses
    assert np.isclose(hits[2], 1.0)  # both hit


def test_moa_hit_rate_at_k_skips_unannotated_best_drug() -> None:
    preds = pd.DataFrame(
        {
            "patient": ["A", "A", "B", "B"],
            "drug": ["d_x", "d_mek1", "d_mek1", "d_mek2"],
            "y_true": [0.1, 0.9, 0.1, 0.5],
            "y_pred": [0.1, 0.9, 0.1, 0.5],
        }
    )
    # A's true best (d_x) has no pathway -> A is skipped; only B counts, and B hits.
    hits = moa_hit_rate_at_k(preds, {"d_mek1": "ERK MAPK", "d_mek2": "ERK MAPK"}, ks=(1,))
    assert np.isclose(hits[1], 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_moa.py -v`
Expected: FAIL with `ImportError: cannot import name 'moa_hit_rate_at_k'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/fmharness/moa.py`:

```python
def moa_hit_rate_at_k(
    preds: pd.DataFrame, pathway: dict[str, str], ks: tuple[int, ...] = (1, 3, 5)
) -> dict[int, float]:
    """Share of lines whose top-k shortlist contains the true-best drug's pathway.

    ``y_pred`` is AUC-like, so shortlists rank ascending. Unlike gap@k this credits a
    mechanistically correct pick even when the compound is wrong, which is the clinical
    question and which collapses me-too compounds. Lines whose observed best drug carries no
    pathway annotation are skipped rather than counted as misses.
    """
    df = preds.copy()
    df["pathway"] = df["drug"].map(pathway)
    best_pw = df.loc[df.groupby("patient")["y_true"].idxmin()].set_index("patient")["pathway"]
    ranked = df.sort_values(["patient", "y_pred"], kind="stable")
    ranked["rank"] = ranked.groupby("patient").cumcount()
    ranked["want"] = ranked["patient"].map(best_pw)
    scored = ranked[ranked["want"].notna()]
    match = scored["pathway"].eq(scored["want"])
    return {
        k: float(
            match.where(scored["rank"] < k, other=False)
            .groupby(scored["patient"])
            .any()
            .mean()
        )
        for k in ks
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_moa.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/fmharness/moa.py tests/test_moa.py && uv run pyright src/fmharness/moa.py`
Expected: no errors. If the dict comprehension line exceeds 100 characters, break it across
lines rather than adding a `noqa`.

- [ ] **Step 6: Report the commit (do not run it)**

Intended: `git add src/fmharness/moa.py tests/test_moa.py`
Message: `feat: add MOA hit-rate@k readout`

---

### Task 6: MOA-stratified interaction and the shuffled-shortlist control

Two things the spec requires that hit-rate@k alone does not give. First, the mechanistic
corroboration (`docs/tahoe_generation_results.md:134-137`): targeted agents are line-specific by
biology and broad cytotoxics are not, so if the base embedding's +0.119 is biological its edge
should concentrate in targeted pathways and vanish in cytotoxics. If it is flat across both, the
signal is likely non-biological. Second, the saturation control (`:138-140`): with a panel this
pan-active, a shortlist drawn at random already hits the right pathway some of the time, so a raw
hit-rate reads as skill when it is base rate.

**Files:**
- Modify: `src/fmharness/moa.py`
- Test: `tests/test_moa.py`

**Interfaces:**
- Consumes: `moa_hit_rate_at_k` and `pathway_map` from Tasks 4-5; `interaction_rho` from
  `src/fmharness/evaluation.py`.
- Produces:
  - `CYTOTOXIC_PATHWAYS: frozenset[str]` -- GDSC `TARGET_PATHWAY` values treated as broad
    cytotoxics.
  - `interaction_by_moa_class(preds: pd.DataFrame, pathway: dict[str, str]) -> dict[str, float]`
    -- `interaction_rho` computed separately over targeted and cytotoxic drugs; keys
    `targeted`, `cytotoxic`, `n_targeted`, `n_cytotoxic`.
  - `shuffled_hit_rate(preds: pd.DataFrame, pathway: dict[str, str], ks: tuple[int, ...], n_perm: int = 200, seed: int = 0) -> dict[int, float]`
    -- mean hit-rate@k when each line's shortlist order is permuted, i.e. the pan-active base
    rate. Used by Task 7.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_moa.py`:

```python
from fmharness.moa import CYTOTOXIC_PATHWAYS, interaction_by_moa_class, shuffled_hit_rate


def test_cytotoxic_pathways_are_gdsc_spellings() -> None:
    # These must match TARGET_PATHWAY values verbatim or the split silently empties.
    assert "DNA replication" in CYTOTOXIC_PATHWAYS
    assert "Mitosis" in CYTOTOXIC_PATHWAYS
    assert "ERK MAPK signalling" not in CYTOTOXIC_PATHWAYS


def test_interaction_by_moa_class_splits_the_panel() -> None:
    # d_mek/d_rtk are targeted, d_dna is cytotoxic. The targeted subset is predicted with the
    # correct line ordering; the cytotoxic subset is predicted backwards.
    preds = pd.DataFrame(
        {
            "patient": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "drug": ["d_mek", "d_rtk", "d_dna"] * 3,
            "y_true": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "y_pred": [0.1, 0.2, 0.9, 0.4, 0.5, 0.6, 0.7, 0.8, 0.3],
        }
    )
    pathway = {
        "d_mek": "ERK MAPK signalling",
        "d_rtk": "RTK signalling",
        "d_dna": "DNA replication",
    }
    out = interaction_by_moa_class(preds, pathway)
    assert out["n_targeted"] == 2.0
    assert out["n_cytotoxic"] == 1.0
    assert np.isfinite(out["targeted"])


def test_shuffled_hit_rate_is_a_base_rate_not_zero() -> None:
    # Every drug shares one pathway, so a random shortlist always contains it: base rate 1.0.
    preds = pd.DataFrame(
        {
            "patient": ["A", "A", "B", "B"],
            "drug": ["d1", "d2", "d1", "d2"],
            "y_true": [0.1, 0.2, 0.2, 0.1],
            "y_pred": [0.1, 0.2, 0.2, 0.1],
        }
    )
    pathway = {"d1": "ERK MAPK signalling", "d2": "ERK MAPK signalling"}
    base = shuffled_hit_rate(preds, pathway, ks=(1,), n_perm=25)
    assert np.isclose(base[1], 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_moa.py -v`
Expected: FAIL with `ImportError: cannot import name 'CYTOTOXIC_PATHWAYS'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/fmharness/moa.py`:

```python
import numpy as np

from fmharness.evaluation import interaction_rho

# GDSC ``TARGET_PATHWAY`` values that denote broad cytotoxic mechanisms rather than a targeted
# agent. Spelling must match the compound table verbatim; Task 6 Step 5 checks that against the
# real file. Line-specific response is expected in the targeted classes and not here, so this
# split is the mechanistic control on any claimed interaction.
CYTOTOXIC_PATHWAYS = frozenset(
    {
        "DNA replication",
        "Mitosis",
        "Cytoskeleton",
        "Genome integrity",
        "Chromatin histone acetylation",
        "Chromatin histone methylation",
        "Chromatin other",
    }
)


def interaction_by_moa_class(
    preds: pd.DataFrame, pathway: dict[str, str]
) -> dict[str, float]:
    """``interaction_rho`` over targeted drugs and over cytotoxic drugs separately.

    Targeted agents are line-specific by biology; broad cytotoxics are not. A representation
    whose interaction is real should concentrate its edge in the targeted subset. A flat split
    means the signal is not tracking mechanism, which points away from a biological
    explanation. Drugs with no pathway annotation are excluded from both subsets.
    """
    pw = preds["drug"].map(pathway)
    cyto = pw.isin(CYTOTOXIC_PATHWAYS)
    targeted = preds[pw.notna() & ~cyto]
    cytotoxic = preds[pw.notna() & cyto]
    return {
        "targeted": float(interaction_rho(targeted, "y_pred")) if len(targeted) else float("nan"),
        "cytotoxic": float(interaction_rho(cytotoxic, "y_pred"))
        if len(cytotoxic)
        else float("nan"),
        "n_targeted": float(targeted["drug"].nunique()),
        "n_cytotoxic": float(cytotoxic["drug"].nunique()),
    }


def shuffled_hit_rate(
    preds: pd.DataFrame,
    pathway: dict[str, str],
    ks: tuple[int, ...] = (1, 3, 5),
    n_perm: int = 200,
    seed: int = 0,
) -> dict[int, float]:
    """Mean hit-rate@k when each line's shortlist order is random: the pan-active base rate.

    On a panel where a few compounds are potent almost everywhere, a random shortlist already
    contains the right pathway a good fraction of the time. Reporting a raw hit-rate against
    zero therefore reads saturation as skill; this is the number it has to beat.
    """
    rng = np.random.default_rng(seed)
    draws = [
        moa_hit_rate_at_k(
            preds.assign(y_pred=rng.permutation(preds["y_pred"].to_numpy())), pathway, ks
        )
        for _ in range(n_perm)
    ]
    return {k: float(np.mean([d[k] for d in draws])) for k in ks}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_moa.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Verify `CYTOTOXIC_PATHWAYS` against the real compound table**

Run:

```bash
uv run python -c "
import pandas as pd
from fmharness.moa import CYTOTOXIC_PATHWAYS
pw = set(pd.read_csv('data/raw/gdsc2_sarcoma/gdsc2/screened_compounds_rel_8.5.csv')['TARGET_PATHWAY'].dropna().astype(str))
print('unmatched constants:', sorted(CYTOTOXIC_PATHWAYS - pw))
print('all pathways:', sorted(pw))
"
```

Expected: `unmatched constants: []`. If any constant does not appear in the file, correct the
spelling to the file's -- a silent mismatch empties the cytotoxic subset and makes the control
vacuous. Review the printed pathway list and move any obviously cytotoxic class that is not yet
in the constant.

- [ ] **Step 6: Lint and typecheck**

Run: `uv run ruff check src/fmharness/moa.py tests/test_moa.py && uv run pyright src/fmharness/moa.py`
Expected: no errors. Move the `numpy` and `interaction_rho` imports to the top of the module
with the others; ruff's `I` rule will flag them otherwise.

- [ ] **Step 7: Report the commit (do not run it)**

Intended: `git add src/fmharness/moa.py tests/test_moa.py`
Message: `feat: add MOA-stratified interaction and shuffled-shortlist base rate`

---

### Task 7: The audit driver, and the metric decision

Produces the three tables Phase 0 exists to produce, and records the decision that Phases 4-6
depend on: is selection scored in raw AUC or in within-drug percentile?

**Files:**
- Create: `scripts/check2_selection_audit.py`
- Modify: `docs/tahoe_generation_results.md` (replace the two `> **Proposed --**` blocks at
  `:126-195` with the measured results)

**Interfaces:**
- Consumes: `results/check2_preds.parquet` (Task 3); `within_drug_percentile` and
  `shortlist_concentration` (Tasks 1-2); `load_moa`, `pathway_map`, `moa_hit_rate_at_k`
  (Tasks 4-5); `regret_norm_at_k` from `src/fmharness/evaluation.py:203`.
- Produces: `results/check2_selection_audit.csv` and printed tables. Consumed by the Phase 4-6
  plans, which inherit whichever metric this task selects.

- [ ] **Step 1: Write the driver**

Create `scripts/check2_selection_audit.py`:

```python
"""Audit the check-2 shortlists: is gap@k in raw AUC a valid selection metric?

Three tables, from the per-pair dump written by ``score_generation_eval.py``:

1. The potency prior (rank drugs by training-fold mean AUC, ignore the cell line) scored with
   the same gap@k on the same folds as every representation. If the models do not beat it,
   their shortlists carry no cell-line information at all.
2. Shortlist concentration -- distinct top-1 picks, modal share, share of picks that are
   broadly active -- against the observed reference. The truth is itself concentrated, so a
   model is only collapsed if it is more concentrated than that row.
3. The same gap@k in within-drug percentile space, where a pan-cytotoxic compound carries no
   advantage.

Usage:
  uv run python scripts/check2_selection_audit.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fmharness.evaluation import regret_norm_at_k
from fmharness.moa import (
    interaction_by_moa_class,
    load_moa,
    moa_hit_rate_at_k,
    pathway_map,
    shuffled_hit_rate,
)
from fmharness.selection import shortlist_concentration, within_drug_percentile

KS = (1, 3, 5)


def _gap_row(frame: pd.DataFrame, pred_col: str, prefix: str = "gap@") -> dict[str, float]:
    # Assign rather than rename: renaming y_true -> y_pred for the prior/observed rows would
    # destroy the y_true column that regret_norm_at_k needs on the other axis.
    gaps = regret_norm_at_k(frame.assign(y_pred=frame[pred_col]), ks=KS)
    return {f"{prefix}{k}": round(gaps[k], 3) for k in KS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", default="results/check2_preds.parquet")
    ap.add_argument(
        "--compounds",
        default="data/raw/gdsc2_sarcoma/gdsc2/screened_compounds_rel_8.5.csv",
    )
    ap.add_argument("--out", default="results/check2_selection_audit.csv")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    preds = pd.read_parquet(repo / args.preds)
    # Trametinib appears twice in the Tahoe drug table (plain and DMSO_TF solvate), both
    # CID 11707110; fold the solvate onto the plain name so MOA and picks are not split.
    preds["drug"] = preds["drug"].str.replace(" (DMSO_TF solvate)", "", regex=False)

    moa = load_moa(repo / args.compounds)
    pathway = pathway_map(moa, sorted(preds["drug"].unique()))
    print(f"MOA join: {len(pathway)}/{preds['drug'].nunique()} drugs annotated\n")

    def _row(name: str, method: str, frame: pd.DataFrame, col: str) -> dict[str, object]:
        scored = frame.assign(y_pred=frame[col])
        conc = shortlist_concentration(frame, pred_col=col)
        pct = within_drug_percentile(frame, cols=("y_true", col))
        moa = moa_hit_rate_at_k(scored, pathway, KS)
        strat = interaction_by_moa_class(scored, pathway)
        return {
            "source": name,
            "method": method,
            **_gap_row(frame, col),
            **_gap_row(pct, col, prefix="pct_gap@"),
            **{f"moa@{k}": round(v, 3) for k, v in moa.items()},
            "int_targeted": round(strat["targeted"], 3),
            "int_cytotoxic": round(strat["cytotoxic"], 3),
            "distinct": conc["distinct"],
            "modal_share": round(float(conc["modal_share"]), 3),
            "broadly_active_share": round(float(conc["broadly_active_share"]), 3),
        }

    rows = [
        _row(str(source), str(method), frame, "y_pred")
        for (source, method), frame in preds.groupby(["source", "method"], sort=True)
    ]

    # The prior and the observed reference: computed once, on any single (source, method)
    # slice, since y_prior and y_true do not vary with the representation.
    first = preds.iloc[0]
    ref = preds[(preds["source"] == first["source"]) & (preds["method"] == first["method"])]
    rows.append(_row("potency_prior", "-", ref, "y_prior"))
    rows.append(_row("observed", "-", ref, "y_true"))

    # The pan-active base rate: what a random shortlist already scores. Any moa@k at or below
    # this is saturation, not skill.
    base = shuffled_hit_rate(ref.assign(y_pred=ref["y_true"]), pathway, KS)
    print("shuffled-shortlist base rate: " + "  ".join(f"moa@{k}={base[k]:.3f}" for k in KS) + "\n")

    table = pd.DataFrame(rows)
    dest = repo / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest, index=False)
    print(table.to_string(index=False))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `uv run python scripts/check2_selection_audit.py`
Expected: three blocks of output and `results/check2_selection_audit.csv`. Every representation
appears, plus a `potency_prior` row and an `observed` row.

- [ ] **Step 3: Read off the two decisions**

Record, in the doc, the answers to:

1. **Does the prior win?** Compare the `potency_prior` `gap@1` to every representation's. The
   doc predicts ~0.06-0.11 for the prior against 0.22-0.36 for the representations
   (`docs/tahoe_generation_results.md:173-175`). If the prior wins, gap@k in raw AUC does not
   measure personalization and Phases 4-6 use the `pct_gap@k` column instead.
2. **Are the models more concentrated than the truth?** Compare each `distinct` and
   `modal_share` to the `observed` row. At 50 lines the expected observed reference is ~6
   distinct drugs (95% band 4-8) with modal share ~0.69 (band 0.58-0.80).

- [ ] **Step 4: Replace the proposal blocks with results**

In `docs/tahoe_generation_results.md`, replace both `> **Proposed --**` blocks (`:126-195`)
with the measured tables and the recorded decision. Keep the reasoning that motivated them --
the concentration trap and the me-too-compound argument are still the interpretation -- but
state them as findings, not proposals. Correct the MOA column names to `TARGET` /
`TARGET_PATHWAY` where the old text says `PUTATIVE_TARGET` / `PATHWAY_NAME`, and delete the
"Why we cannot answer this today" paragraph at `:177-184`, which is now false.

- [ ] **Step 5: Full test suite and lint**

Run: `uv run pytest -q && uv run ruff check src tests && uv run pyright src tests`
Expected: all pass, no errors.

- [ ] **Step 6: Report the commit (do not run it)**

Intended: `git add scripts/check2_selection_audit.py results/check2_selection_audit.csv docs/tahoe_generation_results.md`
Message: `feat: audit check-2 shortlists against the potency prior`

---

## Phase 0 exit criteria

- `results/check2_preds.parquet` exists and covers every representation in the ladder.
- `results/check2_selection_audit.csv` exists with a `potency_prior` row and an `observed` row.
- `docs/tahoe_generation_results.md` states, as a finding, whether the prior beats the
  representations and whether the models are more concentrated than the truth.
- **The selection metric for Phases 4-6 is recorded**: raw-AUC `gap@k`, or within-drug
  percentile `pct_gap@k`.

## Not in this plan

Phases 1-6 of `docs/superpowers/specs/2026-08-06-arm2-harness-validation-design.md` get their own
plans. Phase 1 (sci-Plex drug alignment) is independent of this one and can run concurrently;
Phase 2 (the modular refactor) depends on this plan's outcome, because the readout registry should
be built around the metric this phase selects.
