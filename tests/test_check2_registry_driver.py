"""Tests for the registry-driven Check-2 driver, against small synthetic fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
import pytest

# scripts/check2_registry_driver.py, importable via pytest's pythonpath = ["scripts"].
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
    # dict[str, Any] (rather than the plain `dict(...)` pyright would otherwise infer as
    # dict[str, DataFrame | Path | dict[str, str] | str | int]) -- **kwargs unpacked below into
    # run_check2's specifically-typed parameters would otherwise fail every one of them.
    kwargs: dict[str, Any] = dict(
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
    # Boolean-mask indexing on a DataFrame is typed Series | DataFrame | Unknown by the pandas
    # stubs (see fmharness.check2's own design_target for the same workaround); narrow it back
    # so `merged` below is a definite DataFrame and `merged["n_before"] - ... == 1` stays a
    # Series[bool] rather than collapsing to a plain `bool` with no `.all()`.
    fixed_before = cast(pd.DataFrame, unfiltered[unfiltered["method"].isin(["hallmark", "proliferation"])])
    fixed_after = cast(pd.DataFrame, filtered[filtered["method"].isin(["hallmark", "proliferation"])])
    merged = fixed_before.merge(
        fixed_after, on=["source", "method"], suffixes=("_before", "_after")
    )
    assert len(merged) == len(fixed_before) == len(fixed_after) > 0
    # merged["n_before"] is itself ambiguously typed (Series | DataFrame | Unknown, same pandas-
    # stubs gap) -- without this cast, the subtraction/comparison chain collapses to a plain
    # `bool` with no .all(), rather than the elementwise Series[bool] the assertion needs.
    n_dropped = cast(pd.Series, merged["n_before"] - merged["n_after"] == 1)
    assert n_dropped.all()


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
