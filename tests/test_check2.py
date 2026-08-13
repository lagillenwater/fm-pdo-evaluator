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
    pd.DataFrame({"a": [1.0, 3.0], "b": [2.0, 4.0]}, index=pd.Index(["L1", "L2"])).to_csv(path)
    df = load_line_matrix(path)
    assert list(df.index) == ["L1", "L2"]
    assert np.allclose(df.to_numpy(), [[1.0, 2.0], [3.0, 4.0]])


def test_load_line_matrix_reads_parquet(tmp_path: Path) -> None:
    path = tmp_path / "emb.parquet"
    pd.DataFrame({"a": [1.0, 3.0], "b": [2.0, 4.0]}, index=pd.Index(["L1", "L2"])).to_parquet(path)
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
