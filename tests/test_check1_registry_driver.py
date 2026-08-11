"""Tests for the registry-driven Check-1 driver, against small synthetic fixtures."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

# scripts/check1_registry_driver.py, importable via pytest's pythonpath = ["scripts"].
from check1_registry_driver import corpus_declared_partially, parse_corpus_set, run_check1


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
    base = pd.DataFrame(
        rng.standard_normal((4, 3)) + 10.0, columns=pd.Index(genes), index=pd.Index(lines)
    )

    # query_baseline is what build_generated_deltas reads from disk (the real driver's
    # --query-baseline path) -- same values as `base` here, since the synthetic fixture only
    # needs the wiring to be correct, not a realistic raw-counts-vs-CPM distinction.
    query_baseline = tmp_path / "query_baseline.h5ad"
    _write_adata(query_baseline, base.to_numpy().tolist(), lines, genes)

    gdir = tmp_path / "generated"
    gdir.mkdir()
    # build_generated_deltas scores the LOG fold change logcpm(generated) - logcpm(baseline), so
    # baseline * exp(real_delta) is the exact-recovery construction (up to a per-row library-size
    # constant, which Pearson-across-genes ignores). A plain additive offset instead recovers
    # log(1 + delta/baseline), whose per-gene distortion by the baseline's own level caps r near
    # 0.88 -- a much weaker test of "the driver really read this file".
    generated_vals = base.to_numpy() * np.exp(real_delta.to_numpy())
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


def test_run_check1_keeps_delta_rows_aligned_to_the_filtered_key(tmp_path: Path) -> None:
    # filter_leakage RESETS the surviving design's index, so the kept rows are labelled
    # 0..n_kept-1 -- selecting the real delta by those labels silently returns the FIRST
    # n_kept rows (a different set of cell lines) whenever anything was dropped. Under that
    # misalignment the stack source's near-perfect delta would be scored against the wrong
    # lines' truth and r would collapse; keeping r high proves the driver realigns positionally.
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
    stack_row = table[table["source"] == "stack"].iloc[0]
    assert stack_row["n_pairs"] == 3
    assert stack_row["r"] > 0.9


def test_parse_corpus_set_strips_whitespace_and_drops_empties() -> None:
    # Task 9's workflow has a human copy-paste a comma-separated list into these flags --
    # a stray space after a comma (or a trailing comma) must not produce a corpus entry
    # like " B" or "" that can never match a real line/drug name.
    assert parse_corpus_set(" A , B ,") == {"A", "B"}


def test_parse_corpus_set_passes_none_through() -> None:
    assert parse_corpus_set(None) is None


def test_corpus_declared_partially_rejects_a_half_declared_corpus() -> None:
    # filter_leakage only filters when BOTH pretraining_lines and pretraining_drugs are given
    # -- a half-declared corpus (e.g. only --corpus-lines) silently scores identically to an
    # unfiltered run, with nothing in the output to show the declared corpus was ignored.
    # main() must reject this combination via ap.error before it reaches filter_leakage.
    assert corpus_declared_partially("L1", None) is True
    assert corpus_declared_partially(None, "d1") is True
    assert corpus_declared_partially("L1", "d1") is False
    assert corpus_declared_partially(None, None) is False


def test_run_check1_prints_the_leakage_basis_when_a_corpus_is_declared(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    real_delta, real_key, base, query_baseline, gdir, pert_to_drug = _fixture(tmp_path)
    run_check1(
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
    out = capsys.readouterr().out
    assert "basis=measured" in out
    assert "doubly_exposed_frac" in out


def test_run_check1_prints_unknown_basis_without_a_declared_corpus(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    real_delta, real_key, base, query_baseline, gdir, pert_to_drug = _fixture(tmp_path)
    run_check1(
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
    out = capsys.readouterr().out
    assert "basis=unknown" in out
