"""Tests for sci-Plex fine-tune input validation (raw counts, gene panel, name collisions)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from fmharness.sciplex_prep import check_gene_count, check_perturbation_count, check_raw_counts


def test_check_gene_count_raises_on_a_subset_panel() -> None:
    with pytest.raises(SystemExit, match="2000 genes"):
        check_gene_count(2000, min_genes=5000)


def test_check_gene_count_passes_on_a_near_full_panel() -> None:
    check_gene_count(15012, min_genes=5000)  # must not raise


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


def test_check_perturbation_count_warns_below_the_floor(capsys: pytest.CaptureFixture[str]) -> None:
    perts = pd.Series(["AZ", "AZ", "GSK", "GSK", "control"])
    check_perturbation_count(perts, expected_min_distinct=100)
    assert "WARNING" in capsys.readouterr().out


def test_check_perturbation_count_silent_above_the_floor(capsys: pytest.CaptureFixture[str]) -> None:
    perts = pd.Series([f"drug{i}" for i in range(150)])
    check_perturbation_count(perts, expected_min_distinct=100)
    assert capsys.readouterr().out == ""
