"""Tests for sci-Plex fine-tune input validation (raw counts, gene panel, name collisions)."""

from __future__ import annotations

import pytest

from fmharness.sciplex_prep import check_gene_count


def test_check_gene_count_raises_on_a_subset_panel() -> None:
    with pytest.raises(SystemExit, match="2000 genes"):
        check_gene_count(2000, min_genes=5000)


def test_check_gene_count_passes_on_a_near_full_panel() -> None:
    check_gene_count(15012, min_genes=5000)  # must not raise
