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
    # both declared drugs have a file on disk, matching what generate() would find.
    _write_adata(tmp_path / "BRD-1.h5ad", [[1.0]], ["o1"], ["A"])
    _write_adata(tmp_path / "BRD-2.h5ad", [[1.0]], ["o1"], ["A"])
    gen = PregeneratedStackGenerator(
        tmp_path, {"BRD-1": "D1", "BRD-2": "D2"}, checkpoint_label="test"
    )
    assert gen.context_coverage(["D1", "D2", "D3"]) == {"D1", "D2"}


def test_context_coverage_excludes_declared_drugs_with_no_file_on_disk(tmp_path: Path) -> None:
    _write_adata(tmp_path / "BRD-1.h5ad", [[1.0]], ["o1"], ["A"])
    gen = PregeneratedStackGenerator(
        tmp_path, {"BRD-1": "D1", "BRD-2": "D2"}, checkpoint_label="test"
    )
    # D1 has a file (BRD-1.h5ad); D2 is declared in pert_to_drug but has no file --
    # context_coverage must exclude it, matching what generate() would actually do.
    assert gen.context_coverage(["D1", "D2", "D3"]) == {"D1"}


def test_generate_reads_the_matching_pregenerated_file(tmp_path: Path) -> None:
    _write_adata(tmp_path / "BRD-1.h5ad", [[1.0, 2.0]], ["o1"], ["A", "B"])
    gen = PregeneratedStackGenerator(tmp_path, {"BRD-1": "D1"}, checkpoint_label="test")
    baseline = ad.AnnData(X=np.zeros((1, 2), dtype=np.float32))
    baseline.obs_names = ["o1"]
    out = gen.generate(baseline, "D1")
    assert list(out.obs_names) == ["o1"]
    assert np.allclose(np.asarray(out.X), [[1.0, 2.0]])


def test_generate_restricts_to_the_intersection_of_baseline_and_file_obs_names(
    tmp_path: Path,
) -> None:
    # the generated file covers every line Stack generated for this drug (o1, o2, o3),
    # not just the ones this particular baseline call cares about (o2, o3) -- generate()
    # must return only the rows the caller actually asked about, row-aligned to neither
    # more nor fewer than that intersection.
    _write_adata(
        tmp_path / "BRD-1.h5ad",
        [[1.0], [2.0], [3.0]],
        ["o1", "o2", "o3"],
        ["A"],
    )
    gen = PregeneratedStackGenerator(tmp_path, {"BRD-1": "D1"}, checkpoint_label="test")
    baseline = ad.AnnData(X=np.zeros((2, 1), dtype=np.float32))
    baseline.obs_names = ["o2", "o3"]
    out = gen.generate(baseline, "D1")
    assert set(out.obs_names) == {"o2", "o3"}
    assert "o1" not in out.obs_names
    assert np.allclose(sorted(np.asarray(out.X).ravel()), [2.0, 3.0])


def test_generate_raises_when_baseline_shares_no_obs_names_with_the_file(
    tmp_path: Path,
) -> None:
    _write_adata(tmp_path / "BRD-1.h5ad", [[1.0, 2.0]], ["o1"], ["A", "B"])
    gen = PregeneratedStackGenerator(tmp_path, {"BRD-1": "D1"}, checkpoint_label="test")
    baseline = ad.AnnData(X=np.zeros((1, 2), dtype=np.float32))
    baseline.obs_names = ["0"]  # disjoint from the file's obs_names ("o1")
    with pytest.raises(PerturbationNotInContext):
        gen.generate(baseline, "D1")


def test_generate_handles_space_sanitized_filenames(tmp_path: Path) -> None:
    # stack-generation sanitizes spaces in the split name to underscores when writing.
    _write_adata(tmp_path / "Retinoic_acid.h5ad", [[5.0]], ["o1"], ["A"])
    gen = PregeneratedStackGenerator(
        tmp_path, {"Retinoic acid": "D1"}, checkpoint_label="test"
    )
    baseline = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32))
    baseline.obs_names = ["o1"]
    out = gen.generate(baseline, "D1")
    assert np.allclose(np.asarray(out.X), [[5.0]])


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
