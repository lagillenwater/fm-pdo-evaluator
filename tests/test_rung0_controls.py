"""Positive and negative controls for rung 0's build, score, and null steps (SPEC rule 4).

Every test runs the REAL functions from scripts/delta_reproducibility.py on synthetic
replicate pools with planted, known answers -- not reimplementations of their logic.
"""

# pandas ships no PEP-561 type stubs in this environment; under strict mode that turns every
# pandas call site into a cascade of reportUnknown* noise about *pandas'* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

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
assert _SPEC is not None and _SPEC.loader is not None
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


def test_build_positive_planted_pool_comes_out_with_the_planted_shape(tmp_path: Path) -> None:
    path = _write_fixture_pool(tmp_path)
    de, chosen = dr.build_split_half_frame(
        [str(path)], ["D0", "D1", "D2"], None, tmp_path / "duck", memory_limit="2GB"
    )
    de = de.dropna(subset=["lfc0", "lfc1"])
    assert chosen == "plate"
    pairs = de.groupby(["patient", "drug"]).ngroups
    assert pairs == 12, f"planted 4 lines x 3 drugs, built {pairs} pairs"
    assert de["gene_name"].nunique() == 300


def test_build_negative_no_replication_yields_no_scoreable_pairs(tmp_path: Path) -> None:
    path = _write_fixture_pool(tmp_path, plates=("P1",))  # one plate: one half stays empty
    de, _ = dr.build_split_half_frame(
        [str(path)], ["D0", "D1", "D2"], None, tmp_path / "duck", memory_limit="2GB"
    )
    assert de.dropna(subset=["lfc0", "lfc1"]).empty


def test_score_positive_planted_reliability_is_recovered(tmp_path: Path) -> None:
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


def test_score_negative_zero_signal_returns_null(tmp_path: Path) -> None:
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


def test_null_positive_planted_components_recover_the_stratum_ordering(tmp_path: Path) -> None:
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


def test_null_negative_signal_free_strata_sit_at_their_floors(tmp_path: Path) -> None:
    path = _write_fixture_pool(tmp_path, signal_sd=0.0, drug_sd=0.0, noise_sd=1.0)
    de, _ = dr.build_split_half_frame(
        [str(path)], ["D0", "D1", "D2"], None, tmp_path / "duck", memory_limit="2GB"
    )
    de = de.dropna(subset=["lfc0", "lfc1"])
    _, piv0, piv1 = dr.score_split_half(de, set(de["gene_name"].unique()))
    nulls = dr.stratified_null_draws(piv0, piv1, n_perm=200, seed=0)
    for stratum, draws in nulls.items():
        assert abs(float(np.mean(draws))) < 0.05, f"{stratum} floor is not ~0 on noise"
