"""Integration tests for rung 2's per-cell scorer, run against a small synthetic plan dir.

These exercise the SHIPPED script end to end (not a reimplementation of its logic), because a
test that reimplements the branch it is meant to guard passes even when the real branch crashes
-- which is exactly what happened on the cluster: array cell 15 (``shuffled|in_platform``) died
on ``ValueError: no target organoids have a usable baseline`` while the unit test added in the
same commit (a local reimplementation) stayed green.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rung2_score_one  # noqa: E402

SOURCES = ("prior", "knn", "pca", "nmf", "observed_delta", "shuffled", "planted")
ARMS = ("in_platform", "cross_platform", "bulk_target")


def _write_synthetic_plan(out: Path, *, n_lines: int, n_drugs: int, n_genes: int,
                           n_l1000_only: int, n_overlap: int, n_bulk: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    lines = [f"L{i}" for i in range(n_lines)]
    drugs = [f"D{i}" for i in range(n_drugs)]
    genes = [f"G{i}" for i in range(n_genes)]

    t_base = pd.DataFrame(rng.normal(size=(n_lines, n_genes)), index=pd.Index(lines), columns=genes)
    drug_mean = {dr: rng.normal(scale=2.0, size=n_genes) for dr in drugs}

    rows, key_rows = [], []
    for ln in lines:
        for dr in drugs:
            rows.append(drug_mean[dr] + rng.normal(scale=0.3, size=n_genes))
            key_rows.append((ln, dr))
    t_delta = pd.DataFrame(np.asarray(rows), columns=genes)
    t_key = pd.DataFrame({
        "patient": [r[0] for r in key_rows], "drug": [r[1] for r in key_rows],
        "line": [r[0] for r in key_rows], "dname": [r[1] for r in key_rows],
    })

    l_lines = [f"M{i}" for i in range(n_l1000_only)] + lines[:n_overlap]
    l_base = pd.DataFrame(rng.normal(size=(len(l_lines), n_genes)), index=pd.Index(l_lines), columns=genes)
    l_rows, l_key_rows = [], []
    for ln in l_lines:
        for dr in drugs:
            l_rows.append(drug_mean[dr] + rng.normal(scale=0.3, size=n_genes))
            l_key_rows.append((ln, dr))
    l_delta = pd.DataFrame(np.asarray(l_rows), columns=genes)
    l_key = pd.DataFrame({"patient": [r[0] for r in l_key_rows], "drug": [r[1] for r in l_key_rows]})

    t_delta.to_parquet(out / "tahoe_delta.parquet")
    t_key.to_parquet(out / "tahoe_key.parquet")
    t_base.to_parquet(out / "tahoe_base.parquet")
    l_delta.to_parquet(out / "l1000_delta.parquet")
    l_key.to_parquet(out / "l1000_key.parquet")
    l_base.to_parquet(out / "l1000_base.parquet")

    if n_bulk:
        bulk_lines = lines[:n_bulk]
        bulk_base = pd.DataFrame(
            t_base.loc[bulk_lines].to_numpy() * 0.7 + rng.normal(scale=0.5, size=(n_bulk, n_genes)),
            index=pd.Index(bulk_lines), columns=genes,
        )
        bulk_base.to_parquet(out / "bulk_base.parquet")

    grid = [f"{s}|{a}" for s in SOURCES for a in ARMS]
    (out / "plan.json").write_text(json.dumps({
        "git_sha": "test", "slurm_job_id": "local", "panel_size": n_genes, "grid": grid,
        "n_l1000_lines": len(l_lines), "n_l1000_pairs": len(l_key), "n_tahoe_pairs": len(t_key),
        "shared_drugs": drugs, "args": {},
    }))


def _run_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plan_dir: Path, cell: str) -> dict:
    out_dir = tmp_path / "parts"
    argv = [
        "rung2_score_one.py", "--plan-dir", str(plan_dir), "--cell", cell,
        "--out-dir", str(out_dir), "--n-perm", "40",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rung2_score_one.main()
    fname = f"part_{cell.replace('|', '__')}.csv"
    return pd.read_csv(out_dir / fname).iloc[0].to_dict()


@pytest.fixture
def small_plan(tmp_path: Path) -> Path:
    # 6 lines / 5 folds forces a singleton fold -- the exact shape that killed the cluster's
    # shuffled|in_platform cell (fold held one line, and the old relabelling scheme matched it
    # only with probability len(fold)/len(all_lines)).
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    _write_synthetic_plan(
        # score_pairs requires >=50 finite overlapping genes per pair (rung2_score_one.py's
        # own floor against noisy tiny-panel correlations), so n_genes must clear that.
        plan_dir, n_lines=6, n_drugs=3, n_genes=60, n_l1000_only=4, n_overlap=2, n_bulk=5, seed=0,
    )
    return plan_dir


def test_shuffled_in_platform_scores_every_line_not_just_one(small_plan, tmp_path, monkeypatch) -> None:
    row = _run_cell(tmp_path, monkeypatch, small_plan, "shuffled|in_platform")
    # 6 lines x 3 drugs = 18 pairs. The version that killed cluster array cell 15 either raised
    # ValueError or (before that fix) silently scored only 1 line per fold (a handful of pairs).
    assert row["n_pairs"] == 18


def test_shuffled_bulk_target_differs_from_the_real_fit(small_plan, tmp_path, monkeypatch) -> None:
    # bulk_target previously had no shuffled branch and fell through to the SAME call as pca,
    # so the negative control was byte-identical to the model it was meant to null-test.
    shuffled = _run_cell(tmp_path, monkeypatch, small_plan, "shuffled|bulk_target")
    pca = _run_cell(tmp_path, monkeypatch, small_plan, "pca|bulk_target")
    assert shuffled["mean_rho"] != pca["mean_rho"]


def test_bulk_target_does_not_leak_the_scored_lines_own_delta(small_plan, tmp_path, monkeypatch) -> None:
    # observed_delta/prior route to build_additive_deltas, which is line-independent by
    # construction, so its bulk_target score must exactly equal a properly-folded in_platform
    # score fit on the exact same held-out partition -- any gap proves the fit saw data it
    # should not have.
    bulk = _run_cell(tmp_path, monkeypatch, small_plan, "observed_delta|bulk_target")
    in_platform = _run_cell(tmp_path, monkeypatch, small_plan, "observed_delta|in_platform")
    # bulk_target restricts to lines with a GDSC2 bulk profile (5 of 6 here); in_platform scores
    # all 6. observed_delta's prediction does not depend on which baseline arm queried, so on
    # the shared lines the two arms' PER-PAIR predictions come from the same fold fit -- compare
    # via a tolerance loose enough for the different scored-pair sets, tight enough to catch the
    # in-sample leak this test exists to catch (leaked fit inflated mean_rho by +0.1-0.3 in the
    # larger synthetic reproduction during development).
    assert abs(bulk["mean_rho"] - in_platform["mean_rho"]) < 0.15


def test_planted_positive_control_is_actually_fitted(small_plan, tmp_path, monkeypatch) -> None:
    # The planted truth is a known function of the target baseline. A pipeline that is shown the
    # PLANTED delta as its fit target (not the real one) must recover it well above the
    # unfitted floor -- this is exactly what did not hold before: planted scored near zero
    # (mean_rho ~ -0.004 to -0.008) because every arm fit on the REAL delta and was only
    # evaluated against the planted one.
    for arm in ARMS:
        row = _run_cell(tmp_path, monkeypatch, small_plan, f"planted|{arm}")
        assert row["mean_rho"] > 0.5, f"planted control not recovered under {arm}: {row}"


def test_planted_null_is_not_degenerate(small_plan, tmp_path, monkeypatch) -> None:
    # The old single-global-gene-direction design made every row of the planted truth parallel
    # to the same vector, so ANY two rows -- matched or mismatched -- correlated at +-1 and the
    # null itself sat near 1 in magnitude, making "clear the null" impossible even for a perfect
    # fit. Per-drug directions fix this: the null should sit near zero, like every other source.
    row = _run_cell(tmp_path, monkeypatch, small_plan, "planted|in_platform")
    assert abs(row["null_mean"]) < 0.5, f"planted's null is still degenerate: {row}"
