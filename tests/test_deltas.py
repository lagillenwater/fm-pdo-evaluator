"""Tests for the shared L1000 builders (the cmapPy gctx path is Alpine-only)."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from fmharness.deltas import (
    build_additive_deltas,
    build_generated_deltas,
    build_knn_deltas,
    build_learned_deltas,
    build_tahoe_de_calls,
    build_tahoe_deltas,
    drug_pert_maps,
    learned_gene_panel,
    load_pert_map,
    logcpm,
    loo_baseline_source,
    pseudobulk_de_to_deltas,
    restrict_common_support,
)


def test_logcpm_is_scale_invariant() -> None:
    # two rows with the same relative profile but different depth -> identical log-CPM
    df = pd.DataFrame([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], columns=pd.Index(list("abc")))
    z = logcpm(df)
    assert np.allclose(z.iloc[0].to_numpy(), z.iloc[1].to_numpy())


def test_drug_pert_maps_cid_and_inchikey() -> None:
    # D3 carries a CID (777) whose CID has no matching pert, but its InChIKey
    # prefix does -- so it must still resolve, keyed by its CID. D2's CID matches
    # neither a pert CID nor InChIKey. Drugs are keyed by PubChem CID (string).
    drugs = pd.DataFrame(
        {
            "improve_drug_id": ["D1", "D2", "D3"],
            "pubchem_id": [123, 999999, 777],
            "InChIKey": ["AAAAAAAAAAAAAA-x", "BBB-y", "CCCCCCCCCCCCCC-z"],
        }
    )
    pert = pd.DataFrame(
        {
            "pert_type": ["trt_cp", "trt_cp", "ctl_vehicle"],
            "pubchem_cid": [123, 0, 5],
            "inchi_key_prefix": ["ZZZ", "CCCCCCCCCCCCCC", "QQ"],
            "pert_id": ["BRD-A", "BRD-C", "BRD-V"],
        }
    )
    drug2pert, pert2drug = drug_pert_maps(drugs, pert)
    assert drug2pert["123"] == "BRD-A"  # matched by PubChem CID 123
    assert drug2pert["777"] == "BRD-C"  # matched by 14-char InChIKey prefix
    assert "999999" not in drug2pert  # no CID / InChIKey match
    assert pert2drug["BRD-A"] == "123"


def test_load_pert_map_reads_tab_separated_pert_id_to_cid(tmp_path: Path) -> None:
    p = tmp_path / "pert_map.tsv"
    p.write_text("BRD-1\tD1\nBRD-2\tD2\n")
    assert load_pert_map(p) == {"BRD-1": "D1", "BRD-2": "D2"}


def test_load_pert_map_skips_blank_or_malformed_lines(tmp_path: Path) -> None:
    p = tmp_path / "pert_map.tsv"
    p.write_text("BRD-1\tD1\n\nBRD-2\t\n\tD3\n")
    assert load_pert_map(p) == {"BRD-1": "D1"}


def _write_adata(path: Path, x: list[list[float]], obs: list[str], var: list[str]) -> None:
    a = ad.AnnData(X=np.asarray(x, dtype=np.float32))
    a.obs_names = obs
    a.var_names = var
    a.write_h5ad(path)


def test_build_generated_deltas(tmp_path: Path) -> None:
    genes, orgs = ["A", "B", "C"], ["o1", "o2"]
    base = tmp_path / "baseline.h5ad"
    _write_adata(base, [[10, 20, 30], [40, 50, 60]], orgs, genes)
    gdir = tmp_path / "gen"
    gdir.mkdir()
    _write_adata(gdir / "BRD-1.h5ad", [[12, 18, 33], [44, 48, 66]], orgs, genes)  # -> drug D1
    _write_adata(gdir / "BRD-X.h5ad", [[1, 1, 1], [1, 1, 1]], orgs, genes)  # unmapped

    delta, key = build_generated_deltas(gdir, base, {"BRD-1": "D1"}, use_logcpm=False)
    assert set(delta.columns) == {"A", "B", "C"}
    assert delta.shape == (2, 3)  # only BRD-1's 2 organoids
    assert list(key["drug"].unique()) == ["D1"]  # BRD-X skipped
    assert float(delta.loc[delta.index[0], "A"]) == 2.0  # 12 - 10 for o1


def test_build_additive_deltas_is_drug_mean_per_organoid() -> None:
    # two drugs over cell lines L1/L2; the additive delta is each drug's mean over its
    # lines, assigned identically to every organoid (no organoid x drug interaction).
    genes = ["A", "B"]
    l1000_delta = pd.DataFrame(
        [[2.0, 4.0], [4.0, 8.0], [1.0, 1.0], [3.0, 3.0]], columns=pd.Index(genes)
    )
    l1000_key = pd.DataFrame(
        {"patient": ["L1", "L2", "L1", "L2"], "drug": ["d1", "d1", "d2", "d2"]}
    )
    delta, key = build_additive_deltas(l1000_delta, l1000_key, ["o1", "o2", "o3"])

    assert list(delta.columns) == genes
    assert delta.shape == (2 * 3, 2)  # 2 drugs x 3 organoids
    # every organoid gets d1's mean delta [3, 6] and d2's mean delta [2, 2]
    for drug, want in (("d1", [3.0, 6.0]), ("d2", [2.0, 2.0])):
        rows = delta[key["drug"].to_numpy() == drug].to_numpy()
        assert rows.shape == (3, 2)
        assert np.allclose(rows, want)  # organoid-independent
    assert set(key["patient"]) == {"o1", "o2", "o3"}


def test_build_learned_deltas_is_drug_mean_plus_organoid_correction() -> None:
    # learned predictor: delta(organoid, drug) = drug_mean[drug] + correction(organoid).
    # The correction is drug-independent, so within an organoid the difference between
    # two drugs' predicted deltas equals the difference of their drug means -- exactly,
    # regardless of the fitted ridge. And different organoids get different deltas.
    genes = pd.Index(["A", "B", "C", "D"])
    rng = np.random.default_rng(0)
    cells = [f"L{i}" for i in range(6)]
    train_base = pd.DataFrame(rng.random((6, 4)) + 0.5, index=pd.Index(cells), columns=genes)
    keys = [(c, d) for d in ("d1", "d2") for c in cells]
    train_key = pd.DataFrame(keys, columns=pd.Index(["patient", "drug"]))
    dmean = {"d1": np.array([1.0, 2.0, 3.0, 4.0]), "d2": np.array([-1.0, 0.0, 1.0, 2.0])}
    base_arr = train_base.loc[[p for p, _ in keys]].to_numpy()
    delta_rows = [dmean[d] + 0.1 * base_arr[i] for i, (_, d) in enumerate(keys)]
    train_delta = pd.DataFrame(np.asarray(delta_rows), columns=genes)
    target_base = pd.DataFrame(
        rng.random((3, 4)) + 0.5, index=pd.Index(["o1", "o2", "o3"]), columns=genes
    )

    delta, key = build_learned_deltas(
        train_base, train_delta, train_key, target_base, ["o1", "o2", "o3"], reducer="pca", k=3
    )
    assert delta.shape == (2 * 3, 4)  # 2 drugs x 3 organoids
    assert list(delta.columns) == list(genes)

    want_diff = dmean["d1"] - dmean["d2"]
    for p in ("o1", "o2", "o3"):
        d1 = delta[(key["patient"] == p) & (key["drug"] == "d1")].to_numpy()[0]
        d2 = delta[(key["patient"] == p) & (key["drug"] == "d2")].to_numpy()[0]
        assert np.allclose(d1 - d2, want_diff)  # correction cancels -> drug-mean difference
    # organoid-specific: o1 and o2 do not get identical predicted deltas
    o1 = delta[(key["patient"] == "o1") & (key["drug"] == "d1")].to_numpy()[0]
    o2 = delta[(key["patient"] == "o2") & (key["drug"] == "d1")].to_numpy()[0]
    assert not np.allclose(o1, o2)


def test_build_knn_deltas_picks_nearest_line() -> None:
    # 3 training lines with orthogonal baselines; each query points along one line's
    # direction, so its k=1 neighbor (per drug) is that line -> it inherits that line's
    # real delta. This is the cell-specific behavior the drug-agnostic map lacked.
    genes = pd.Index(["A", "B", "C"])
    train_base = pd.DataFrame(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        index=pd.Index(["L1", "L2", "L3"]),
        columns=genes,
    )
    keys = [(c, d) for d in ("d1", "d2") for c in ("L1", "L2", "L3")]
    train_key = pd.DataFrame(keys, columns=pd.Index(["patient", "drug"]))
    per_line = {"L1": [1.0, 0.0, 0.0], "L2": [0.0, 1.0, 0.0], "L3": [0.0, 0.0, 1.0]}
    train_delta = pd.DataFrame([per_line[c] for c, _ in keys], columns=genes)
    # o1 aligns with L2, o2 with L1
    target_base = pd.DataFrame(
        [[0.0, 9.0, 1.0], [9.0, 1.0, 0.0]], index=pd.Index(["o1", "o2"]), columns=genes
    )

    delta, key = build_knn_deltas(
        train_base, train_delta, train_key, target_base, ["o1", "o2"], k=1
    )
    assert delta.shape == (2 * 2, 3)  # 2 drugs x 2 targets
    assert list(delta.columns) == list(genes)
    for d in ("d1", "d2"):
        o1 = delta[(key["patient"] == "o1") & (key["drug"] == d)].to_numpy()[0]
        o2 = delta[(key["patient"] == "o2") & (key["drug"] == d)].to_numpy()[0]
        assert np.allclose(o1, [0.0, 1.0, 0.0])  # nearest L2 -> L2's delta
        assert np.allclose(o2, [1.0, 0.0, 0.0])  # nearest L1 -> L1's delta

    # determinism: identical inputs -> identical output
    d2, _ = build_knn_deltas(train_base, train_delta, train_key, target_base, ["o1", "o2"], k=1)
    assert np.allclose(delta.to_numpy(), d2.to_numpy())


def test_build_tahoe_deltas_pseudobulks_and_logfc() -> None:
    # two cell lines, one drug (CID 100) + DMSO, a few single cells each. The real delta is
    # the log fold-change of the (line, drug) treated pseudobulk vs the line's DMSO pseudobulk.
    genes = ["A", "B", "C"]
    x = np.array(
        [
            [10.0, 0.0, 0.0],  # ACH-1 DMSO
            [20.0, 0.0, 0.0],  # ACH-1 DMSO   -> control mean [15, 0, 0]
            [0.0, 10.0, 0.0],  # ACH-1 CID100
            [0.0, 30.0, 0.0],  # ACH-1 CID100 -> treated mean [0, 20, 0]
            [0.0, 0.0, 10.0],  # ACH-2 DMSO   -> control mean [0, 0, 10]; cell_id empty
            [5.0, 5.0, 0.0],  # ACH-2 CID100 -> treated mean [5, 5, 0]
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {
            "cell_id": ["ACH-1", "ACH-1", "ACH-1", "ACH-1", "", ""],  # last line: no DepMap id
            "cell_line_id": ["CVCL_1", "CVCL_1", "CVCL_1", "CVCL_1", "CVCL_2", "CVCL_2"],
            "pubchem_cid": ["0", "0", "100", "100", "0", "100"],
            "is_control": [True, True, False, False, True, False],
        }
    )
    adata = ad.AnnData(X=x, obs=obs)
    adata.var_names = genes

    delta, key, base = build_tahoe_deltas(adata)

    # baseline = raw pseudobulk mean per line; ACH-2 falls back to its cell_line_id.
    assert set(base.index) == {"ACH-1", "CVCL_2"}
    assert np.allclose(base.loc["ACH-1"].to_numpy(), [15.0, 0.0, 0.0])
    assert np.allclose(base.loc["CVCL_2"].to_numpy(), [0.0, 0.0, 10.0])

    # one row per (line, drug), drug keyed by PubChem CID.
    assert set(map(tuple, key.to_numpy())) == {("ACH-1", "100"), ("CVCL_2", "100")}
    assert list(delta.columns) == genes

    # delta is logcpm(treated) - logcpm(line's own DMSO), computed via the shared logcpm.
    idx = pd.Index(["ACH-1", "CVCL_2"])
    base_lc = logcpm(pd.DataFrame([[15.0, 0, 0], [0, 0, 10.0]], index=idx, columns=pd.Index(genes)))
    trt_lc = logcpm(pd.DataFrame([[0, 20.0, 0], [5.0, 5.0, 0]], index=idx, columns=pd.Index(genes)))
    for p in ("ACH-1", "CVCL_2"):
        row = delta[key["patient"].to_numpy() == p].to_numpy()[0]
        assert np.allclose(row, trt_lc.loc[p].to_numpy() - base_lc.loc[p].to_numpy())


def test_build_tahoe_de_calls_significant_gene_flagged_by_wilcoxon_and_lfc() -> None:
    # one line, one drug; gene A clearly separates control vs treated (Wilcoxon-significant,
    # large log2fc); gene B is flat (not significant). n=6/group gives the rank-sum test enough
    # resolution to reach padj < 0.05 on a complete separation (the minimum possible two-sided
    # exact p-value at n1=n2=6 is ~0.0043, well under 0.05; at n=2/group it could never go below
    # 1/3, so this fixture needs >=6 cells per group, not the 2-per-group used elsewhere in this
    # file for pseudobulk-only tests).
    #
    # gene B's raw count (1000) is held constant across every cell (both groups) but must stay
    # LARGE relative to gene A's swing (1 -> 50): build_tahoe_de_calls library-size-normalizes
    # (CPM) before testing, and in a toy 2-gene panel a small, unchanged B would still get
    # compositionally diluted by A eating up an outsized share of the per-cell library (B=5
    # would drop ~87% in CPM terms purely from A's swing, |log2fc| ~ 3.2 -- a false positive
    # unrelated to Wilcoxon/BH correctness). B=1000 keeps A's swing to <5% of the library, so
    # B's post-normalization log2fc stays under the 0.25 threshold, matching its true "flat"
    # biological signal -- the same compositional effect real-data HVG panels absorb painlessly
    # over thousands of genes, exaggerated here only because the fixture has just two.
    genes = ["A", "B"]
    ctl = np.array([[1.0, 1000.0]] * 6, dtype=np.float32)
    trt = np.array([[50.0, 1000.0]] * 6, dtype=np.float32)
    x = np.vstack([ctl, trt])
    obs = pd.DataFrame(
        {
            "cell_id": ["ACH-1"] * 12,
            "cell_line_id": ["CVCL_1"] * 12,
            "pubchem_cid": ["0"] * 6 + ["100"] * 6,
            "is_control": [True] * 6 + [False] * 6,
        }
    )
    adata = ad.AnnData(X=x, obs=obs)
    adata.var_names = genes

    calls = build_tahoe_de_calls(adata)

    assert set(calls.columns) == {"patient", "drug", "gene", "log2fc", "padj", "significant"}
    assert set(map(tuple, calls[["patient", "drug"]].drop_duplicates().to_numpy())) == {
        ("ACH-1", "100")
    }
    a = calls[calls["gene"] == "A"].iloc[0]
    b = calls[calls["gene"] == "B"].iloc[0]
    assert bool(a["significant"])
    assert a["log2fc"] > 0  # treated > control
    assert a["padj"] < 0.05
    assert not bool(b["significant"])


def test_build_tahoe_de_calls_uses_paper_grounded_default_thresholds() -> None:
    # locks in the exact threshold decision (Methods 4.8's cell-eval LFC/FDR pair -- the only
    # concrete number the paper states anywhere for cell-eval-based DE calling) as an explicit,
    # checkable contract rather than an accidental default.
    import inspect

    sig = inspect.signature(build_tahoe_de_calls)
    assert sig.parameters["lfc_threshold"].default == 0.25
    assert sig.parameters["fdr_threshold"].default == 0.05


def test_pseudobulk_de_to_deltas_pools_doses_and_rekeys() -> None:
    # ACH-1 has two doses (logFC A: 1,3 -> mean 2; B: -1,-3 -> mean -2), ACH-2 one; the
    # 'other' drug row maps to no CID and is dropped from both the delta and the baseline.
    de = pd.DataFrame(
        {
            "gene_name": ["A", "B", "A", "B", "A", "B", "A"],
            "log2FoldChange": [1.0, -1.0, 3.0, -3.0, 2.0, 0.0, 9.0],
            "baseMean": [10.0, 20.0, 10.0, 20.0, 5.0, 5.0, 1.0],
            "Cell_ID_DepMap": ["ACH-1", "ACH-1", "ACH-1", "ACH-1", "ACH-2", "ACH-2", "ACH-2"],
            "drug": ["drugX", "drugX", "drugX", "drugX", "drugX", "drugX", "other"],
        }
    )
    delta, key, base = pseudobulk_de_to_deltas(de, {"drugX": "555"})

    assert set(map(tuple, key.to_numpy())) == {("ACH-1", "555"), ("ACH-2", "555")}
    assert list(delta.columns) == ["A", "B"]
    i1 = key.index[(key["patient"] == "ACH-1") & (key["drug"] == "555")][0]
    assert np.allclose(delta.loc[i1].to_numpy(), [2.0, -2.0])  # pooled over the two doses
    i2 = key.index[(key["patient"] == "ACH-2") & (key["drug"] == "555")][0]
    assert np.allclose(delta.loc[i2].to_numpy(), [2.0, 0.0])  # 'other'-drug logFC 9.0 excluded
    # baseline = mean baseMean per line per gene; the dropped 'other' row does not affect ACH-2/A
    assert np.allclose(base.loc["ACH-1"].to_numpy(), [10.0, 20.0])
    assert np.allclose(base.loc["ACH-2"].to_numpy(), [5.0, 5.0])


def test_loo_baseline_source_additive_never_sees_its_own_held_out_line() -> None:
    # 3 lines, 1 drug each with a distinct delta value; additive's held-out prediction for
    # each line must come only from the OTHER two lines' mean, never its own value.
    real_delta = pd.DataFrame({"A": [10.0, 20.0, 30.0]})
    real_key = pd.DataFrame({"patient": ["L1", "L2", "L3"], "drug": ["d1", "d1", "d1"]})
    base = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=pd.Index(["L1", "L2", "L3"]))

    delta, key = loo_baseline_source("additive", real_delta, real_key, base, k=1)

    assert len(delta) == 3
    want = {"L1": (20.0 + 30.0) / 2, "L2": (10.0 + 30.0) / 2, "L3": (10.0 + 20.0) / 2}
    for line, expected in want.items():
        row = delta.loc[key["patient"].to_numpy() == line]
        assert np.isclose(float(row["A"].iloc[0]), expected)


def test_loo_baseline_source_raises_on_unknown_kind() -> None:
    # 2 lines, so leaving one out still leaves training data behind -- with only 1 line the
    # held-out line's training mask would be all-False and the loop would `continue` before
    # ever reaching the kind dispatch, masking the intended error with a different one.
    real_delta = pd.DataFrame({"A": [1.0, 2.0]})
    real_key = pd.DataFrame({"patient": ["L1", "L2"], "drug": ["d1", "d1"]})
    base = pd.DataFrame({"A": [0.0, 0.0]}, index=pd.Index(["L1", "L2"]))
    with pytest.raises(ValueError, match="unknown baseline source"):
        loo_baseline_source("bogus", real_delta, real_key, base, k=1)


def test_learned_gene_panel_unions_hvgs_and_hallmark_genes(tmp_path: Path) -> None:
    gmt = tmp_path / "hallmark.gmt"
    # load_hallmark only keeps names in its fixed allow-list (see _HALLMARK_DIRECTION);
    # an unrecognized set name would be silently dropped, so use a real one here.
    gmt.write_text("HALLMARK_P53_PATHWAY\thttp://example\tSIGGENE1\tSIGGENE2\n")
    # 4 genes, HVG1 has the highest variance (picked at n_hvg=1); SIGGENE1/SIGGENE2 come in
    # from the hallmark set regardless of their own variance.
    real_delta = pd.DataFrame(
        {
            "HVG1": [1.0, 100.0, -50.0],
            "HVG2": [1.0, 1.0, 1.0],
            "SIGGENE1": [1.0, 1.0, 1.0],
            "SIGGENE2": [1.0, 1.0, 1.0],
        }
    )
    panel = learned_gene_panel(real_delta, gmt, n_hvg=1)
    assert set(panel) == {"HVG1", "SIGGENE1", "SIGGENE2"}


def test_learned_gene_panel_ranks_variance_skipping_nan(tmp_path: Path) -> None:
    gmt = tmp_path / "hallmark.gmt"
    gmt.write_text("HALLMARK_P53_PATHWAY\thttp://example\tSIGGENE1\tSIGGENE2\n")
    # NANVAR has one NaN row but is otherwise the highest-variance column among its non-NaN
    # values; LOWVAR1/2 are constant (zero variance). A naive numpy ndarray.var() propagates the
    # NaN into NANVAR's overall variance (making it NaN, which pandas sort_values pushes to the
    # bottom regardless of ascending=), knocking it out of an n_hvg=1 panel. Pandas' skipna
    # variance -- what the original inline code and the still-standalone hvg line in
    # scripts/score_generation_eval.py both use -- ranks NANVAR first despite the missing value;
    # learned_gene_panel must match that.
    real_delta = pd.DataFrame(
        {
            "NANVAR": [1.0, 1000.0, np.nan],
            "LOWVAR1": [5.0, 5.0, 5.0],
            "LOWVAR2": [3.0, 3.0, 3.0],
            "SIGGENE1": [1.0, 1.0, 1.0],
            "SIGGENE2": [1.0, 1.0, 1.0],
        }
    )
    panel = learned_gene_panel(real_delta, gmt, n_hvg=1)
    assert set(panel) == {"NANVAR", "SIGGENE1", "SIGGENE2"}


def _source(pairs: list[tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # one-column delta whose value encodes the pair's position, so a test can confirm
    # row alignment survives the restriction (not just the row count).
    delta = pd.DataFrame({"g": [float(i) for i in range(len(pairs))]})
    key = pd.DataFrame(pairs, columns=pd.Index(["patient", "drug"]))
    return delta, key


def test_restrict_common_support_keeps_only_shared_labeled_pairs() -> None:
    # "wide" covers 3 patients x 2 drugs, but design only labels 3 of those 6 pairs; "narrow"
    # (like Stack's generated delta) covers just 2 pairs, both labeled -- the real Path-B
    # shape where a broadcast baseline's native coverage is much wider than Stack's but
    # mostly unlabeled, and comparing raw scores across sources with different (patient,
    # drug) support silently compares different evaluation sets, not just different methods.
    wide = _source([("p1", "d1"), ("p2", "d1"), ("p3", "d1"), ("p1", "d2"), ("p2", "d2"), ("p3", "d2")])
    narrow = _source([("p1", "d1"), ("p2", "d1")])
    design = pd.DataFrame(
        {"patient": ["p1", "p2", "p1"], "drug": ["d1", "d1", "d2"], "y": [0.1, 0.2, 0.3]}
    )

    out = restrict_common_support({"wide": wide, "narrow": narrow}, design)

    wide_delta, wide_key = out["wide"]
    assert list(zip(wide_key["patient"], wide_key["drug"])) == [("p1", "d1"), ("p2", "d1")]
    assert wide_delta["g"].tolist() == [0.0, 1.0]  # rows stay aligned to their original pair
    narrow_delta, narrow_key = out["narrow"]
    assert list(zip(narrow_key["patient"], narrow_key["drug"])) == [("p1", "d1"), ("p2", "d1")]
    assert narrow_delta["g"].tolist() == [0.0, 1.0]


def test_restrict_common_support_raises_when_no_shared_pairs() -> None:
    a = _source([("p1", "d1")])
    b = _source([("p2", "d2")])
    design = pd.DataFrame({"patient": ["p1", "p2"], "drug": ["d1", "d2"]})
    with pytest.raises(ValueError, match="no .* shared"):
        restrict_common_support({"a": a, "b": b}, design)
