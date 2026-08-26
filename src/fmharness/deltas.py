"""Shared L1000 builders, so the validator, the generated-bridge, and the
viability-adapter runner use one code path (no drift):

- ``build_l1000_gdsc_pairs`` -- real L1000 treated-minus-DMSO deltas paired with
  GDSC2 AUC on shared (cell line, drug) pairs. The validation / supervised-training
  cohort. Reads the Level-3 ``.gctx`` in column chunks (bounded memory); ``cmapPy``
  is imported lazily, so importing this module never requires it (Alpine only).
- ``build_generated_deltas`` -- Stack-generated treated profiles minus the organoid
  baseline, per (organoid, drug). The target cohort. AnnData only.
- ``build_additive_deltas`` -- the non-Stack baseline delta source: each drug's mean
  real L1000 treated-minus-DMSO delta, applied to every organoid (organoid-independent).
  The generation analogue of the drug-mean baseline, so Stack's organoid-specific
  generated delta is compared against "the drug does the same thing everywhere."
- ``build_learned_deltas`` -- PCA/NMF organoid-specific delta predictors: a linear
  baseline -> delta-residual map learned on real L1000, applied to each organoid's
  baseline. The generation analogue of the expression baselines, between the additive
  floor and Stack.

All return a delta frame (rows = samples, columns = gene symbols) plus a key frame
(``patient``, ``drug``) aligned row-for-row, ready for ``score_signatures`` or the
viability adapters -- so any delta source flows through the same readout adapters and
metrics, and the comparison is delta-source vs delta-source on equal footing.
"""

from __future__ import annotations

import re
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from fmharness.adapters import make_penalty
from fmharness.data.loaders import load_tranche
from fmharness.evaluation import build_sample_design
from fmharness.signatures import load_hallmark

# candidate grid for CV-selecting a reducer/neighbor count (build_learned_deltas' n_components,
# build_knn_deltas' k) instead of a hardcoded default -- the same "let CV choose it" principle
# fmharness.adapters.make_penalty already applies to Ridge/Lasso/ElasticNet's penalty strength
# (Kurilov 2020), extended to the reducer's own capacity so kNN/PCA/NMF aren't compared against
# Stack's embedding (or each other) at an arbitrary, undertuned fixed capacity.
_K_GRID: tuple[int, ...] = (2, 3, 5, 10, 15, 20, 30)


def _cv_select_k(
    feat_by_k: Callable[[int], np.ndarray],
    y: np.ndarray,
    candidates: tuple[int, ...],
    seed: int,
) -> int:
    """Pick the candidate ``k`` (from ``candidates``) minimizing 3-fold CV error of a
    CV-tuned ridge fit (``make_penalty("l2")``) on ``(feat_by_k(k), y)``.

    ``feat_by_k`` must build its features from TRAINING data only (never the held-out
    line/patient this whole call is ultimately predicting) -- the caller is responsible for
    that; this function only does the inner selection. Falls back to the smallest candidate
    if every candidate fails to produce a usable CV score (e.g. too few samples).
    """
    n = len(y)
    folds = min(3, n)
    if folds < 2:
        return candidates[0]
    best_k, best_score = candidates[0], -np.inf
    for k in candidates:
        x = feat_by_k(k)
        if x.shape[1] == 0:
            continue
        try:
            scores = cross_val_score(
                make_penalty("l2"),  # type: ignore[arg-type]
                x,
                y,
                cv=KFold(folds, shuffle=True, random_state=seed),
                scoring="r2",
            )
        except ValueError:
            continue
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score, best_k = mean_score, k
    return best_k

PERT_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
    "GSE92742_Broad_LINCS_pert_info.txt.gz"
)


def dense(x: object) -> np.ndarray:
    """AnnData.X may be sparse; return a dense 2-D float array."""
    to_array = getattr(x, "toarray", None)
    arr = to_array() if callable(to_array) else np.asarray(x)
    return np.asarray(arr, dtype=np.float64)


def logcpm(df: pd.DataFrame) -> pd.DataFrame:
    """Library-size normalize (per 10k) and log1p, so a treated-minus-baseline
    difference is a log fold-change rather than a raw-count difference dominated by
    per-sample sequencing depth (which would inflate the random-gene-set baseline)."""
    # copy=True because .to_numpy() may hand back a read-only view -- newer numpy does, and the
    # in-place zero-guard below then raises "assignment destination is read-only". Pre-existing;
    # it does not fire under Alpine's pinned pandas/numpy, so it only ever failed locally.
    lib = df.sum(axis=1).to_numpy(dtype=np.float64, copy=True)
    lib[lib == 0] = 1.0
    return pd.DataFrame(
        np.log1p(df.to_numpy(dtype=np.float64) / lib[:, None] * 1e4),
        index=df.index,
        columns=df.columns,
    )


def _ncid(x: object) -> str:
    try:
        return str(int(float(x)))  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return ""


def _norm(s: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def drug_pert_maps(
    drugs: pd.DataFrame,
    pert_info: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, str]]:
    """``(drug2pert, pert2drug)`` mapping **PubChem CID** (string) to L1000
    ``trt_cp`` pert_ids by PubChem CID or InChIKey prefix. CID is the canonical
    cross-dataset drug key the designs use (build_sample_design
    drug_key='pubchem_cid'); drugs without a CID are skipped. ``drugs`` needs
    columns ``pubchem_id``, ``InChIKey``."""
    cp = pert_info[pert_info["pert_type"] == "trt_cp"]
    cid2p = {_ncid(c): p for c, p in zip(cp["pubchem_cid"], cp["pert_id"], strict=True)}
    ikb2p = {str(k): p for k, p in zip(cp["inchi_key_prefix"], cp["pert_id"], strict=True)}
    drug2pert: dict[str, str] = {}
    pert2drug: dict[str, str] = {}
    for _, r in drugs.drop_duplicates(subset=["pubchem_id", "InChIKey"]).iterrows():
        try:
            cid = str(int(r["pubchem_id"]))  # skips None / NaN (no canonical CID)
        except (TypeError, ValueError):
            continue
        pid = cid2p.get(_ncid(r["pubchem_id"])) or ikb2p.get(str(r["InChIKey"])[:14])
        if pid:
            drug2pert[cid] = pid
            pert2drug[pid] = cid
    return drug2pert, pert2drug


def sarcoma_organoids_2024_pert_map(repo: Path) -> dict[str, str]:
    """pert_id -> Soragni PubChem CID (string) (downloads L1000 pert_info to /tmp)."""
    cache = Path("/tmp/l1000_pert_info.txt.gz")
    if not cache.exists():
        urllib.request.urlretrieve(PERT_INFO_URL, cache)
    pert = pd.read_csv(cache, sep="\t", low_memory=False)
    dr = pd.read_csv(repo / "data/raw/coderdata/sarcoma_drugs.tsv.gz", sep="\t")
    _, ds = build_sample_design(
        load_tranche("sarcoma", repo), "tumor", "viability", drug_key="pubchem_cid"
    )
    sarcoma_organoids_2024_cids = [str(d) for d in ds["drug"]]
    dr_cid = dr["pubchem_id"].map(lambda c: str(int(c)) if pd.notna(c) else None)
    sor = cast("pd.DataFrame", dr[dr_cid.isin(sarcoma_organoids_2024_cids)])
    _, pert2drug = drug_pert_maps(sor, pert)
    return pert2drug


def load_pert_map(path: Path) -> dict[str, str]:
    """Read a ``pert_id<TAB>cid`` TSV into ``{pert_id: cid}`` for build_generated_deltas.

    The generated files are named by Tahoe pert_id (drug name); this maps each back to
    the PubChem CID the real deltas / designs are keyed by. Written by 03's context split.
    """
    m: dict[str, str] = {}
    for line in path.read_text().splitlines():
        pert, _, cid = line.partition("\t")
        if pert.strip() and cid.strip():
            m[pert.strip()] = cid.strip()
    return m


def _drug_of(path: Path, gen: ad.AnnData, valid: set[str]) -> str:
    """Find the pert_id a generated file corresponds to (Stack writes
    ``generated/<pert_id>.h5ad``)."""
    if path.stem in valid:
        return path.stem
    # stack-generation sanitizes spaces in the split name to underscores when it writes the
    # file, so 'Retinoic_acid.h5ad' is really pert_id 'Retinoic acid' -- undo that first.
    if path.stem.replace("_", " ") in valid:
        return path.stem.replace("_", " ")
    # NB: no single-token fallback -- 'Trametinib_DMSO_TF_solvate_' would wrongly match 'Trametinib'
    # and mis-attribute the solvate's delta. Fall back to the pert_id in uns instead.
    for key in ("pert_id", "condition", "drug"):
        v = gen.uns.get(key) if key in gen.uns else None
        if isinstance(v, str) and v in valid:
            return v
    return ""


def build_generated_deltas(
    generated_dir: Path,
    baseline_path: Path,
    pert_to_drug: dict[str, str],
    *,
    use_logcpm: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """delta = generated_treated - baseline per (organoid, drug), gene-aligned.

    Returns ``(delta[pairs x genes], key[patient, drug])``. Files that do not map to
    a drug in ``pert_to_drug`` are skipped (with a note)."""
    base = ad.read_h5ad(baseline_path)
    base_df = pd.DataFrame(
        dense(base.X),
        index=pd.Index([str(o) for o in base.obs_names]),
        columns=pd.Index([str(g) for g in base.var_names]),
    )
    if use_logcpm:
        base_df = logcpm(base_df)
    valid = set(pert_to_drug)
    delta_rows: list[np.ndarray] = []
    keys: list[tuple[str, str]] = []
    genes: pd.Index | None = None
    for f in sorted(Path(generated_dir).glob("*.h5ad")):
        gen = ad.read_h5ad(f)
        pid = _drug_of(f, gen, valid)
        if not pid:
            print(f"  skip {f.name}: no pert_id match")
            continue
        g = pd.DataFrame(
            dense(gen.X),
            index=pd.Index([str(o) for o in gen.obs_names]),
            columns=pd.Index([str(x) for x in gen.var_names]),
        )
        if use_logcpm:
            g = logcpm(g)
        if genes is None:
            genes = base_df.columns.intersection(g.columns)
        orgs = base_df.index.intersection(g.index)
        d = g.loc[orgs, genes].to_numpy() - base_df.loc[orgs, genes].to_numpy()
        for org, row in zip(orgs, d, strict=True):
            delta_rows.append(row)
            keys.append((str(org), pert_to_drug[pid]))
    if genes is None or not delta_rows:
        raise ValueError("no generated files matched a drug; check generated_dir / mapping")
    delta = pd.DataFrame(np.asarray(delta_rows), columns=genes)
    key = pd.DataFrame(keys, columns=pd.Index(["patient", "drug"]))
    # Guard against two files mapping to the same (line, drug) -- keep the first so the
    # downstream per-drug regression never gets duplicate rows for a line.
    dup = key.duplicated(["patient", "drug"]).to_numpy()
    if dup.any():
        print(f"  build_generated_deltas: dropped {int(dup.sum())} duplicate (line, drug) rows")
        delta, key = delta[~dup].reset_index(drop=True), key[~dup].reset_index(drop=True)
    return delta, key


def build_additive_deltas(
    l1000_delta: pd.DataFrame,
    l1000_key: pd.DataFrame,
    patients: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Non-Stack baseline: each drug's mean L1000 delta, broadcast to every organoid.

    Takes real L1000 treated-minus-DMSO deltas (``build_l1000_gdsc_pairs``) and their
    ``(patient, drug)`` key, averages the delta over cell lines per drug, then assigns
    that single per-drug delta to every organoid in ``patients`` -- so the predicted
    delta is organoid-independent. This is the generation analogue of the drug-mean
    baseline: it carries the drug's main transcriptional effect but no organoid x drug
    interaction, the floor Stack's generated delta must beat. Returns ``(delta[pairs x
    genes], key[patient, drug])`` in the same shape as ``build_generated_deltas``.
    """
    drug_mean = l1000_delta.groupby(l1000_key["drug"].to_numpy()).mean()
    drugs = np.asarray(drug_mean.index, dtype=object)
    pats = np.asarray([str(p) for p in patients], dtype=object)
    n_p = len(pats)
    # each drug's delta repeated once per organoid; keys tile organoids within drug.
    delta = pd.DataFrame(np.repeat(drug_mean.to_numpy(), n_p, axis=0), columns=drug_mean.columns)
    key = pd.DataFrame(
        {"patient": np.tile(pats, len(drugs)), "drug": np.repeat(drugs, n_p)},
    )
    return delta, key


def restrict_common_support(
    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    design: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Restrict every delta source to the (patient, drug) pairs ALL of them share -- and
    that carry a real label in ``design`` -- so a head-to-head comparison across sources
    scores every method on the identical evaluation set.

    Sources can have very different native coverage: a broadcast baseline like
    ``build_additive_deltas`` spans every drug in the training cohort (mostly unlabeled
    for a given patient), while ``build_generated_deltas`` only covers whatever was
    actually generated. Scoring each source against its own native intersection with
    ``design`` (as a naive per-source inner join does) compares different evaluation
    sets, not just different methods -- a source with denser native coverage on its own
    drug set can look better or worse for that reason alone. Returns a dict shaped like
    ``sources``, each (delta, key) row-filtered (order-preserving) to the common support.
    """
    def _pairs(frame: pd.DataFrame) -> pd.MultiIndex:
        return pd.MultiIndex.from_frame(cast("pd.DataFrame", frame[["patient", "drug"]].astype(str)))

    design_pairs = _pairs(design)
    labeled: dict[str, pd.Index] = {}
    for name, (_, key) in sources.items():
        pairs = _pairs(key)
        labeled[name] = cast("pd.MultiIndex", pairs[pairs.isin(design_pairs)]).unique()
    common = labeled[next(iter(labeled))]
    for idx in list(labeled.values())[1:]:
        common = common.intersection(idx)
    if len(common) == 0:
        raise ValueError("no (patient, drug) pairs shared across every source and design")
    out: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, (delta, key) in sources.items():
        mask = _pairs(key).isin(common)
        out[name] = (delta.loc[mask].reset_index(drop=True), key.loc[mask].reset_index(drop=True))
    return out


def build_learned_deltas(
    train_base: pd.DataFrame,
    train_delta: pd.DataFrame,
    train_key: pd.DataFrame,
    target_base: pd.DataFrame,
    patients: list[str],
    *,
    reducer: str = "pca",
    k: int | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Organoid-specific delta predictor -- the generation analogue of the expression
    baselines, sitting between the additive floor and Stack.

    Learns a baseline -> delta map on real L1000: reduce each training cell line's DMSO
    baseline by PCA/NMF, regress (CV-tuned ridge, ``make_penalty``) the treated-minus-DMSO
    delta *residual* (delta minus the per-drug mean) on those components, then predict each
    Soragni organoid's correction from its own baseline. The prediction is
    ``delta(organoid, drug) = drug_mean[drug] + correction(organoid)`` -- organoid-
    specific (so it can express within-drug interaction) but driven by a simple linear
    map. The correction transfers across the L1000<->Soragni platform gap in standardized
    units (PCA z-scores per cohort; NMF clips to non-negative). Returns ``(delta[pairs x
    genes], key[patient, drug])`` in the same shape as the other delta sources.

    ``k`` (n_components) defaults to ``None``, CV-selecting it from ``_K_GRID`` (inner 3-fold
    CV on the training fold only, never touching ``target_base``/``patients``) instead of a
    hardcoded fixed value -- the same principle ``make_penalty`` already applies to the ridge
    penalty, extended to the reducer's own capacity so PCA/NMF aren't compared against a
    foundation-model embedding (or each other) at an arbitrary, undertuned component count.
    Pass an explicit int to keep the old fixed-k behavior.
    """
    if reducer not in ("pca", "nmf"):
        raise ValueError("reducer must be 'pca' or 'nmf'")
    g = sorted(
        {str(c) for c in train_base.columns}
        & {str(c) for c in train_delta.columns}
        & {str(c) for c in target_base.columns}
    )
    if not g:
        raise ValueError("no shared genes among train_base, train_delta, target_base")

    drug_mean = train_delta[g].groupby(train_key["drug"].to_numpy()).mean()  # drug x gene
    resid = train_delta[g].to_numpy(dtype=np.float64) - drug_mean.loc[train_key["drug"]].to_numpy(
        dtype=np.float64
    )

    cells = train_base[g]
    cap = max(1, min(len(cells) - 1, len(g)))
    candidates: tuple[int, ...]
    if k is not None:
        candidates = (min(max(k, 1), cap),)
    else:
        candidates = tuple(sorted({c for c in _K_GRID if c <= cap} | {cap}))
    tgt = np.nan_to_num(target_base.reindex(columns=g).to_numpy(dtype=np.float64))
    cells_arr = cells.to_numpy(dtype=np.float64)
    sc = StandardScaler().fit(cells_arr) if reducer != "nmf" else None
    tr_pats = [str(p) for p in train_key["patient"]]
    ok = pd.Index(tr_pats).isin({str(c) for c in cells.index})

    def _reduce(k_try: int) -> tuple[np.ndarray, np.ndarray]:
        if reducer == "nmf":
            # sklearn-stubs mis-types n_components as str; the API takes an int.
            red = NMF(n_components=k_try, init="nndsvda", random_state=seed, max_iter=2000)  # type: ignore[arg-type]
            zc = red.fit_transform(np.maximum(cells_arr, 0.0))
            zo = red.transform(np.maximum(tgt, 0.0))
        else:
            assert sc is not None
            pca = PCA(n_components=k_try, random_state=seed)
            zc = pca.fit_transform(sc.transform(cells_arr))
            zo = pca.transform(sc.transform(tgt))
        return zc, zo

    def _pair_feat(zc: np.ndarray) -> np.ndarray:
        zdf = pd.DataFrame(zc, index=pd.Index([str(c) for c in cells.index]))
        return zdf.reindex(tr_pats).to_numpy()

    if len(candidates) > 1:
        k_eff = _cv_select_k(
            lambda k_try: _pair_feat(_reduce(k_try)[0])[ok], resid[ok], candidates, seed
        )
    else:
        k_eff = next(iter(candidates))

    z_cell, z_org = _reduce(k_eff)
    pair_feat = _pair_feat(z_cell)
    model = make_penalty("l2").fit(pair_feat[ok], resid[ok])  # type: ignore[attr-defined]

    z_org_df = pd.DataFrame(z_org, index=pd.Index([str(o) for o in target_base.index]))
    pats = [str(p) for p in patients]
    z_use = z_org_df.reindex(pats)
    keep = np.atleast_1d(~z_use.isna().to_numpy().any(axis=1))
    pats_keep = [p for p, kp in zip(pats, keep, strict=True) if kp]
    if not pats_keep:
        raise ValueError("no target organoids have a usable baseline")
    correction = model.predict(z_use[keep].to_numpy())  # type: ignore[attr-defined]

    drugs = np.asarray(drug_mean.index, dtype=object)
    n_p = len(pats_keep)
    dm = drug_mean.to_numpy(dtype=np.float64)  # (drug, gene)
    # delta(drug i, organoid j) = drug_mean[i] + correction[j]; rows are drug-major.
    delta_mat = np.repeat(dm, n_p, axis=0) + np.tile(correction, (len(drugs), 1))
    delta = pd.DataFrame(delta_mat, columns=pd.Index(g))
    key = pd.DataFrame(
        {
            "patient": np.tile(np.asarray(pats_keep, dtype=object), len(drugs)),
            "drug": np.repeat(drugs, n_p),
        }
    )
    return delta, key


def build_knn_deltas(
    train_base: pd.DataFrame,
    train_delta: pd.DataFrame,
    train_key: pd.DataFrame,
    target_base: pd.DataFrame,
    patients: list[str],
    *,
    k: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """k-NN delta predictor -- the cell-specific baseline matched to Stack's information.

    For each (target sample, drug) the predicted delta is the mean *real* delta of the
    ``k`` training cell lines whose baseline expression is closest to the target's
    baseline, among the lines treated with that drug. This is the transparent analogue of
    Stack's in-context generation: both see the query baseline and the drug's treated
    examples, but k-NN simply averages the nearest examples' responses instead of
    generating one. It sits between the additive floor (which ignores the baseline) and
    Stack: the query baseline selects *which* responses to average, so -- unlike the
    drug-agnostic linear map -- it can express cell x drug interaction.

    Baselines are compared on standardized, L2-normalized shared-gene profiles (cosine
    similarity, scale-free). Returns ``(delta[pairs x genes], key[patient, drug])`` in the
    same shape as the other delta sources.

    ``k`` (neighbor count) defaults to ``None``, CV-selecting it from ``_K_GRID`` via a
    leave-one-training-line-out prediction error (never touching ``target_base``/
    ``patients``) -- the same "don't hardcode capacity" principle
    ``build_learned_deltas``/``make_penalty`` apply to n_components/penalty strength,
    extended here to kNN's own neighbor count. Pass an explicit int to keep the old
    fixed-k behavior.
    """
    g = sorted(
        {str(c) for c in train_base.columns}
        & {str(c) for c in train_delta.columns}
        & {str(c) for c in target_base.columns}
    )
    if not g:
        raise ValueError("no shared genes among train_base, train_delta, target_base")

    sc = StandardScaler().fit(train_base[g].to_numpy(dtype=np.float64))

    def _emb(frame: pd.DataFrame) -> np.ndarray:
        z = sc.transform(np.nan_to_num(frame.reindex(columns=g).to_numpy(dtype=np.float64)))
        norm = np.linalg.norm(z, axis=1, keepdims=True)
        norm[norm == 0.0] = 1.0
        return z / norm

    pats = [str(p) for p in patients]
    have = [p for p in pats if p in {str(i) for i in target_base.index}]
    if not have:
        raise ValueError("no target samples have a usable baseline")
    q_emb = _emb(target_base.loc[have])  # (n_have x dim), unit vectors

    line_ids = [str(i) for i in train_base.index]
    line_emb = _emb(train_base)  # (n_lines x dim)
    line_pos = {lid: i for i, lid in enumerate(line_ids)}

    tk_drug = train_key["drug"].astype(str).to_numpy()
    row_line = pd.Series(train_key["patient"].astype(str).to_numpy()).map(line_pos).to_numpy()
    td = train_delta[g].to_numpy(dtype=np.float64)

    def _loo_error(k_try: int) -> float:
        # leave-one-training-line-out: predict each training line's OWN delta from its
        # (excluding itself) nearest same-drug training neighbors, at this k -- training-only,
        # never the held-out target this call ultimately predicts.
        errs: list[np.ndarray] = []
        for d in sorted(set(tk_drug)):
            rows_d = np.flatnonzero(tk_drug == d)
            li_d = row_line[rows_d]
            keep_d = ~pd.isna(li_d)
            rows_d, li_d = rows_d[keep_d], li_d[keep_d].astype(int)
            if rows_d.size < 2:
                continue
            sims_d = line_emb[li_d] @ line_emb[li_d].T
            np.fill_diagonal(sims_d, -np.inf)  # exclude self as its own neighbor
            kk_d = min(k_try, rows_d.size - 1)
            if kk_d < 1:
                continue
            nn_d = np.argpartition(-sims_d, kk_d - 1, axis=1)[:, :kk_d]
            pred_d = td[rows_d][nn_d].mean(axis=1)
            errs.append(((pred_d - td[rows_d]) ** 2).mean(axis=1))
        return float(np.concatenate(errs).mean()) if errs else float("inf")

    if k is None:
        n_lines = len(line_ids)
        knn_candidates: tuple[int, ...] = tuple(sorted({c for c in _K_GRID if c < n_lines} | {1}))
        if len(knn_candidates) > 1:
            k_eff = min(knn_candidates, key=_loo_error)
        else:
            k_eff = next(iter(knn_candidates))
    else:
        k_eff = max(1, k)

    # one pass per drug (drugs are few); the query x line neighbor search is vectorized.
    delta_blocks: list[np.ndarray] = []
    keys: list[tuple[str, str]] = []
    for d in sorted(set(tk_drug)):
        rows = np.flatnonzero(tk_drug == d)
        li = row_line[rows]
        keep = ~pd.isna(li)
        rows, li = rows[keep], li[keep].astype(int)
        if rows.size == 0:
            continue
        sims = q_emb @ line_emb[li].T  # (n_have x n_d) cosine, unit vectors
        kk = min(k_eff, rows.size)
        nn = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]  # k nearest lines per target
        delta_blocks.append(td[rows][nn].mean(axis=1))  # (n_have x genes)
        keys.extend((p, d) for p in have)
    if not delta_blocks:
        raise ValueError("no drug had a training line with a usable baseline")
    delta = pd.DataFrame(np.vstack(delta_blocks), columns=pd.Index(g))
    key = pd.DataFrame(keys, columns=pd.Index(["patient", "drug"]))
    return delta, key


def loo_baseline_source(
    kind: str,
    real_delta: pd.DataFrame,
    real_key: pd.DataFrame,
    base: pd.DataFrame,
    *,
    k: int | None,
    genes: pd.Index | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-cell-line-out baseline deltas: for each line, rebuild the source from the
    OTHER lines and predict the held-out line, so it never sees its own treated cells.

    ``additive``/``knn`` use all genes; ``pca``/``nmf`` (``build_learned_deltas``) reduce on
    the ``genes`` HVG panel, which keeps the per-line PCA/NMF fast and well-conditioned (49
    lines vs ~50k genes is hopelessly p>>n; on ~2k informative genes it is sane).

    ``k`` is ``build_knn_deltas``'/``build_learned_deltas``'s neighbor-count/n_components --
    ``None`` CV-selects it per held-out-line fold (training-fold-only, see those functions'
    docstrings); an explicit int keeps it fixed.
    """
    pats = real_key["patient"].astype(str).to_numpy()
    rdl = (
        real_delta
        if genes is None
        else cast("pd.DataFrame", real_delta[[str(g) for g in genes if g in real_delta.columns]])
    )
    bl = (
        base
        if genes is None
        else cast("pd.DataFrame", base[[str(g) for g in rdl.columns if g in base.columns]])
    )
    d_blocks: list[pd.DataFrame] = []
    k_blocks: list[pd.DataFrame] = []
    for line in [str(i) for i in base.index]:
        tr = pats != line
        if not tr.any():
            continue
        rd = real_delta.loc[tr].reset_index(drop=True)
        rk = real_key.loc[tr].reset_index(drop=True)
        if kind == "additive":
            d, kk = build_additive_deltas(rd, rk, [line])
        elif kind == "knn":
            d, kk = build_knn_deltas(base.drop(index=line), rd, rk, base.loc[[line]], [line], k=k)
        elif kind in ("pca", "nmf"):
            d, kk = build_learned_deltas(
                bl.drop(index=line),
                rdl.loc[tr].reset_index(drop=True),
                rk,
                bl.loc[[line]],
                [line],
                reducer=kind,
                k=k,
            )
        else:
            raise ValueError(f"unknown baseline source {kind!r}")
        d_blocks.append(d)
        k_blocks.append(kk)
    if not d_blocks:
        raise ValueError(f"no held-out lines produced a {kind} delta")
    return pd.concat(d_blocks, ignore_index=True), pd.concat(k_blocks, ignore_index=True)


def learned_gene_panel(
    real_delta: pd.DataFrame, hallmark_path: Path, *, n_hvg: int = 2000
) -> pd.Index:
    """HVG-union-Hallmark gene panel for the ``pca``/``nmf`` delta sources.

    The top ``n_hvg`` most-variable genes of the real delta, unioned with every gene named
    in any Hallmark signature -- so the learned reducers see both the highest-signal genes
    and the genes the fixed-signature readouts score on, keeping the two checks comparable.
    """
    hallmark = load_hallmark(hallmark_path)
    sig_genes = pd.Index(sorted({g for genes, _ in hallmark.values() for g in genes}))
    # DataFrame.var(axis=0) directly has an ambiguous pandas-stubs overload; go via numpy
    # (ddof=1 matches pandas' default sample variance) so the result is unambiguously a Series.
    # np.nanvar (not ndarray.var) so a NaN in one row doesn't propagate and zero out a gene's
    # variance -- matches pandas' default skipna=True, which is what real_delta.var(axis=0)
    # (the original inline code, and the still-standalone hvg line in
    # scripts/score_generation_eval.py) does.
    var = np.nanvar(real_delta.to_numpy(dtype=np.float64), axis=0, ddof=1)
    var_s = pd.Series(var, index=real_delta.columns)
    hvg = pd.Index(var_s.sort_values(ascending=False).index[:n_hvg])
    return hvg.union(sig_genes)


def common_gene_panel(
    real_delta: pd.DataFrame, generated: dict[str, pd.DataFrame] | None = None
) -> pd.Index:
    """The largest gene set EVERY source can supply, so all are scored on the same genes.

    Nothing controlled this before. `additive`/`knn`/`measured_delta` are derived from
    ``real_delta`` and carried all 53,393 genes; the Stack variants carried the 14,725 their
    generation gene list produced; and `pca`/`nmf` carried 2,647 because
    ``learned_gene_panel`` restricted them at construction time. Three gene spaces, and
    ``de_fidelity`` drops whatever a source lacks -- so each was scored on its own universe.
    That is the gene-axis twin of the (patient, drug) support bug ``restrict_common_support``
    exists to fix, and it hits ``pr_auc`` hardest: average precision depends on the positive
    rate, and a 5%-of-transcriptome panel chosen as high-variance-union-Hallmark is enriched
    for genes that actually move.

    The ceiling is set by the only genuinely constrained sources, the generated deltas: a
    generator cannot emit a gene its gene list never had. Everything else is a subset of
    ``real_delta`` and can be restricted to whatever this returns. Measured 2026-08-25:
    real_delta 53,393, both Stack variants 14,725 (identical sets), intersection 14,588.

    ``pca``/``nmf`` must be REBUILT on this panel, not filtered onto it -- only 925 of their
    2,647 genes fall inside it, so filtering alone would leave them narrower still.
    """
    panel = pd.Index(real_delta.columns)
    for name, gen in (generated or {}).items():
        before = len(panel)
        panel = panel.intersection(pd.Index(gen.columns))
        print(f"  common panel: {before} -> {len(panel)} after intersecting {name}")
    return panel


def assert_common_genes(sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> None:
    """Fail loudly if any source carries a different gene set from the others.

    A guard rather than a fix: the panel is applied at construction, and this catches a source
    that slipped past it. Scoring sources on different genes produces a table that looks
    well-formed and compares different things.
    """
    sizes = {name: d.shape[1] for name, (d, _) in sources.items()}
    if len(set(sizes.values())) > 1:
        raise ValueError(
            "sources do not share a gene panel, so their metrics are not comparable: "
            f"{sizes}. Build them with genes=common_gene_panel(...)."
        )



def build_l1000_gdsc_pairs(
    repo: Path,
    l1000_dir: Path,
    gctx: str,
    *,
    time: float = 24.0,
    chunk: int = 2000,
    treated_cap: int = 8,
    dmso_cap: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Real L1000 treated-minus-DMSO deltas paired with GDSC2 AUC on shared
    (cell line, drug) pairs.

    Returns ``(delta[pairs x gene symbols], key[patient, drug], gdsc_design[patient,
    drug, y], baseline[cell line x gene symbols])`` -- the last is the per-cell-line DMSO
    baseline for the learned delta predictors. Caps replicates and reads the ``.gctx`` in
    column chunks so memory is bounded; ``cmapPy`` is imported here (Alpine only)."""
    from cmapPy.pandasGEXpress.parse_gctx import parse  # type: ignore  # Alpine-only dep

    pert = pd.read_csv(
        l1000_dir / "GSE92742_Broad_LINCS_pert_info.txt.gz", sep="\t", low_memory=False
    )
    inst = pd.read_csv(
        l1000_dir / "GSE92742_Broad_LINCS_inst_info.txt.gz", sep="\t", low_memory=False
    )
    gene = pd.read_csv(l1000_dir / "GSE92742_Broad_LINCS_gene_info.txt.gz", sep="\t")
    gb = load_tranche("gdscv2", repo)
    xg, dg = build_sample_design(gb, "all", "auc", drug_key="pubchem_cid")
    gdr = pd.read_csv(repo / "data/raw/coderdata/gdscv2_drugs.tsv.gz", sep="\t")
    _, pert2drug = drug_pert_maps(gdr, pert)

    # GDSC2 samples are keyed by DepMap ModelID (ACH-...); L1000 wells are keyed by
    # cell-line name (e.g. "A375"). Map each ACH id to its stripped cell-line name so
    # the cohorts join on the shared name namespace -- xg.index is ACH ids, so a
    # direct name match would be empty.
    ach2name = {
        str(p.patient_id): str(p.metadata.get("stripped_cell_line_name") or "") for p in gb.patients
    }
    gcell = {_norm(ach2name[str(c)]): str(c) for c in xg.index if ach2name.get(str(c))}
    lcell = {_norm(c): str(c) for c in inst["cell_id"].unique()}
    shared = set(gcell) & set(lcell)
    l_ids = [lcell[k] for k in shared]
    l_to_g = {lcell[k]: gcell[k] for k in shared}
    print(f"shared: {len(pert2drug)} drugs, {len(shared)} cell lines")

    drug_ids = list(pert2drug)
    t = inst[
        inst["pert_id"].isin(drug_ids) & inst["cell_id"].isin(l_ids) & (inst["pert_time"] == time)
    ].copy()
    c = inst[
        (inst["pert_iname"] == "DMSO") & inst["cell_id"].isin(l_ids) & (inst["pert_time"] == time)
    ].copy()
    print(
        f"wells: {len(t)} treated + {len(c)} DMSO; "
        f"capping to <= {treated_cap}/(cell,drug), <= {dmso_cap}/cell"
    )
    t = (
        t.sort_values(by="inst_id")  # type: ignore[call-overload]
        .groupby(  # type: ignore[call-overload]
            ["cell_id", "pert_id"], sort=False
        )
        .head(treated_cap)
    )
    c = (
        c.sort_values(by="inst_id")  # type: ignore[call-overload]
        .groupby(  # type: ignore[call-overload]
            "cell_id", sort=False
        )
        .head(dmso_cap)
    )
    print(f"  after cap: {len(t)} treated + {len(c)} DMSO; reading in chunks of {chunk} ...")

    sym = gene.set_index("pr_gene_id")["pr_gene_symbol"].astype(str)
    t_lab = dict(
        zip(t["inst_id"], t["cell_id"].astype(str) + "\t" + t["pert_id"].astype(str), strict=True)
    )
    c_lab = dict(zip(c["inst_id"], c["cell_id"].astype(str), strict=True))

    def group_means(ids: list[str], lab: dict[str, str]) -> pd.DataFrame:
        tot: pd.DataFrame | None = None
        cnt: pd.Series | None = None
        for i in range(0, len(ids), chunk):
            block = parse(gctx, cid=ids[i : i + chunk]).data_df.T  # wells x genes
            block.index = block.index.map(lab)
            s, n = block.groupby(level=0).sum(), block.groupby(level=0).size()
            tot = s if tot is None else tot.add(s, fill_value=0.0)
            cnt = n if cnt is None else cnt.add(n, fill_value=0)
        assert tot is not None and cnt is not None
        return tot.div(cnt, axis=0)

    tmean = group_means(t["inst_id"].tolist(), t_lab)
    dmean = group_means(c["inst_id"].tolist(), c_lab)
    parts = pd.Series(tmean.index).str.split("\t", expand=True)
    cells, perts = parts[0].to_numpy(), parts[1].to_numpy()
    keep = pd.Series(cells).isin(dmean.index).to_numpy()
    tmean, cells, perts = tmean[keep], cells[keep], perts[keep]
    delta = pd.DataFrame(
        tmean.to_numpy() - dmean.reindex(index=cells, columns=tmean.columns).to_numpy(),
        columns=pd.Index([str(sym.get(int(i), "")) for i in tmean.columns]),
    )
    delta = delta.loc[:, [str(col) != "" for col in delta.columns]]
    delta = delta.loc[:, ~pd.Index(delta.columns).duplicated()]
    key = pd.DataFrame(
        {
            "patient": pd.Series(cells).map(l_to_g).to_numpy(),
            "drug": pd.Series(perts).map(pert2drug).to_numpy(),
        }
    )
    # per-cell-line DMSO baseline (gene symbols, GDSC2-name index) for the learned
    # delta predictors; same gene mapping / dedup as the delta.
    base = pd.DataFrame(
        dmean.to_numpy(),
        index=pd.Index([str(l_to_g.get(str(c), str(c))) for c in dmean.index]),
        columns=pd.Index([str(sym.get(int(i), "")) for i in dmean.columns]),
    )
    base = base.loc[:, [str(col) != "" for col in base.columns]]
    base = base.loc[:, ~pd.Index(base.columns).duplicated()]
    return delta, key, cast("pd.DataFrame", dg), base


def _group_mean(x: sparse.csr_matrix, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Per-group mean of the rows of ``x`` (cells x genes, CSR), vectorized via an indicator
    matmul -- no per-group loop and no densifying the full cell matrix.

    ``codes`` are integer group ids in ``[0, n_groups)`` aligned to the rows of ``x``.
    """
    n = cast("tuple[int, int]", x.shape)[0]
    g = sparse.csr_matrix(
        (np.ones(n, dtype=np.float64), (codes, np.arange(n))),
        shape=(n_groups, n),
    )
    sums = dense(g @ x)  # (n_groups x genes)
    counts = np.asarray(g.sum(axis=1)).ravel()
    counts[counts == 0] = 1.0
    return sums / counts[:, None]


def build_tahoe_deltas(
    adata: ad.AnnData,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Real pseudobulk treated-minus-DMSO deltas + per-line baseline from a Tahoe context.

    Aggregates the single cells of a ``build_tahoe_context`` AnnData to a pseudobulk profile
    per (cell line, drug) treated condition and per (cell line) DMSO control, then returns the
    log-fold-change ``delta = logcpm(treated) - logcpm(control)`` over the gene panel. This is
    the in-domain "truth" for generation quality (compare a generated delta to it) and the
    real-delta source the additive / k-NN baselines consume -- the Tahoe analogue of
    ``build_l1000_gdsc_pairs``, on the same ``logcpm`` log-fold-change scale so any source is
    comparable.

    The cell line is keyed by its DepMap id (obs ``cell_id``; falls back to ``cell_line_id``
    when empty), the drug by PubChem CID (obs ``pubchem_cid``) -- the canonical cross-dataset
    keys the viability join and the designs use. Returns ``(delta[pairs x genes], key[patient,
    drug], baseline[line x genes])``; ``baseline`` is the raw pseudobulk mean (counts), since
    the delta predictors expect a baseline expression profile, not a log fold-change.
    """
    obs = adata.obs
    genes = pd.Index([str(v) for v in adata.var_names])
    cid = obs["cell_id"].astype(str).to_numpy()
    cln = obs["cell_line_id"].astype(str).to_numpy()
    patient = np.where((cid != "") & (cid != "nan"), cid, cln)
    drug = obs["pubchem_cid"].astype(str).to_numpy()
    is_ctl = obs["is_control"].to_numpy(dtype=bool)
    x = adata.X
    xc = cast(
        "sparse.csr_matrix",
        x if sparse.issparse(x) else sparse.csr_matrix(np.asarray(x, dtype=np.float64)),
    )

    ctl = np.flatnonzero(is_ctl)
    trt = np.flatnonzero(~is_ctl)
    if ctl.size == 0:
        raise ValueError("no DMSO control cells (is_control) in the Tahoe context")
    if trt.size == 0:
        raise ValueError("no treated cells in the Tahoe context")

    # control pseudobulk per cell line (drug-agnostic).
    ccodes, cuniq = pd.factorize(patient[ctl])
    base = pd.DataFrame(
        _group_mean(xc[ctl], ccodes, len(cuniq)),
        index=pd.Index([str(u) for u in cuniq]),
        columns=genes,
    )

    # treated pseudobulk per (cell line, drug); a NUL-joined key keeps the pair atomic.
    tkey = pd.Series(patient[trt]).str.cat(pd.Series(drug[trt]), sep="\x1f").to_numpy()
    tcodes, tuniq = pd.factorize(tkey)
    tmean = _group_mean(xc[trt], tcodes, len(tuniq))
    parts = pd.Series(tuniq).str.split("\x1f", expand=True)
    tpat, tdrug = parts[0].to_numpy(), parts[1].to_numpy()

    # log fold-change vs each line's own DMSO baseline; drop pairs with no baseline.
    base_lc = logcpm(base)
    trt_lc = logcpm(pd.DataFrame(tmean, index=pd.Index(tpat), columns=genes))
    keep = np.asarray(pd.Index(tpat).isin(base_lc.index))
    if not keep.any():
        raise ValueError("no treated (line, drug) pair has a matching DMSO baseline")
    delta = pd.DataFrame(
        trt_lc.to_numpy()[keep] - base_lc.reindex(tpat[keep]).to_numpy(),
        columns=genes,
    )
    key = pd.DataFrame({"patient": tpat[keep], "drug": tdrug[keep]})
    return delta, key, base


def build_tahoe_de_calls(
    adata: ad.AnnData,
    *,
    lfc_threshold: float = 0.25,
    fdr_threshold: float = 0.05,
) -> pd.DataFrame:
    """Ground-truth per-(line, drug, gene) DE calls: two-sided Wilcoxon rank-sum test (treated
    vs. that line's own DMSO control cells), Benjamini-Hochberg FDR correction, significant =
    ``(padj < fdr_threshold) & (|log2fc| > lfc_threshold)`` -- the paper's own cell-eval-based DE
    procedure (Methods 4.6.3: "Wilcoxon rank-sum tests for DE detection and Benjamini-Hochberg
    correction... cell-eval v0.6.6 with default parameters"; the default thresholds here come
    from Methods 4.8's "LFC threshold of 0.25 and a FDR threshold of 0.05 for cell-eval
    evaluations" -- the only concrete LFC/FDR pair stated anywhere in the paper for a cell-eval
    DE call). Uses ``scanpy.tl.rank_genes_groups(method="wilcoxon")``, which already implements
    this exact test + BH correction, rather than a hand-rolled scipy loop.

    Loops once per cell line (not per (line, drug) pair): for each line, every drug applied in
    that line is compared against the line's own control cells in a single ``rank_genes_groups``
    call (scanpy natively supports multiple groups vs. one reference), so the gene-level
    computation for every drug in that line is vectorized together.

    Returns one row per (line, drug, gene): ``patient, drug, gene, log2fc, padj, significant`` --
    the ground-truth side of Check 1's DE metrics (``fmharness.evaluation.de_fidelity``); the
    predicted side needs no test (ranked by ``|log2fc|`` alone -- see that function's docstring).
    """
    obs = adata.obs
    genes = pd.Index([str(v) for v in adata.var_names])
    cid = obs["cell_id"].astype(str).to_numpy()
    cln = obs["cell_line_id"].astype(str).to_numpy()
    patient = np.where((cid != "") & (cid != "nan"), cid, cln)
    drug = obs["pubchem_cid"].astype(str).to_numpy()
    is_ctl = obs["is_control"].to_numpy(dtype=bool)

    x = adata.X
    xc = cast(
        "sparse.csr_matrix",
        x if sparse.issparse(x) else sparse.csr_matrix(np.asarray(x, dtype=np.float64)),
    )
    lib = np.asarray(xc.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0
    log1p_cpm = xc.multiply(1e4 / lib[:, None]).tocsr()
    log1p_cpm.data = np.log1p(log1p_cpm.data)

    rows: list[pd.DataFrame] = []
    for line in sorted(set(patient[is_ctl])):
        line_mask = patient == line
        ctl_mask = line_mask & is_ctl
        trt_mask = line_mask & ~is_ctl
        if not ctl_mask.any() or not trt_mask.any():
            continue
        idx = np.flatnonzero(ctl_mask | trt_mask)
        group = np.where(is_ctl[idx], "control", drug[idx])
        drugs_here = [str(d) for d in pd.unique(group) if str(d) != "control"]
        if not drugs_here:
            continue
        sub = ad.AnnData(
            X=log1p_cpm[idx],
            obs=pd.DataFrame({"de_group": group}, index=pd.Index([str(i) for i in idx])),
            var=pd.DataFrame(index=genes),
        )
        sc.tl.rank_genes_groups(
            sub, groupby="de_group", groups=drugs_here, reference="control", method="wilcoxon"
        )
        # group=None returns every tested group's rows concatenated in one DataFrame (with its
        # own "group" column) -- no per-drug loop needed alongside the per-line loop above. When
        # exactly one drug is tested against control, rank_genes_groups_df drops the "group"
        # column entirely ("backward compat" for the single-group case) -- restore it explicitly
        # rather than indexing a column that may not exist.
        res = sc.get.rank_genes_groups_df(sub, group=None)
        if "group" not in res.columns:
            res = res.assign(group=drugs_here[0])
        rows.append(
            pd.DataFrame(
                {
                    "patient": line,
                    "drug": res["group"].to_numpy(),
                    "gene": res["names"].to_numpy(),
                    "log2fc": res["logfoldchanges"].to_numpy(dtype=float),
                    "padj": res["pvals_adj"].to_numpy(dtype=float),
                }
            )
        )
    if not rows:
        raise ValueError("no (line, drug) pair had both control and treated cells")
    out = pd.concat(rows, ignore_index=True)
    out["significant"] = (out["padj"] < fdr_threshold) & (out["log2fc"].abs() > lfc_threshold)
    return out


def pseudobulk_de_to_deltas(
    de: pd.DataFrame,
    name_to_cid: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Real per-(line, drug) deltas + per-line baseline from Tahoe's pseudobulk DESeq2 table.

    The streaming-free shortcut to ``build_tahoe_deltas``: Tahoe ships a
    ``pseudobulk_differential_expression`` table (per cell line x drug x dose x plate, per gene)
    carrying ``log2FoldChange`` (treated vs DMSO) and ``baseMean``. This aggregates it to the same
    ``(delta, key, baseline)`` contract -- delta = mean ``log2FoldChange`` per (DepMap line, drug)
    pooled over dose and plate; baseline = mean ``baseMean`` per line (a proxy for the line's
    expression profile, used only to choose k-NN neighbors). The drug is re-keyed from Tahoe's
    name to its PubChem CID via ``name_to_cid`` (names without a CID are dropped). The log2 scale
    and the baseMean proxy are harmless downstream: delta_fidelity is correlation-based and the
    readouts z-score the delta.

    ``de`` needs columns ``gene_name, log2FoldChange, baseMean, Cell_ID_DepMap, drug``. Returns
    ``(delta[pairs x genes], key[patient, drug], baseline[line x genes])``.
    """
    d = de.loc[:, ["gene_name", "log2FoldChange", "baseMean", "Cell_ID_DepMap", "drug"]].copy()
    d["drug"] = d["drug"].astype(str).map(name_to_cid)
    d["patient"] = d["Cell_ID_DepMap"].astype(str)
    d = d[d["drug"].notna()]
    if d.empty:
        raise ValueError("no pseudobulk rows mapped to a target drug CID")

    # mean over dose/plate -> one delta per (line, drug); baseMean -> one baseline per line.
    delta_wide = d.pivot_table(
        index=["patient", "drug"], columns="gene_name", values="log2FoldChange", aggfunc="mean"
    ).fillna(0.0)
    base = d.pivot_table(
        index="patient", columns="gene_name", values="baseMean", aggfunc="mean"
    ).fillna(0.0)

    key = pd.DataFrame(
        {
            "patient": [str(p) for p in delta_wide.index.get_level_values(0)],
            "drug": [str(x) for x in delta_wide.index.get_level_values(1)],
        }
    )
    delta = delta_wide.reset_index(drop=True)
    delta.columns = pd.Index([str(c) for c in delta.columns])
    base.columns = pd.Index([str(c) for c in base.columns])
    base.index = pd.Index([str(i) for i in base.index])
    return delta, key, base
