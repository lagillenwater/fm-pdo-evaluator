"""Check-2 scoring building blocks: penalty models, per-drug representation splitting, and
the leave-cell-line-out penalized-regression fit, plus ``score_check2``, the composition of
these into the fixed-signature-readout scoring and the representation-controlled penalized
grid.

Shared by ``scripts/score_generation_eval.py`` and ``scripts/check2_registry_driver.py``.
Stays leakage-agnostic -- ``score_check2`` scores whatever ``design`` frame it is handed, the
same way ``evaluation.score_delta_sources`` does for Check 1. ``filter_leakage`` is always the
caller's job, not this module's.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse, spmatrix
from sklearn.preprocessing import StandardScaler

from fmharness.adapters import PENALTY_NAMES, build_adapters, make_penalty
from fmharness.controls import plant_interaction
from fmharness.deltas import restrict_common_support
from fmharness.evaluation import score_predictions
from fmharness.signatures import PROLIFERATION

FIXED_READOUTS = ("hallmark", "proliferation")  # fixed-signature readouts, applied to delta sources


def load_line_matrix(path: Path) -> pd.DataFrame:
    """Load a per-cell-line feature matrix (index = line id) for the check-2 grid, from a
    ``.h5ad`` (X + obs_names), ``.parquet``, or ``.csv``. Used to fold a precomputed FM
    embedding (one vector per line) in head-to-head with expr/pca."""
    if path.suffix == ".h5ad":
        a = ad.read_h5ad(path)
        # scipy.sparse type stubs don't expose .toarray() on spmatrix base class even though
        # it's guaranteed present at runtime when issparse(a.X) is True.
        x = cast(spmatrix, a.X).toarray() if issparse(a.X) else np.asarray(a.X)  # type: ignore[attr-defined]
        return pd.DataFrame(x, index=pd.Index([str(o) for o in a.obs_names])).astype(float)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, index_col=0)
    df.index = pd.Index([str(i) for i in df.index])
    return df.astype(float)


def repr_by_drug(
    delta: pd.DataFrame, key: pd.DataFrame, genes: pd.Index
) -> dict[str, pd.DataFrame]:
    """Split a delta source into ``{drug: DataFrame[line x genes]}`` for per-drug regression."""
    d = delta.reindex(columns=genes).fillna(0.0)
    pat = key["patient"].astype(str).to_numpy()
    drg = key["drug"].astype(str).to_numpy()
    out: dict[str, pd.DataFrame] = {}
    for drug in pd.unique(drg):
        mask = drg == drug
        m = d.loc[mask]
        m.index = pd.Index(pat[mask])
        out[str(drug)] = m
    return out


def penalized_preds(
    feat: dict[str, pd.DataFrame] | Callable[[str], pd.DataFrame],
    design: pd.DataFrame,
    fold_of: dict[str, int],
    n_folds: int,
    uniq_lines: list[str],
    penalty: str,
    *,
    min_lines: int = 8,
    min_train: int = 5,
) -> pd.DataFrame:
    """Per-drug penalized regression (representation -> AUC), leave-cell-line-out by fold.

    ``feat`` maps a drug to a (line x gene) frame -- a dict for a delta source, or a callable for a
    drug-independent representation (baseline expression). For each drug the model is fit on the
    training-fold lines' features vs AUC and predicts the held-fold lines; the StandardScaler is fit
    on the training lines only, so a single held line (true LOO) is scored leakage-free. All
    representations share one model class, so a difference is the representation, not the model.
    Returns preds (patient, drug, y_true, y_pred); y_pred is an AUC estimate (same sign as y_true).
    """
    auc_by_drug = {
        str(d): dict(zip(g["patient"].astype(str), g["y"], strict=False))
        for d, g in design.groupby("drug")
    }
    rows: list[tuple[str, str, float, float]] = []
    for drug, auc in auc_by_drug.items():
        fdf = feat(drug) if callable(feat) else feat.get(drug)  # type: ignore[union-attr]
        if fdf is None or fdf.empty:
            continue
        fdf = fdf.copy()
        fdf.index = pd.Index([str(i) for i in fdf.index])
        lines_d = [ln for ln in fdf.index if ln in auc]
        if len(lines_d) < min_lines:
            continue
        for f in range(n_folds):
            held = {ln for ln in uniq_lines if fold_of[ln] == f}
            tr = [ln for ln in lines_d if ln not in held]
            te = [ln for ln in lines_d if ln in held]
            if len(tr) < min_train or not te:
                continue
            sc = StandardScaler().fit(fdf.loc[tr].to_numpy(dtype=np.float64))
            # make_penalty returns `object` (RidgeCV/LassoCV/ElasticNetCV share no common
            # base class pyright can see .fit/.predict on) -- both calls need an ignore.
            model = make_penalty(penalty).fit(  # type: ignore[attr-defined]
                sc.transform(fdf.loc[tr].to_numpy(dtype=np.float64)), [auc[ln] for ln in tr]
            )
            te_x = sc.transform(fdf.loc[te].to_numpy(dtype=np.float64))
            pred = model.predict(te_x)  # type: ignore[attr-defined]
            rows.extend(
                (ln, drug, float(auc[ln]), float(p)) for ln, p in zip(te, pred, strict=False)
            )
    cols = pd.Index(["patient", "drug", "y_true", "y_pred"])
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def restrict_representation_support(
    representations: dict[str, dict[str, pd.DataFrame] | Callable[[str], pd.DataFrame]],
    design: pd.DataFrame,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Restrict every representation to the SAME per-drug line coverage before scoring.

    ``restrict_common_support`` fixes this for delta ``(delta, key)`` row-pairs; the
    representation-controlled penalized grid needs the same guarantee for a different
    shape -- a drug-independent representation (``"expr"``, any ``stack_emb`` entry) sees
    the SAME full line index for every drug, while a delta-source representation
    (``repr_by_drug``) only has a drug entry, and only the lines under it, that source's
    own construction actually produced. Scoring both in the same table without
    restricting first compares different (patient, drug) support per representation, not
    just different representations -- the same bug class ``restrict_common_support``
    fixes for delta sources, generalized to this dict-of-per-drug-frames shape.

    Materializes every representation to ``{drug: DataFrame[line x features]}`` over the
    drugs in ``design``, drops a drug entirely if any representation lacks it, and
    restricts every surviving drug's frame to the patients labeled for it in ``design``
    AND present in every representation's own line index for it.
    """
    drugs = sorted({str(d) for d in design["drug"]})
    materialized: dict[str, dict[str, pd.DataFrame]] = {
        name: ({d: rep(d) for d in drugs} if callable(rep) else dict(rep))
        for name, rep in representations.items()
    }
    out: dict[str, dict[str, pd.DataFrame]] = {name: {} for name in materialized}
    for drug in drugs:
        labeled = {
            str(p) for p in design.loc[design["drug"].astype(str) == drug, "patient"]
        }
        frames = {name: reps.get(drug) for name, reps in materialized.items()}
        if any(f is None or f.empty for f in frames.values()):
            continue
        common = labeled
        for f in frames.values():
            assert f is not None  # narrowed by the any(...) check above
            common = common & {str(i) for i in f.index}
        if not common:
            continue
        common_sorted = sorted(common)
        for name, f in frames.items():
            assert f is not None
            out[name][drug] = f.loc[common_sorted]
    return out


def random_feature_control(shape: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
    """A same-shape (index, n_features), i.i.d. standard-normal negative control for a
    high-dimensional FM-embedding representation (Stack now, scFoundation later).

    Fit through the identical CV-tuned model as the real embedding, this tests whether an
    apparent win reflects learned structure or just raw parameter count -- a foundation
    model's embedding width comes from its pretraining corpus, not the small eval cohort,
    so it can't be fairly capacity-matched against PCA/NMF/kNN by truncating dimensions
    (see docs/superpowers/specs/2026-08-21-cross-check-fairness-and-capacity-design.md).
    Instead the real embedding must clear this same-width random control by a real margin
    to attribute any win to structure rather than dimension count -- generalizes
    ``fmharness.signatures.score_signatures``'s same-size random-gene-set control
    (``rnd_p95``/``p_vs_random``) from gene-set size to embedding width.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.standard_normal((len(shape), shape.shape[1])),
        index=shape.index,
        columns=pd.Index([f"rand{i}" for i in range(shape.shape[1])]),
    )


def random_control_representation(
    rep: dict[str, pd.DataFrame] | Callable[[str], pd.DataFrame],
    drugs: list[str],
    *,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """A same-shape, per-drug random-feature negative control for one representation of
    the penalized grid -- one ``random_feature_control`` call per drug, so a drug-specific
    representation's control matches its OWN per-drug line coverage and feature width, not
    a single global shape. Every representation in the grid gets one (not just a
    foundation-model embedding): an apparent win from ANY representation should be checked
    against raw capacity alone, not assumed away because it isn't obviously high-dimensional.
    """
    out: dict[str, pd.DataFrame] = {}
    for d in drugs:
        frame = rep(d) if callable(rep) else rep.get(d)
        if frame is None or frame.empty:
            continue
        out[d] = random_feature_control(frame, seed=seed)
    return out


def score_check2(
    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    real_key: pd.DataFrame,
    base: pd.DataFrame,
    hvg: pd.Index,
    design: pd.DataFrame,
    *,
    hallmark: dict[str, tuple[tuple[str, ...], int]],
    fixed_methods: tuple[str, ...] = FIXED_READOUTS,
    penalties: tuple[str, ...] = PENALTY_NAMES,
    folds: int = 5,
    stack_emb: dict[str, pd.DataFrame] | None = None,
    oracle: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    n_permutations: int = 1000,
) -> pd.DataFrame:
    """Check-2 table: fixed-signature readouts + representation-controlled penalized grid.

    ``design`` is the (patient, drug, y) AUC label frame -- the caller's responsibility to
    leakage-filter first (this function does not know ``filter_leakage`` exists; it scores
    whatever ``design`` it is handed, exactly like ``evaluation.score_delta_sources`` does for
    Check 1). (a) scores every ``sources`` delta through each named fixed Hallmark-derived
    readout (sensitivity -> ``-score`` vs AUC); (b) fits the SAME penalized regression
    (RidgeCV/LassoCV/ElasticNetCV, one per ``penalties`` entry) to the untreated ``base``
    expression, every ``sources`` delta, and any ``stack_emb`` embedding, leave-cell-line-out
    by grouped fold, so a difference across representations is the representation and not the
    model (Kurilov 2020). Returns one row per (source, method) with
    global/interaction/perdrug/p_label/regret@1/regret@3/n.

    ``oracle`` (2026-08-22), if given, is a real ``(delta, key)`` pair -- e.g. the real
    measured Tahoe delta -- scored ONLY in part (b), not part (a): part (a)'s real-delta-vs-
    real-AUC validation is already the caller's own "Gate" print (real delta through Hallmark
    with a random-gene-set null, richer than a plain oracle row there would be), so including
    it in ``sources``/part (a) would just duplicate that at lower fidelity. Part (b) has no
    such existing check, so ``oracle`` gives the penalized grid's l1/l2/en axes a best-case
    ceiling the Gate doesn't cover.
    """
    fixed_sigs = {
        "hallmark": hallmark,
        "proliferation": {n: hallmark[n] for n in PROLIFERATION if n in hallmark},
    }
    fixed_readouts = {
        m: build_adapters(["hallmark"], signatures=fixed_sigs[m])[0]
        for m in fixed_methods
        if m in fixed_sigs
    }
    uniq_lines = sorted(set(real_key["patient"].astype(str)))
    n_folds = max(1, min(folds, len(uniq_lines)))
    fold_of = {ln: i % n_folds for i, ln in enumerate(uniq_lines)}
    target_drugs = list(set(real_key["drug"].astype(str)))
    # Boolean-mask indexing on a DataFrame is typed Series | DataFrame | Unknown by the pandas
    # stubs (they can't see this mask always selects rows of the same frame); narrow it back.
    design_target = cast(pd.DataFrame, design[design["drug"].astype(str).isin(target_drugs)])

    def _row(s: dict[str, float]) -> dict[str, object]:
        return {
            "global": s["global"],
            "interaction": s["interaction"],
            "perdrug": s["perdrug"],
            "p_label": s["p_label"],
            "regret@1": s["regret@1"],
            "regret@3": s["regret@3"],
            "n": int(s["n"]),
        }

    out: list[dict[str, object]] = []

    # (a) fixed-signature readouts on the delta sources (sensitivity -> -y_pred vs AUC).
    # restrict_common_support first: sources can have very different native (patient, drug)
    # coverage (a broadcast baseline vs a narrowly-generated one), so scoring each against its
    # own native intersection with `design` (a naive per-source inner join, as below) compares
    # different evaluation sets, not just different methods -- the Path-B bug (commit d9f94ec),
    # fixed here too. Local to part (a) only -- part (b) needs its own, wider per-drug fold
    # training data and gets restrict_representation_support instead (see below).
    fixed_sources = restrict_common_support(sources, design)
    for name, (d, kk) in fixed_sources.items():
        for method, adapter in fixed_readouts.items():
            sens = np.asarray(adapter.predict(d), dtype=float)
            merged = pd.DataFrame(
                {"patient": kk["patient"].to_numpy(), "drug": kk["drug"].to_numpy(), "_s": sens}
            ).merge(design.rename(columns={"y": "y_true"}), on=["patient", "drug"], how="inner")
            if merged.empty:
                continue
            preds = pd.DataFrame(
                {
                    "patient": merged["patient"],
                    "drug": merged["drug"],
                    "y_true": merged["y_true"].to_numpy(),
                    "y_pred": -merged["_s"].to_numpy(),
                }
            )
            s = score_predictions(preds, n_perm=n_permutations)
            out.append({"source": name, "method": method, **_row(s)})

    # (b) representation-controlled penalized regression: baseline expression + every delta source.
    base_hvg = base.reindex(columns=hvg).fillna(0.0)
    representations: dict[str, dict[str, pd.DataFrame] | Callable[[str], pd.DataFrame]] = {
        "expr": lambda _drug: base_hvg
    }
    for name, (d, kk) in sources.items():
        representations[name] = repr_by_drug(d, kk, hvg)
    for label, emb in (stack_emb or {}).items():
        representations[label] = (lambda e: lambda _drug: e)(emb)
    if oracle is not None:
        representations["oracle"] = repr_by_drug(oracle[0], oracle[1], hvg)
    # same-width random-feature negative control (random_control_representation), one per
    # representation -- not just stack_emb: an apparent win from ANY representation (expr,
    # pca, nmf, stack, ...) should be checked against raw capacity alone, so every row of
    # the grid has a matching "<name>_random" row scored through the identical pipeline.
    for name in list(representations):
        representations[f"{name}_random"] = random_control_representation(
            representations[name], target_drugs, seed=0
        )
    # restrict_representation_support: "expr"/stack_emb are drug-independent (always the full
    # line index for every drug) while a delta-source representation only has the lines its own
    # construction produced for that drug -- without restricting first, rows of the SAME table
    # are scored on different per-drug (patient, drug) support, not just different
    # representations.
    restricted_representations = restrict_representation_support(representations, design_target)
    for repr_name, feat in restricted_representations.items():
        for pen in penalties:
            preds = penalized_preds(feat, design_target, fold_of, n_folds, uniq_lines, pen)
            if preds.empty:
                continue
            s = score_predictions(preds, n_perm=n_permutations)
            out.append({"source": repr_name, "method": pen, **_row(s)})

    # positive control: plant a KNOWN organoid x drug interaction into the untreated baseline
    # expression space (fmharness.controls.plant_interaction) and confirm the SAME penalized
    # grid recovers it -- proves the fold structure + model + interaction_rho actually detect
    # a real interaction when one is guaranteed to exist, the flowchart's "positive control:
    # planted interaction, recovered" (distinct from the oracle/random-feature controls,
    # which use real data or noise, not a simulation with a known, controlled effect size).
    emb_per_row = base_hvg.reindex(design_target["patient"]).to_numpy()
    within_drug_sd = float(
        np.std(
            design_target["y"].to_numpy()
            - design_target.groupby("drug")["y"].transform("mean").to_numpy()
        )
    )
    plant_scale = within_drug_sd if within_drug_sd > 0 else 1.0
    planted_y = plant_interaction(
        cast("pd.Series", design_target["drug"]),
        cast("pd.Series", design_target["y"]),
        emb_per_row,
        effect=plant_scale,
        noise_sd=plant_scale,
        rng=np.random.default_rng(0),
    )
    planted_design = design_target.assign(y=planted_y)
    for pen in penalties:
        preds = penalized_preds(
            lambda _drug: base_hvg, planted_design, fold_of, n_folds, uniq_lines, pen
        )
        if preds.empty:
            continue
        s = score_predictions(preds, n_perm=n_permutations)
        out.append({"source": "planted", "method": pen, **_row(s)})

    return pd.DataFrame(out)
