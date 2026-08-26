"""Shared evaluation helpers.

Build a sample-level design from a CoderData bundle, run grouped K-fold with a
probe, and score the held-out predictions. Used by the evaluation scripts and
by the controls so they share one code path.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

from fmharness.cv import CVScheme, group_k_fold
from fmharness.data.loaders import CoderDataBundle

_MODEL_TYPE = {"organoid": "patient derived organoid", "tumor": "tumor"}


def cpm_bundle(bundle: CoderDataBundle) -> CoderDataBundle:
    """Return a copy of ``bundle`` with expression X as per-million (CPM).

    Prefers raw integer counts when present (GDSC2 keeps them in
    ``layers['raw_counts']``); otherwise X is already count-derived and
    length-free (Soragni CPM) and is renormalized to per-million. This puts
    GDSC2 and Soragni on one shared, length-free normalization -- required for a
    fair cross-substrate comparison, since the native loaders otherwise leave
    GDSC2 on DESeq2 median-of-ratios and Soragni on CPM.
    """
    expr = bundle.expression.copy()
    m = np.asarray(expr.layers.get("raw_counts", expr.X), dtype=np.float64)
    lib = m.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    expr.X = m / lib * 1e6
    return dataclasses.replace(bundle, expression=expr)


def build_sample_design(
    bundle: CoderDataBundle,
    rna_source: str = "all",
    metric: str = "auc",
    drug_key: str = "drug_id",
):
    """Return (sample x gene expression frame, design[patient, drug, y]).

    ``rna_source`` selects one substrate's RNA by model_type, or "all".
    Expression is averaged per patient over its samples of that substrate.
    The design has one row per (patient, drug) with the mean response of the
    chosen metric.

    ``drug_key`` chooses how drugs are identified in the ``drug`` column:
    ``"drug_id"`` (each dataset's native id -- fine within a dataset) or
    ``"pubchem_cid"`` (the canonical cross-dataset key -- required when joining
    GDSC2 to Soragni, whose native drug ids share no namespace). Assays missing
    the chosen key are dropped; multiple ids collapsing to one key are averaged.
    """
    improve_to_patient = {
        str(s.metadata.get("improve_sample_id")): s.patient_id for s in bundle.samples
    }
    improve_to_model = {
        str(s.metadata.get("improve_sample_id")): str(s.metadata.get("model_type"))
        for s in bundle.samples
    }
    sid_to_patient = {s.sample_id: s.patient_id for s in bundle.samples}

    expr = bundle.expression
    obs = [str(s) for s in expr.obs_names]
    if rna_source == "all":
        keep = list(range(len(obs)))
    else:
        target = _MODEL_TYPE[rna_source]
        keep = [i for i, s in enumerate(obs) if improve_to_model.get(s) == target]
    sub = expr[keep]
    x_df = (
        pd.DataFrame(
            np.asarray(sub.X, dtype=np.float64),
            index=pd.Index([improve_to_patient[str(s)] for s in sub.obs_names]),
            columns=pd.Index([str(v) for v in expr.var_names]),
        )
        .groupby(level=0)
        .mean()
    )

    raw_drug = [getattr(x, drug_key) for x in bundle.drug_assays]
    a = pd.DataFrame(
        {
            "patient": [sid_to_patient.get(x.sample_id, x.sample_id) for x in bundle.drug_assays],
            "drug": [None if d is None else str(d) for d in raw_drug],
            "metric": [x.response_metric for x in bundle.drug_assays],
            "y": [x.response_value for x in bundle.drug_assays],
        }
    )
    a = a[(a["metric"] == metric) & a["drug"].notna() & (a["patient"].isin(x_df.index.tolist()))]
    design = a.groupby(["patient", "drug"], as_index=False)["y"].mean()
    return x_df, design


def grouped_cv_predict(
    probe_factory: Callable[[], object],
    x_df: pd.DataFrame,
    design: pd.DataFrame,
    *,
    n_splits: int | None = None,
    cv: CVScheme | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Fit a fresh probe per fold, collect held-out (base, residual, prediction).
    ``probe_factory`` returns a probe exposing ``fit(emb, drugs, y, groups)``
    and ``predict_parts(emb, drugs)``.

    Exactly one of ``n_splits`` (grouped K-fold split by patient -- the
    harness's original CV shape) or ``cv`` (any ``CVScheme``, e.g. from
    ``fmharness.cv.resolve_cv`` or ``leave_subtype_out``) must be given.
    ``cv`` is the harness's registry-driven path -- it composes with
    ``Modality.recommended_cv()`` and with ``filter_leakage``'s
    row-count-changing output without a second capping step; ``n_splits``
    stays for existing callers that just want K folds by patient.

    ``seed`` has no effect on the ``n_splits``/``group_k_fold`` path --
    ``GroupKFold`` is a deterministic size-balancing assignment, not shuffled.
    It is accepted for callers passing a ``cv`` that carries its own
    randomness (e.g. ``leave_subtype_out(seed=...)``); seed that scheme
    directly rather than relying on this parameter to reach it.
    """
    if (n_splits is None) == (cv is None):
        raise ValueError("pass exactly one of n_splits or cv")
    scheme = group_k_fold(n_splits) if n_splits is not None else cv
    assert scheme is not None
    rows: list[dict[str, object]] = []
    for tr, te in scheme.splits(design):
        d_tr, d_te = design.iloc[tr], design.iloc[te]
        probe = probe_factory()
        probe.fit(  # type: ignore[attr-defined]
            x_df.loc[d_tr["patient"]],
            list(d_tr["drug"]),
            d_tr["y"].to_numpy(),
            groups=list(d_tr["patient"]),
        )
        base, resid = probe.predict_parts(  # type: ignore[attr-defined]
            x_df.loc[d_te["patient"]], list(d_te["drug"])
        )
        for (_, r), b, rs in zip(d_te.iterrows(), base, resid, strict=True):
            rows.append(
                {
                    "patient": r["patient"],
                    "drug": r["drug"],
                    "y_true": float(r["y"]),
                    "y_pred": float(b + rs),
                    "y_resid": float(rs),
                }
            )
    return pd.DataFrame.from_records(rows)


def _within_drug_corr(preds: pd.DataFrame, true_col: str, pred_col: str, min_n: int = 3) -> float:
    """Pooled within-drug rank correlation of ``true_col`` vs ``pred_col``.

    Ranks both columns inside each drug, centers the ranks, pools across drugs,
    then takes one Pearson correlation. Working within drug removes the drug
    (column) mean, so this is the per-sample signal beyond drug identity.
    """
    ct, cp = [], []
    for _, g in preds.groupby("drug"):
        if len(g) < min_n:
            continue
        rt = g[true_col].rank().to_numpy()
        rp = g[pred_col].rank().to_numpy()
        ct.append(rt - rt.mean())
        cp.append(rp - rp.mean())
    if not ct:
        return float("nan")
    cta, cpa = np.concatenate(ct), np.concatenate(cp)
    if np.std(cta) < 1e-12 or np.std(cpa) < 1e-12:
        return 0.0
    return float(np.asarray(pearsonr(cta, cpa))[0])


def within_drug_rho(preds: pd.DataFrame, pred_col: str = "y_resid", min_n: int = 3) -> float:
    """Within-drug rank correlation of observed vs prediction (``pred_col``).

    Removes the drug mean only. Includes the general-sensitivity effect: an
    organoid sensitive to most drugs ranks low-AUC inside every drug, so this
    rewards predicting that overall sensitivity as well as drug-specific signal.
    """
    return _within_drug_corr(preds, "y_true", pred_col, min_n)


def _two_way_demean(values: np.ndarray, patient: np.ndarray, drug: np.ndarray) -> np.ndarray:
    """Remove additive patient AND drug main effects (the two-way within transformation).

    On a BALANCED panel one pass of each is exact. On an incomplete panel it is not:
    subtracting patient means shifts the drug means and vice versa, so this alternates until
    the residual stops moving. What remains carries neither main effect, which is the
    definition of the interaction term.

    Implemented with ``np.bincount`` over factorised codes rather than pandas groupby. The
    permutation null calls this 1000 times per scored row, and the groupby version made each
    ``score_check2`` test take 45-63 seconds on its own.

    WHY iterate rather than one pass each: artifact
    ``tests/test_evaluation.py::test_interaction_rho_is_zero_for_a_predictor_with_no_patient_information``
    still fails with a single pass at the raggedness of the real Tahoe/GDSC2 design
    (44 lines x 30 drugs, 17 pairs missing, 1.3% incomplete).
    """
    v = np.asarray(values, dtype=np.float64).copy()
    pc, pn = _codes(patient)
    dc, dn = _codes(drug)
    pcount = np.bincount(pc, minlength=pn).astype(np.float64)
    dcount = np.bincount(dc, minlength=dn).astype(np.float64)
    for _ in range(100):
        prev = v.copy()
        v -= (np.bincount(dc, weights=v, minlength=dn) / dcount)[dc]
        v -= (np.bincount(pc, weights=v, minlength=pn) / pcount)[pc]
        if float(np.max(np.abs(v - prev))) < 1e-12:
            break
    return v


def _codes(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Integer codes for a label array, and how many distinct labels there are."""
    codes, uniq = pd.factorize(pd.Index(labels), sort=False)
    return np.asarray(codes, dtype=np.intp), len(uniq)


def interaction_rho(preds: pd.DataFrame, pred_col: str = "y_resid", min_n: int = 3) -> float:
    """Drug-specific (organoid x drug interaction) rank correlation.

    Removes BOTH the organoid main effect and the drug main effect from observed and
    predicted before the within-drug correlation. What remains is whether the model
    predicts that an organoid responds to *this* drug better or worse than its overall
    sensitivity and the drug's overall potency imply. This is the headline: it measures
    drug-specific signal, the part a shared slope cannot produce.

    Both effects are removed jointly (``_two_way_demean``). Removing only the organoid
    mean is WRONG on an incomplete panel, and was the behaviour here until 2026-08-24:
    each organoid was centred by the mean over the drugs IT happens to have, so a
    predictor constant within a drug -- one that cannot know which organoid it is scoring
    -- picked up organoid-to-organoid variation purely from differing coverage. Measured
    consequence on the real published predictions: the per-drug prior, which contains no
    organoid information at all, scored -0.2345 rather than 0, identically for all 24 rows
    of the Check-2 grid, so every reported interaction sat on that offset.
    """
    p = preds.copy()
    pat = p["patient"].to_numpy()
    drg = p["drug"].to_numpy()
    p["_t"] = _two_way_demean(p["y_true"].to_numpy(dtype=float), pat, drg)
    p["_p"] = _two_way_demean(p[pred_col].to_numpy(dtype=float), pat, drg)
    # A predictor carrying no interaction -- a per-drug constant, or one per-organoid
    # offset -- demeans to floating-point dust (~1e-17). Ranking that would manufacture a
    # correlation out of rounding error. Return 0 outright.
    scale = float(np.std(preds[pred_col].to_numpy(dtype=float)))
    if float(np.std(p["_p"].to_numpy(dtype=float))) <= 1e-9 * max(scale, 1e-30):
        return 0.0
    return _within_drug_corr(p, "_t", "_p", min_n)


def global_spearman(preds: pd.DataFrame) -> float:
    return float(np.asarray(spearmanr(preds["y_true"], preds["y_pred"]))[0])


def regret_norm_at_k(preds: pd.DataFrame, ks: tuple[int, ...] = (1, 3, 5)) -> dict[int, float]:
    """Mean normalized regret@k over patients (lower is better; 0 is best).

    ``y_true`` / ``y_pred`` are AUC-like, so a lower value is a more sensitive (better)
    drug. For each patient the drugs are ranked by ascending ``y_pred`` (predicted best
    first); for the top-k picks, regret is the gap between the best *observed* response
    among them and the patient's actual best, normalized by that patient's observed
    spread so it is panel-size invariant. 0 means the top-k shortlist already contains
    the patient's best drug. Patients with fewer than 2 drugs or no spread are skipped.
    """
    acc: dict[int, list[float]] = {k: [] for k in ks}
    for _, g in preds.groupby("patient"):
        yt = g["y_true"].to_numpy(dtype=np.float64)
        yp = g["y_pred"].to_numpy(dtype=np.float64)
        rng = float(yt.max() - yt.min())
        if len(yt) < 2 or rng <= 0.0:
            continue
        order = np.argsort(yp, kind="stable")  # predicted best (lowest AUC) first
        best = float(yt.min())
        for k in ks:
            topk = order[:k]
            acc[k].append((float(yt[topk].min()) - best) / rng)
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}


def per_drug_spearman(preds: pd.DataFrame, pred_col: str = "y_pred", min_n: int = 3) -> float:
    """Median within-drug Spearman across drugs, KEEPING the cell-line main effect.

    For each drug, the rank correlation between observed and predicted response across cell
    lines; the median over drugs. Unlike ``interaction_rho`` this does NOT remove each line's
    overall sensitivity, so it also rewards predicting which lines are broadly sensitive -- the
    literature-standard per-drug metric (Kurilov et al. 2020), sitting between ``global`` and
    ``interaction``. Drugs with fewer than ``min_n`` lines or no spread are skipped."""
    rhos: list[float] = []
    for _, g in preds.groupby("drug"):
        yt = g["y_true"].to_numpy(dtype=np.float64)
        yp = g[pred_col].to_numpy(dtype=np.float64)
        if len(g) >= min_n and yt.std() > 0 and yp.std() > 0:
            rho = float(np.asarray(spearmanr(yt, yp))[0])
            if np.isfinite(rho):
                rhos.append(rho)
    return float(np.median(rhos)) if rhos else float("nan")


def _interaction_from_demeaned(
    t_dem: np.ndarray, p_dem: np.ndarray, drug: np.ndarray, scale: float, min_n: int = 3
) -> float:
    """interaction_rho's tail, on already-demeaned arrays. Same result, no re-demeaning."""
    if float(np.std(p_dem)) <= 1e-9 * max(scale, 1e-30):
        return 0.0
    frame = pd.DataFrame({"drug": drug, "_t": t_dem, "_p": p_dem})
    return _within_drug_corr(frame, "_t", "_p", min_n)



def score_predictions(
    preds: pd.DataFrame, *, n_perm: int = 1000, seed: int = 0
) -> dict[str, float]:
    """Score a predictions frame (patient, drug, y_true, y_pred; AUC-like, lower = better).

    One place for the composition the eval scripts share: global Spearman, the headline
    interaction rho (organoid x drug), its within-drug label-permutation p-value, the per-drug
    Spearman (keeps the line main effect), and normalized regret@{1,3}. Returns a flat dict of
    floats (``n`` is the pair count).

    Also returns the permutation null's own spread. ``null_p95`` is this row's minimum
    significant interaction -- its detection floor -- measured on the real data at the real
    n, p and fold structure, in the same units as ``interaction``. The null array was already
    being built and then discarded; reporting it is what lets a null result say "we could have
    detected X and we observe Y" rather than merely "we found nothing".

    ``p_label`` uses the (1 + count) / (1 + n_perm) estimator, so the smallest value it can
    report is 1 / (n_perm + 1). The uncorrected ``mean(null >= observed)`` used before
    2026-08-24 could print 0.000, claiming more resolution than the test has."""
    from fmharness.controls import permute_within_drug  # local import: avoid an import cycle

    # Score the RESIDUAL when a prior is supplied. y_pred carries the leave-fold-out per-drug
    # intercept, which is anti-correlated with the held-out truth by construction and offsets
    # every representation equally (-0.312 measured on the real design). Removing it is what
    # probe/estimator.py's base/residual contract has always prescribed; Check 2 did not do it
    # until 2026-08-25. Without a prior column this is a no-op, so existing callers are unchanged.
    scored = preds
    if "y_prior" in preds.columns:
        scored = preds.assign(y_pred=preds["y_pred"] - preds["y_prior"])
    it = interaction_rho(scored, "y_pred")
    # y_pred is identical across permutations -- only the labels move -- so demean it once
    # rather than once per draw.
    _pat = scored["patient"].to_numpy()
    _drg = scored["drug"].to_numpy()
    _p_dem = _two_way_demean(scored["y_pred"].to_numpy(dtype=float), _pat, _drg)
    _scale = float(np.std(scored["y_pred"].to_numpy(dtype=float)))
    null = np.array(
        [
            _interaction_from_demeaned(
                _two_way_demean(
                    permute_within_drug(
                        cast("pd.Series", scored["drug"]),
                        cast("pd.Series", scored["y_true"]),
                        np.random.default_rng(seed + 1 + b),
                    ),
                    _pat,
                    _drg,
                ),
                _p_dem,
                _drg,
                _scale,
            )
            for b in range(n_perm)
        ]
    )
    regret = regret_norm_at_k(preds)
    return {
        "global": round(global_spearman(preds), 3),
        "interaction": round(it, 3),
        "perdrug": round(per_drug_spearman(preds), 3),
        "p_label": round(float((1 + np.sum(null >= it)) / (1 + n_perm)), 4),
        "null_p95": round(float(np.quantile(null, 0.95)), 3),
        "null_sd": round(float(np.std(null, ddof=1)), 3),
        "regret@1": round(regret.get(1, float("nan")), 3),
        "regret@3": round(regret.get(3, float("nan")), 3),
        "n": float(len(preds)),
    }


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation between every row of ``a`` (m x g) and ``b`` (n x g),
    returned as an ``m x n`` matrix. Each row is centered and L2-normalized, so the matmul
    of the normalized matrices is the correlation; a constant (zero-variance) row becomes a
    zero vector and therefore correlates 0 with everything (rather than NaN)."""

    def _unit(m: np.ndarray) -> np.ndarray:
        c = m - m.mean(axis=1, keepdims=True)
        nrm = np.linalg.norm(c, axis=1, keepdims=True)
        nrm[nrm == 0.0] = 1.0
        return c / nrm

    return _unit(np.asarray(a, dtype=np.float64)) @ _unit(np.asarray(b, dtype=np.float64)).T


def delta_fidelity(
    pred_delta: pd.DataFrame,
    pred_key: pd.DataFrame,
    real_delta: pd.DataFrame,
    real_key: pd.DataFrame,
    *,
    n_hvg: int | None = 2000,
) -> pd.DataFrame:
    """Faithfulness of a predicted expression delta to the real one, per (patient, drug).

    For every (patient, drug) present in both sources, the Pearson correlation between the
    predicted and the real log-fold-change profile over genes -- Stack's own generation
    metric (predicted vs observed expression *changes*), and the data-level concordance metric
    for pseudobulk-vs-bulk. To expose the failure mode a smooth, non-specific predictor hides
    (every profile correlates with every other), it also reports, per matched pair, the mean
    correlation to the *wrong* real pairs (``r_offdiag``) and the matched pair's specificity
    rank among all real pairs (``rank``; 1.0 = the right pair is the single best match). A
    faithful, specific predictor has ``r`` >> ``r_offdiag`` and ``rank`` ~ 1.

    ``n_hvg`` restricts scoring to the most variable genes of the real delta across the matched
    pairs (mirroring the paper's top-2000 log-normalized HVGs); ``None`` uses all shared genes.
    Returns one row per matched pair: ``patient, drug, r, r_offdiag, rank, n_genes``.
    """
    genes = pred_delta.columns.intersection(real_delta.columns)
    if len(genes) == 0:
        raise ValueError("pred_delta and real_delta share no genes")
    pk, rk = pred_key.reset_index(drop=True), real_key.reset_index(drop=True)
    m = pk.assign(_i=np.arange(len(pk))).merge(
        rk.assign(_j=np.arange(len(rk))), on=["patient", "drug"], how="inner"
    )
    if m.empty:
        raise ValueError("pred and real share no (patient, drug) pairs")
    p = pred_delta[genes].to_numpy(dtype=np.float64)[m["_i"].to_numpy()]
    r = real_delta[genes].to_numpy(dtype=np.float64)[m["_j"].to_numpy()]
    if n_hvg is not None and len(m) > 1 and n_hvg < len(genes):
        top = np.argsort(r.var(axis=0))[::-1][:n_hvg]
        p, r = p[:, top], r[:, top]

    c = _row_corr(p, r)  # (pairs x pairs); the matched correlations are the diagonal
    matched = np.diag(c).copy()
    n = len(matched)
    if n > 1:
        r_offdiag = (c.sum(axis=1) - matched) / (n - 1)
        rank = (c < matched[:, None]).sum(axis=1) / (n - 1)  # frac of wrong pairs below matched
    else:
        r_offdiag = np.full(n, np.nan)
        rank = np.full(n, np.nan)
    return pd.DataFrame(
        {
            "patient": m["patient"].to_numpy(),
            "drug": m["drug"].to_numpy(),
            "r": matched,
            "r_offdiag": r_offdiag,
            "rank": rank,
            "n_genes": int(p.shape[1]),
        }
    )


def score_delta_sources(
    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    real_delta: pd.DataFrame,
    real_key: pd.DataFrame,
    *,
    n_hvg: int | None = 2000,
) -> pd.DataFrame:
    """Check-1 table: one row per delta source, scored against the same real Tahoe delta.

    ``sources`` maps a source name (``"observed_delta"``, ``"stack"``, ...) to its own
    ``(delta, key)`` pair, exactly as produced by ``loo_baseline_source`` /
    ``build_generated_deltas``. Every source is first restricted to the (patient, drug)
    support they all share (``restrict_common_support``, 2026-08-21) -- a broadcast
    baseline like ``additive`` can natively cover a much wider (patient, drug) set than a
    narrowly-generated source like ``stack``, and scoring each against its own native
    intersection with ``real_key`` (as a per-source ``delta_fidelity`` call alone would)
    compares different evaluation sets, not just different methods -- exactly the bug
    fixed in ``scripts/score_viability_adapters.py`` (commit d9f94ec). Each restricted
    source is then scored independently via ``delta_fidelity`` against the same
    ``real_delta``/``real_key`` -- the shared scoring step both
    ``scripts/score_generation_eval.py`` and ``scripts/check1_registry_driver.py``
    need, kept here so the two never drift on how a row is built, and both get the
    common-support restriction for free.
    """
    from fmharness.deltas import restrict_common_support  # local import: avoid an import
    # cycle -- fmharness.deltas imports build_sample_design from this module.

    sources = restrict_common_support(sources, real_key)
    rows: list[dict[str, object]] = []
    for name, (d, kk) in sources.items():
        f = delta_fidelity(d, kk, real_delta, real_key, n_hvg=n_hvg)
        rows.append(
            {
                "source": name,
                "r": round(float(f["r"].mean()), 3),
                "r_offdiag": round(float(f["r_offdiag"].mean()), 3),
                "rank": round(float(f["rank"].mean()), 3),
                "n_pairs": len(f),
                "n_genes": int(f["n_genes"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def de_fidelity(
    pred_delta: pd.DataFrame,
    pred_key: pd.DataFrame,
    de_calls: pd.DataFrame,
) -> pd.DataFrame:
    """DE-based faithfulness of a predicted expression delta, per (patient, drug), against
    ground-truth Wilcoxon DE calls (``fmharness.deltas.build_tahoe_de_calls``).

    For every (patient, drug) present in both ``pred_key`` and ``de_calls``, computes four
    metrics matching the Stack paper's cell-eval-based DE evaluation (Methods 4.6.3): DE Spearman
    LFC (Spearman rank correlation between predicted delta and real log2FC, restricted to the
    real-significant genes), PR-AUC (average precision of ``|predicted delta|`` as a score against
    the real ``significant`` binary label, over all tested genes), and DE Overlap Accuracy /
    Jaccard similarity (both from the top-N genes by ``|predicted delta|``, N = the number of
    real-significant genes for that pair, against the real-significant gene set -- the paper's own
    top-N-overlap definition). Our predicted side has no per-cell distribution to run a formal
    significance test against (a single generated delta per line, not multiple cells to test), so
    it is ranked by ``|predicted delta|`` alone in place of a predicted p-value -- the design's
    sanctioned adaptation, since only the ground truth needs a formal significance call.

    Returns one row per matched pair: ``patient, drug, de_spearman_lfc, pr_auc,
    de_overlap_accuracy, jaccard, n_sig_genes``. A pair with zero real-significant genes has
    ``de_spearman_lfc``/``de_overlap_accuracy``/``jaccard`` as NaN (undefined without at least one
    true positive); if every gene in that pair is one single class (all- or none-significant),
    ``pr_auc`` (needs both classes present) is NaN too -- both cases explicit, never silently
    defaulted to 0 or 1.
    """
    pk = pred_key.reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for key, grp in de_calls.groupby(["patient", "drug"]):
        patient, drug = cast("tuple[str, str]", key)
        match = pk[(pk["patient"] == patient) & (pk["drug"] == drug)]
        if match.empty:
            continue
        i = cast(int, match.index[0])
        genes = grp["gene"].to_numpy()
        pred_row = pred_delta.iloc[[i]].reindex(columns=genes).iloc[0].to_numpy(dtype=np.float64)
        have = ~np.isnan(pred_row)
        if not have.any():
            continue
        genes, pred_row = genes[have], pred_row[have]
        by_gene = grp.set_index("gene").loc[genes]
        real_lfc = by_gene["log2fc"].to_numpy(dtype=np.float64)
        sig = by_gene["significant"].to_numpy(dtype=bool)
        n_sig = int(sig.sum())

        pr_auc = (
            float(average_precision_score(sig, np.abs(pred_row)))
            if 0 < n_sig < len(sig)
            else float("nan")
        )
        if n_sig == 0:
            de_spearman_lfc = overlap = jaccard = float("nan")
        else:
            de_spearman_lfc = float(np.asarray(spearmanr(pred_row[sig], real_lfc[sig]))[0])
            order = np.argsort(-np.abs(pred_row))
            pred_top_n = set(genes[order[:n_sig]])
            true_sig = set(genes[sig])
            inter = len(pred_top_n & true_sig)
            overlap = inter / n_sig
            union = len(pred_top_n | true_sig)
            jaccard = inter / union if union else float("nan")
        rows.append(
            {
                "patient": patient,
                "drug": drug,
                "de_spearman_lfc": de_spearman_lfc,
                "pr_auc": pr_auc,
                "de_overlap_accuracy": overlap,
                "jaccard": jaccard,
                "n_sig_genes": n_sig,
            }
        )
    if not rows:
        raise ValueError("pred_key and de_calls share no (patient, drug) pairs")
    return pd.DataFrame(rows)


def score_de_metrics(
    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    de_calls: pd.DataFrame,
) -> pd.DataFrame:
    """DE-metrics analogue of ``score_delta_sources``: one row per delta source, averaged over
    its matched (patient, drug) pairs' DE Spearman LFC / PR-AUC / DE Overlap Accuracy / Jaccard
    (``de_fidelity``), against the same ground-truth DE-calls bundle
    (``fmharness.deltas.build_tahoe_de_calls``). Pairs with zero real-significant genes contribute
    NaN to the rank-based columns for that source and are excluded from those means via pandas'
    default ``skipna``, but still count toward the ``pr_auc`` mean when it is defined.
    """
    rows: list[dict[str, object]] = []
    for name, (d, kk) in sources.items():
        f = de_fidelity(d, kk, de_calls)
        rows.append(
            {
                "source": name,
                "de_spearman_lfc": round(float(f["de_spearman_lfc"].mean()), 3),
                "pr_auc": round(float(f["pr_auc"].mean()), 3),
                "de_overlap_accuracy": round(float(f["de_overlap_accuracy"].mean()), 3),
                "jaccard": round(float(f["jaccard"].mean()), 3),
                "n_pairs": len(f),
            }
        )
    return pd.DataFrame(rows)
