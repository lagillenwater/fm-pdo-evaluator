"""Compare delta sources x viability adapters against the Soragni AUC target.

The generation axis must be fair: the readout adapters (l1/l2 CV-tuned penalized
regression supervised on real L1000 deltas vs GDSC2 AUC; hallmark unsupervised) are
applied to EVERY delta source, not just Stack's. The "hallmark" method expands to one
row per individual Hallmark signature (``_build_readout_adapters`` /
``build_hallmark_breakout``, 2026-08-21), not one score averaged across all of them --
pass ``--hallmark-sets`` to restrict which signatures are scored at all (e.g. just
HALLMARK_E2F_TARGETS,HALLMARK_G2M_CHECKPOINT for proliferation only). Sources:

  - ``additive`` (always): each drug's mean real L1000 delta, applied to every patient
    (patient-independent) -- the generation analogue of the drug-mean baseline. The
    floor Stack must beat: it carries the drug main effect but no patient x drug
    interaction.
  - ``stack`` (when --generated-dir is given): Stack-generated patient-specific deltas.

Every (source, adapter) cell is scored against the real Soragni AUC with the same
global / interaction rho + within-drug label-permutation null, so Stack's generated
delta is compared head-to-head against the additive baseline under each readout.
Run on Alpine (needs the L1000 .gctx for the training cohort and additive source).

Before scoring, every source is restricted to the SAME (patient, drug) support
(``restrict_common_support``, 2026-08-21): sources have very different native
coverage -- additive/pca/nmf broadcast over the whole L1000 training cohort (mostly
unlabeled for a given patient) while stack only covers its own generated drug set --
so without this, each source's ``n`` differed (stack n=202 vs additive/pca/nmf n=150
in the first l1/l2 rerun) and the interaction/global numbers were being compared
across different evaluation sets, not just different methods.

--baseline must be the same tumor-RNA query file the generation step used
(``stack_input_sarcoma.h5ad``, per the June 2026-06-26 "use tumor RNA as the Soragni
model input" switch), NOT ``stack_input_soragni.h5ad`` -- a pre-switch, organoid-RNA
artifact left behind as a stale default until 2026-08-20 (correlation as low as 0.57
against stack_input_sarcoma.h5ad on the same patient IDs; using it silently subtracts
a mismatched-substrate baseline from every non-additive delta source).

--stack-emb width vs pca/nmf's small, sample-size-capped component counts is not truncated
to match: ``load_line_matrix`` loads the embedding as-is (no dimensionality reduction), and
it flows UNTRUNCATED into ``penalized_preds``' CV-tuned RidgeCV/LassoCV (``make_penalty``).
Ridge is well-posed for p >> n, so the CV-tuned penalty automatically shrinks effective
capacity to what the small Soragni cohort supports, without needing to truncate the
embedding itself -- truncating would throw away real learned structure to force a capacity
match that the CV-tuned penalty already provides for free. Each --stack-emb representation
is also scored against a same-width i.i.d.-standard-normal negative control
(``fmharness.check2.random_feature_control``, as ``f"{label}_random"``) through the
identical penalized_preds pipeline, so an apparent win is attributable to learned structure
rather than raw embedding width (same treatment as Check 2's stack_emb loop). Every
generation-based source (additive/pca/nmf/stack) gets the SAME random-feature control for
its l1/l2 rows too, fit AND applied on matching-shape noise (not just noise fed through a
model trained on real genes, which would test something unrelated) -- ``f"{source}_random"``.

No oracle/ceiling VALIDATION (the real measured post-treatment delta scored as its own
"prediction", as Check 1/2's driver scripts now add via ``score_check2``'s ``oracle=``):
Soragni has no real treated-organoid RNA-seq, only the untreated tumor-RNA baseline -- there
is nothing to feed as a ceiling input here, unlike Tahoe/GDSC2 where the real per-(line, drug)
delta exists. DOES get the flowchart's actual "positive control" (planted interaction,
recovered, ``fmharness.controls.plant_interaction``) -- a ``"planted"`` row using the
tumor-RNA baseline, since that data gap doesn't apply to a simulation.

  PYTHONPATH=src python scripts/score_viability_adapters.py --l1000-dir . \\
      --gctx GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx \\
      --generated-dir generated_rich/ --baseline data/reference/stack_input_sarcoma.h5ad
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fmharness.adapters import ALL_METHODS, ViabilityAdapter, build_adapters, build_hallmark_breakout
from fmharness.check2 import load_line_matrix, penalized_preds, random_feature_control, seed_for_name
from fmharness.controls import plant_interaction
from fmharness.data.loaders import load_tranche
from fmharness.deltas import (
    build_additive_deltas,
    build_generated_deltas,
    build_l1000_gdsc_pairs,
    build_learned_deltas,
    restrict_common_support,
    soragni_pert_map,
)
from fmharness.drug_targets import score_target_gene_predictors
from fmharness.evaluation import build_sample_design, score_predictions
from fmharness.signatures import SIGNATURES, load_hallmark


def _build_readout_adapters(
    methods: list[str], sigs: dict[str, tuple[tuple[str, ...], int]] | None
) -> list[ViabilityAdapter]:
    """Same selection as ``build_adapters``, except ``"hallmark"`` expands to one adapter
    PER Hallmark signature (``build_hallmark_breakout``) instead of one score averaged
    across every signature -- on Tahoe only the proliferation sets (E2F/G2M) cleared the
    random-gene-set control while P53/apoptosis added noise, so a blended score can hide
    a real per-pathway signal (docs/tahoe_generation_results.md's Gate table)."""
    rest = [m for m in methods if m != "hallmark"]
    out: list[ViabilityAdapter] = []
    if "hallmark" in methods:
        if sigs is None:
            raise ValueError("the hallmark adapter requires signatures=")
        out.extend(build_hallmark_breakout(sigs))
    if rest:
        out.extend(build_adapters(rest, signatures=sigs))
    return out


def _read_baseline(path: Path) -> pd.DataFrame:
    """Soragni tumor-RNA baseline AnnData -> DataFrame (patient x gene symbol)."""
    a = ad.read_h5ad(path)
    x = a.X
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    return pd.DataFrame(
        np.asarray(x, dtype=np.float64),
        index=pd.Index([str(o) for o in a.obs_names]),
        columns=pd.Index([str(g) for g in a.var_names]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l1000-dir", default=".")
    ap.add_argument("--gctx", required=True)
    ap.add_argument(
        "--generated-dir",
        default=None,
        help="Stack-generated per-drug .h5ad dir; omit to score the additive baseline only",
    )
    ap.add_argument("--baseline", default="data/reference/stack_input_sarcoma.h5ad")
    ap.add_argument(
        "--methods",
        default=",".join(ALL_METHODS),
        help="comma-separated subset of hallmark,l1,l2",
    )
    ap.add_argument(
        "--signatures",
        choices=["curated", "hallmark"],
        default="hallmark",
        help="gene sets for the hallmark adapter",
    )
    ap.add_argument(
        "--hallmark-sets",
        default=None,
        help="comma-separated Hallmark set names to restrict the hallmark adapter to "
        "(default: all four loaded from --signatures hallmark); pass "
        "HALLMARK_E2F_TARGETS,HALLMARK_G2M_CHECKPOINT to score proliferation only -- "
        "on Tahoe, the only two sets that beat a random gene set (docs/tahoe_generation_"
        "results.md's Gate table), so averaging in P53/apoptosis may just be diluting "
        "signal with noise. Requires --signatures hallmark.",
    )
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument(
        "--stack-emb",
        nargs="+",
        default=None,
        metavar="LABEL=PATH",
        help="Stack embedding representations (label=path.h5ad, one row per patient -- e.g. "
        "base=emb_soragni_base.h5ad aligned=emb_soragni_aligned.h5ad, from "
        "scripts/alpine/13_soragni_embed.sbatch). Scored via leave-patient-out penalized "
        "regression against the real Soragni AUC design, same treatment as Check 2's "
        "stack_emb representations (fmharness.check2.penalized_preds) -- l1/l2 only, no "
        "hallmark (an embedding has no gene identity to score a signature against).",
    )
    ap.add_argument(
        "--folds",
        type=int,
        default=5,
        help="leave-patient-out CV folds for --stack-emb scoring (same default as Check 2)",
    )
    ap.add_argument(
        "--out-csv",
        default=None,
        help="also write the source x adapter results table here (for downstream plotting)",
    )
    ap.add_argument("--time", type=float, default=24.0)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--treated-cap", type=int, default=8)
    ap.add_argument("--dmso-cap", type=int, default=60)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    sigs: dict[str, tuple[tuple[str, ...], int]] | None = None
    if "hallmark" in methods:
        sigs = (
            load_hallmark(repo / "data/static/hallmark_signatures.gmt")
            if args.signatures == "hallmark"
            else SIGNATURES
        )
        if args.hallmark_sets:
            if args.signatures != "hallmark":
                raise ValueError("--hallmark-sets requires --signatures hallmark")
            keep = {s.strip() for s in args.hallmark_sets.split(",") if s.strip()}
            missing = keep - set(sigs)
            if missing:
                raise ValueError(
                    f"--hallmark-sets: unknown set(s) {sorted(missing)}; have {sorted(sigs)}"
                )
            sigs = {k: v for k, v in sigs.items() if k in keep}

    # Key the Soragni target by PubChem CID: the delta sources (additive / learned /
    # stack) all key drugs by CID, so the target must too or the merge below is empty.
    # (The native loader sets Soragni drug_id = drug name, unlike the old CoderData ids.)
    _, design = build_sample_design(
        load_tranche("sarcoma", repo), "tumor", "viability", drug_key="pubchem_cid"
    )

    # train cohort: real L1000 deltas -> GDSC2 AUC (for the supervised adapters and the
    # additive baseline). Keep the full delta for the additive per-drug mean; fit the
    # supervised adapters on the subset that has a GDSC2 AUC label.
    tr_delta, tr_key, dg, tr_base = build_l1000_gdsc_pairs(
        repo,
        Path(args.l1000_dir),
        args.gctx,
        time=args.time,
        chunk=args.chunk,
        treated_cap=args.treated_cap,
        dmso_cap=args.dmso_cap,
    )
    tr_via_all = tr_key.merge(dg.rename(columns={"y": "_y"}), on=["patient", "drug"], how="left")[
        "_y"
    ].to_numpy()
    ok = ~np.isnan(tr_via_all)
    tr_delta_fit, tr_via = tr_delta[ok], tr_via_all[ok]

    # delta sources, fed through the SAME readout adapters:
    #   additive  -- drug-mean L1000 delta (organoid-independent floor)
    #   pca / nmf -- learned organoid-specific delta predictors (need the Soragni baseline)
    #   stack     -- Stack-generated organoid-specific delta (when --generated-dir given)
    patients = sorted(str(p) for p in design["patient"].unique())
    base_path = Path(args.baseline) if Path(args.baseline).is_absolute() else repo / args.baseline
    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "additive": build_additive_deltas(tr_delta, tr_key, patients)
    }
    if base_path.exists():
        soragni_base = _read_baseline(base_path)
        for reducer in ("pca", "nmf"):
            sources[reducer] = build_learned_deltas(
                tr_base, tr_delta, tr_key, soragni_base, patients, reducer=reducer
            )
    else:
        print(f"(skipping pca/nmf sources: baseline {base_path} not found)")
    if args.generated_dir:
        sources["stack"] = build_generated_deltas(
            Path(args.generated_dir), base_path, soragni_pert_map(repo)
        )

    # Restrict every source to the SAME (patient, drug) support before scoring: sources have
    # very different native coverage (stack only covers its own generated drug set; additive/
    # pca/nmf broadcast over the whole L1000 training cohort, mostly unlabeled for a given
    # patient), so scoring each against its own native intersection with `design` compares
    # different evaluation sets, not just different methods (see restrict_common_support).
    native_n = {name: len(skey) for name, (_, skey) in sources.items()}
    sources = restrict_common_support(sources, design)
    common_n = len(next(iter(sources.values()))[1])
    print(
        f"\ncommon (patient, drug) support across {list(sources)}: {common_n} pairs "
        f"(native source rows: {', '.join(f'{k}={v}' for k, v in native_n.items())})"
    )

    out: list[dict[str, object]] = []
    for src_name, (sdelta, skey) in sources.items():
        common = tr_delta_fit.columns.intersection(sdelta.columns)
        tr_x, sx = tr_delta_fit[common], sdelta[common]
        print(
            f"[{src_name}] train {len(tr_x)} pairs | source {len(sx)} pairs | "
            f"{len(common)} shared genes | methods {methods}"
        )
        for adapter in _build_readout_adapters(methods, sigs):
            if adapter.supervised:
                adapter.fit(tr_x, tr_via)
            sens = adapter.predict(sx)
            merged = pd.DataFrame(
                {
                    "patient": skey["patient"].to_numpy(),
                    "drug": skey["drug"].to_numpy(),
                    "_sens": sens,
                }
            ).merge(design.rename(columns={"y": "y_true"}), on=["patient", "drug"], how="inner")
            preds = pd.DataFrame(
                {
                    "patient": merged["patient"],
                    "drug": merged["drug"],
                    "y_true": merged["y_true"].to_numpy(),
                    "y_pred": -merged["_sens"].to_numpy(),
                }
            )
            s = score_predictions(preds, n_perm=args.n_permutations)
            out.append(
                {
                    "source": src_name,
                    "method": adapter.name,
                    "global": s["global"],
                    "interaction": s["interaction"],
                    "p_label": s["p_label"],
                    "regret@1": s["regret@1"],
                    "regret@3": s["regret@3"],
                    "n": int(s["n"]),
                }
            )

    # Same-width random-feature negative control for the supervised (l1/l2) adapters, per
    # source -- not just --stack-emb: fit AND apply on matching-shape random noise (not just
    # predict noise through a model trained on real genes, which would test something
    # unrelated) so an apparent l1/l2 win on ANY delta source can be checked against raw
    # dimensionality alone, the same treatment every representation gets in Check 2's grid
    # (fmharness.check2.random_feature_control).
    for src_name, (sdelta, skey) in sources.items():
        common = tr_delta_fit.columns.intersection(sdelta.columns)
        # seed derived from src_name (not a fixed literal): two sources that happen to
        # reduce to the same shared-gene shape (e.g. pca/nmf) would otherwise get the
        # exact same "random" draw instead of independent ones.
        src_seed = seed_for_name(src_name)
        tr_x_rand = random_feature_control(tr_delta_fit[common], seed=src_seed)
        sx_rand = random_feature_control(sdelta[common], seed=(src_seed + 1) & 0xFFFFFFFF)
        for adapter in _build_readout_adapters(methods, sigs):
            if not adapter.supervised:
                continue
            adapter.fit(tr_x_rand, tr_via)
            sens = adapter.predict(sx_rand)
            merged = pd.DataFrame(
                {
                    "patient": skey["patient"].to_numpy(),
                    "drug": skey["drug"].to_numpy(),
                    "_sens": sens,
                }
            ).merge(design.rename(columns={"y": "y_true"}), on=["patient", "drug"], how="inner")
            preds = pd.DataFrame(
                {
                    "patient": merged["patient"],
                    "drug": merged["drug"],
                    "y_true": merged["y_true"].to_numpy(),
                    "y_pred": -merged["_sens"].to_numpy(),
                }
            )
            s = score_predictions(preds, n_perm=args.n_permutations)
            out.append(
                {
                    "source": f"{src_name}_random",
                    "method": adapter.name,
                    "global": s["global"],
                    "interaction": s["interaction"],
                    "p_label": s["p_label"],
                    "regret@1": s["regret@1"],
                    "regret@3": s["regret@3"],
                    "n": int(s["n"]),
                }
            )

    # Stack embedding representations (base / aligned checkpoints, --stack-emb): unlike the
    # generation-based sources above, these fit/predict directly on the real Soragni AUC via
    # leave-patient-out CV (fmharness.check2.penalized_preds) -- the same treatment Check 2
    # gives its stack_emb representations, no cross-domain L1000->GDSC2 transfer involved, so
    # they are not run through restrict_common_support (that's a delta-source, key-frame
    # concept; penalized_preds already scores every representation on its own per-drug CV
    # folds against the same design_target the same way Check 2 does).
    if args.stack_emb:
        uniq_lines = patients
        n_folds = max(1, min(args.folds, len(uniq_lines)))
        fold_of = {ln: i % n_folds for i, ln in enumerate(uniq_lines)}
        emb_methods = [m for m in methods if m in ("l1", "l2")]
        for spec in args.stack_emb:
            label, sep, path = spec.partition("=")
            if not sep or not label or not path:
                raise ValueError(f"--stack-emb expects 'label=path', got {spec!r}")
            emb_path = Path(path) if Path(path).is_absolute() else repo / path
            emb = load_line_matrix(emb_path)
            print(f"[{label}] embedding: {len(emb)} patients x {emb.shape[1]} dims | methods {emb_methods}")
            # same-width random-feature negative control (random_feature_control) -- must clear
            # this by a real margin, or an apparent win is attributable to raw embedding width
            # rather than learned structure (same treatment as Check 2's stack_emb loop). Seed
            # derived from label (not a fixed literal) -- two labels of the same width (e.g.
            # base/aligned) would otherwise get the identical "random" draw.
            emb_sources = {
                label: emb,
                f"{label}_random": random_feature_control(emb, seed=seed_for_name(label)),
            }
            for src_label, src_emb in emb_sources.items():
                for pen in emb_methods:
                    preds = penalized_preds(
                        (lambda e: lambda _drug: e)(src_emb), design, fold_of, n_folds, uniq_lines, pen
                    )
                    if preds.empty:
                        print(f"  {src_label}/{pen}: no scorable (patient, drug) pairs -- skipping")
                        continue
                    s = score_predictions(preds, n_perm=args.n_permutations)
                    out.append(
                        {
                            "source": src_label,
                            "method": pen,
                            "global": s["global"],
                            "interaction": s["interaction"],
                            "p_label": s["p_label"],
                            "regret@1": s["regret@1"],
                            "regret@3": s["regret@3"],
                            "n": int(s["n"]),
                        }
                    )

    # Known-biology positive control: each drug's own molecular target gene's baseline
    # expression (fmharness.drug_targets, target-dependency hypothesis: higher target
    # expression -> more sensitive to inhibiting it). Complements the oracle/ceiling
    # VALIDATION Check 1/2's driver scripts add via score_check2's oracle= -- Soragni has
    # no real treated-organoid RNA to build that kind of oracle/validation from (see this
    # script's module docstring), but DOES have the tumor-RNA baseline this control needs.
    if base_path.exists():
        tgt = score_target_gene_predictors(design, soragni_base, n_perm=args.n_permutations)
        if tgt["n"]:
            print(f"\n[target_gene] known-biology positive control: n={int(tgt['n'])} pairs")
            out.append(
                {
                    "source": "target_gene",
                    "method": "direct",
                    "global": tgt["global"],
                    "interaction": tgt["interaction"],
                    "p_label": tgt["p_label"],
                    "regret@1": tgt["regret@1"],
                    "regret@3": tgt["regret@3"],
                    "n": int(tgt["n"]),
                }
            )
        else:
            print("\n[target_gene] no drug's target gene covered by this baseline -- skipping")

    # Positive control: plant a KNOWN organoid x drug interaction into the tumor-RNA baseline
    # space (fmharness.controls.plant_interaction) and confirm the SAME leave-patient-out
    # penalized grid recovers it -- the flowchart's actual "positive control: planted
    # interaction, recovered" (distinct from the oracle/ceiling VALIDATION Check 1/2 add,
    # which uses real data with no controlled effect size, and from the known-biology
    # target_gene control, which is a real hypothesis, not a simulation). Soragni's data gap
    # (no real treated-organoid RNA) doesn't apply here since nothing is measured -- it's
    # simulated on top of the real tumor-RNA baseline every other Path-B source is built from.
    if base_path.exists():
        uniq_lines = patients
        n_folds = max(1, min(args.folds, len(uniq_lines)))
        fold_of = {ln: i % n_folds for i, ln in enumerate(uniq_lines)}
        # Plant AND score in the SAME small PCA subspace of soragni_base -- planting in the
        # raw gene space (thousands of genes) and then fitting RidgeCV directly on it with
        # only ~n/folds training patients per fold cannot recover ANY signal at any effect
        # size (verified empirically against fmharness.check2.penalized_preds: r2/interaction
        # stayed negative even at 30x effect on unreduced features). n_components is capped
        # well below the smallest per-fold training-patient count so the fit is well-posed.
        plant_k = max(
            1,
            min(5, len(uniq_lines) // (n_folds + 2), soragni_base.shape[1], len(uniq_lines) - 1),
        )
        plant_sc = StandardScaler().fit(soragni_base.to_numpy())
        plant_z = PCA(n_components=plant_k, random_state=0).fit_transform(
            plant_sc.transform(soragni_base.to_numpy())
        )
        plant_z_df = pd.DataFrame(plant_z, index=soragni_base.index)
        emb_per_row = plant_z_df.reindex(design["patient"]).to_numpy()
        within_drug_sd = float(
            np.std(
                design["y"].to_numpy() - design.groupby("drug")["y"].transform("mean").to_numpy()
            )
        )
        plant_scale = within_drug_sd if within_drug_sd > 0 else 1.0
        planted_y = plant_interaction(
            design["drug"],
            design["y"],
            emb_per_row,
            effect=2 * plant_scale,
            noise_sd=plant_scale,
            rng=np.random.default_rng(0),
            n_components=plant_k,
        )
        planted_design = design.assign(y=planted_y)
        for pen in [m for m in methods if m in ("l1", "l2")]:
            preds = penalized_preds(
                lambda _drug: plant_z_df, planted_design, fold_of, n_folds, uniq_lines, pen
            )
            if preds.empty:
                print(f"  planted/{pen}: no scorable (patient, drug) pairs -- skipping")
                continue
            s = score_predictions(preds, n_perm=args.n_permutations)
            out.append(
                {
                    "source": "planted",
                    "method": pen,
                    "global": s["global"],
                    "interaction": s["interaction"],
                    "p_label": s["p_label"],
                    "regret@1": s["regret@1"],
                    "regret@3": s["regret@3"],
                    "n": int(s["n"]),
                }
            )

    results = pd.DataFrame(out)
    print("\n=== delta source x viability adapter vs Soragni AUC ===")
    print(results.to_string(index=False))
    if args.out_csv:
        results.to_csv(args.out_csv, index=False)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
