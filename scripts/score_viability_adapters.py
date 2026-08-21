"""Compare delta sources x viability adapters against the Soragni AUC target.

The generation axis must be fair: the readout adapters (l1/l2 CV-tuned penalized
regression supervised on real L1000 deltas vs GDSC2 AUC; hallmark unsupervised) are
applied to EVERY delta source, not just Stack's. Sources:

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

from fmharness.adapters import ALL_METHODS, build_adapters
from fmharness.data.loaders import load_tranche
from fmharness.deltas import (
    build_additive_deltas,
    build_generated_deltas,
    build_l1000_gdsc_pairs,
    build_learned_deltas,
    restrict_common_support,
    soragni_pert_map,
)
from fmharness.evaluation import build_sample_design, score_predictions
from fmharness.signatures import SIGNATURES, load_hallmark


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
        for adapter in build_adapters(methods, signatures=sigs):
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

    results = pd.DataFrame(out)
    print("\n=== delta source x viability adapter vs Soragni AUC ===")
    print(results.to_string(index=False))
    if args.out_csv:
        results.to_csv(args.out_csv, index=False)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
