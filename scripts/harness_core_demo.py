"""End-to-end demo: the harness-core registries driving a real evaluation.

This plan (docs/superpowers/plans/2026-08-07-modular-harness-core.md) built the
`Modality`/`Estimator`/`CVScheme`/`LeakageQueryable` registries but deliberately
left driver integration out of scope. This script is that driver: for each
dataset it

  1. loads features + design in one call (`Modality.load_with_features` --
     `Modality.load()` alone used to force a second, manual, redundant tranche
     load to get the expression matrix; the two datasets this script actually
     runs, Gdsc2Auc and SoragniViability, are unit-tested against real data to
     agree with `load()` exactly -- see tests/test_modality.py's *_matches_load
     tests. CtrpAuc/PrismAuc share the identical code shape but have no local
     data to test against yet),
  2. strips pretraining-contaminated rows (`filter_leakage`) against a model
     that declares its own pretraining corpus (`KnownCorpusAdapter` --
     plain expression PCA has no real corpus to declare, so this simulates the
     check a pretrained encoder like Stack or scFoundation would need for
     real; see tests/test_leakage.py for the exact tiered-drop rule),
  3. resolves the CV scheme from the Modality's own recommendation instead of
     a hand-picked split count (`resolve_cv(modality.recommended_cv())`, then
     `grouped_cv_predict(..., cv=...)`; see tests/test_evaluation.py's
     grouped_cv_predict cv= tests),
  4. fits and scores with the existing, already-validated `SimpleProbe` +
     `interaction_rho` machinery.

(`BiomarkerEstimator` also now runs through `grouped_cv_predict` -- see
tests/test_evaluation.py::test_grouped_cv_predict_drives_biomarker_estimator --
not demoed here since a real biomarker rule table needs WES alteration data
this script doesn't have on hand; fabricating one would look like a real
finding when it isn't.)

Two datasets, each with an already-known expected outcome:

  - GDSC2 sarcoma, within-cohort CV: a known-signal benchmark. A prior run
    (scripts/gdsc_representation_increment.py, raw-count log1p expression,
    drug_id-keyed) scored expression PCA at interaction_rho ~0.224. This
    script normalizes to CPM and keys drugs by pubchem_cid (the harness's now-
    canonical, cross-dataset convention), so this is not an exact replica --
    it should land in a similar range if the new registries are loading the
    same underlying signal correctly, not an exact-digit match.
  - Soragni within-cohort CV: a known-null benchmark. This project has already
    established that within-Soragni expression (n=15-17 organoids, 11
    subtypes) cannot predict response -- Soragni is train-elsewhere/test-only
    in the project's actual pipeline. A near-zero result here is the expected,
    already-documented outcome, not a new finding.

Run:
  uv run python scripts/harness_core_demo.py
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np

from fmharness.cv import resolve_cv
from fmharness.evaluation import grouped_cv_predict, interaction_rho
from fmharness.leakage import filter_leakage
from fmharness.modality import Gdsc2Auc, Modality, SoragniViability
from fmharness.models import KnownCorpusAdapter
from fmharness.probe import SimpleProbe

SEED = 0
REPO = Path(__file__).resolve().parent.parent


def _run(modality: Modality, label: str, *, declared_overlap_lines: int) -> None:
    print(f"=== {label}, within-cohort CV (via the harness-core registries) ===")
    x_df, design = modality.load_with_features(REPO)
    print(
        f"loaded: {len(design)} rows, {design['patient'].nunique()} patients, "
        f"{design['drug'].nunique()} drugs"
    )

    # SYNTHETIC pretraining overlap, declared on a few of this cohort's own real
    # lines/drugs -- plain expression PCA has no real pretrained corpus, so this
    # is illustrative only: it shows filter_leakage actually drops rows before
    # scoring, not a real contamination measurement of a real model. A real
    # pretrained encoder (Stack, scFoundation) would need its own adapter
    # reporting its actual training manifest here.
    lines = set(design["patient"].unique()[:declared_overlap_lines])
    drugs = set(design["drug"].unique()[:2])
    model = KnownCorpusAdapter(
        pretraining_lines=lines, pretraining_drugs=drugs, task_signal_in_pretrain="adjacent"
    )
    filtered, profile = filter_leakage(design, model)
    dropped = len(design) - len(filtered)
    doubly_exposed = (
        f"{profile.doubly_exposed_frac:.3f}" if profile.doubly_exposed_frac is not None else "n/a"
    )
    print(
        f"filter_leakage [SYNTHETIC corpus, illustrative only -- not a real "
        f"contamination measurement]: {len(design)} -> {len(filtered)} rows "
        f"(dropped {dropped} doubly-exposed pairs; basis={profile.basis}, "
        f"doubly_exposed_frac={doubly_exposed}, model={profile.model_version})"
    )

    cv_key = modality.recommended_cv()
    cv = resolve_cv(cv_key)
    print(f"CV scheme: modality.recommended_cv() = {cv_key!r} -> resolved via resolve_cv()")

    expr = np.log1p(x_df)
    # per_drug=True gives each drug its own ridge slope on the PCs, which is
    # what lets the residual carry patient x drug interaction signal at all --
    # the shared-slope default collapses to one residual per patient (zero
    # interaction by construction), matching the "per-drug head" convention
    # scripts/gdsc_representation_increment.py used for the 0.224 reference.
    # seed= has no effect on this CV scheme (GroupKFold is deterministic, not
    # shuffled) -- grouped_cv_predict's seed only matters to a CVScheme that
    # carries its own randomness, e.g. leave_subtype_out(seed=...).
    factory = partial(SimpleProbe, n_components=10, per_drug=True, std_floor=0.5)
    preds = grouped_cv_predict(factory, expr, filtered, cv=cv, seed=SEED)
    rho = interaction_rho(preds)
    print(
        f"interaction_rho (expression PCA, k=10, per-drug head, {cv_key} CV, "
        f"{dropped} rows dropped to SYNTHETIC leakage filter) = {rho:.3f}\n"
    )


def main() -> None:
    _run(Gdsc2Auc(cancer_type_filter=["sarcoma"]), "GDSC2 sarcoma", declared_overlap_lines=3)
    print("Reference (old ad-hoc loading, raw-count log1p, drug_id-keyed): 0.224\n")

    _run(SoragniViability(rna_source="tumor"), "Soragni", declared_overlap_lines=2)
    print("Reference: already-documented null (n=15-17, too small for within-cohort signal)")


if __name__ == "__main__":
    main()
