"""Leakage filtering: strip pretraining-contaminated rows before scoring.

Distinguished from ``Generator.context_coverage`` (``model_protocols.py``), which
looks similar but answers a different question: coverage asks whether a model can
*represent* a perturbation (capability); this module asks whether a model already
*saw the answer* during pretraining (validity). A model can be fully capable of
representing a drug and still have memorized its specific response label.

Filtering happens on the whole panel for a given model, before any CV split -- not
fold-specific. If a line was in a model's own pretraining, it is invalid as a test
case for that model's representation quality regardless of which CV fold it would
otherwise land in; this is a different, earlier-stage contamination than ordinary
train/test CV leakage.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

import pandas as pd

from fmharness.model_protocols import Encoder, Generator
from fmharness.schema import LeakageProfile


@runtime_checkable
class LeakageQueryable(Protocol):
    """Optional: a model that can expose what its pretraining corpus actually covered."""

    def pretraining_lines(self) -> set[str] | None: ...

    def pretraining_drugs(self) -> set[str] | None: ...


def filter_leakage(
    design: pd.DataFrame,
    model: Encoder | Generator,
) -> tuple[pd.DataFrame, LeakageProfile]:
    """Drop pretraining-contaminated (patient, drug) rows from ``design``.

    - Always drops doubly-exposed rows (line AND drug both in pretraining) -- the
      sharpest risk, matching Phase 1's actual exclusion of 6 of 1,568 Tahoe/sci-Plex
      pairs (0.4%).
    - If ``task_signal_in_pretrain == "direct"`` (model trained on actual response
      labels), also drops single-axis overlap (line OR drug).
    - If task signal is "none" or "adjacent", single-axis overlap is reported in the
      profile but not hard-excluded -- a blanket single-axis filter would make a
      model pretrained on a broad public atlas (which shows large nominal line
      overlap with almost any cancer cohort) untestable on nearly anything.
    - If ``model`` does not implement ``LeakageQueryable``: the design is returned
      unmodified and ``basis="unknown"``. Filtering requires measured overlap; it
      never guesses clean.
    """
    version = model.version()
    meta = model.metadata()

    if not isinstance(model, LeakageQueryable):
        return design, LeakageProfile(
            tranche_id="unspecified",
            model_version=version,
            drug_overlap_fraction=0.0,
            generated_at=datetime.now(UTC),
            line_overlap_frac=None,
            doubly_exposed_frac=None,
            basis="unknown",
        )

    lines = model.pretraining_lines()
    drugs = model.pretraining_drugs()
    if lines is None or drugs is None:
        return design, LeakageProfile(
            tranche_id="unspecified",
            model_version=version,
            drug_overlap_fraction=0.0,
            generated_at=datetime.now(UTC),
            line_overlap_frac=None,
            doubly_exposed_frac=None,
            basis="unknown",
        )

    line_hit = design["patient"].isin(list(lines))
    drug_hit = design["drug"].isin(list(drugs))
    doubly_exposed = line_hit & drug_hit
    single_axis = line_hit | drug_hit

    n = len(design)
    line_overlap_frac = float(line_hit.mean()) if n else 0.0
    drug_overlap_frac = float(drug_hit.mean()) if n else 0.0
    doubly_exposed_frac = float(doubly_exposed.mean()) if n else 0.0

    drop_mask = doubly_exposed
    if meta.task_signal_in_pretrain == "direct":
        drop_mask = drop_mask | single_axis

    filtered = design.loc[~drop_mask].reset_index(drop=True)
    profile = LeakageProfile(
        tranche_id="unspecified",
        model_version=version,
        drug_overlap_fraction=drug_overlap_frac,
        generated_at=datetime.now(UTC),
        line_overlap_frac=line_overlap_frac,
        doubly_exposed_frac=doubly_exposed_frac,
        basis="measured",
    )
    return filtered, profile


def corpus_declared_partially(corpus_lines: str | None, corpus_drugs: str | None) -> bool:
    """True iff exactly one of --corpus-lines/--corpus-drugs was given, not both, not neither.

    filter_leakage only filters when it has a measured declaration on BOTH axes; a
    half-declared corpus silently scores identically to an unfiltered run (basis="unknown"),
    with no signal in the output that the declared corpus was ignored. A driver's main()
    should reject this combination up front rather than letting it through.
    """
    return (corpus_lines is None) != (corpus_drugs is None)


def ground_truth_source_declared_ambiguously(
    context: str | None, deltas_bundle: str | None
) -> bool:
    """True iff --context/--deltas-bundle are both given or both omitted.

    Exactly one must select the ground-truth (real_delta, real_key, base) source: --context
    rebuilds it live from a single-cell AnnData, --deltas-bundle reads a precomputed pseudobulk
    parquet bundle. The two are not interchangeable in practice -- a live --context rebuild is
    whatever Tahoe context snapshot currently sits on Alpine, not necessarily the one a
    published table was computed from -- so a driver's main() should reject an ambiguous
    combination rather than silently picking one.
    """
    return (context is None) == (deltas_bundle is None)


def parse_corpus_set(raw: str | None) -> set[str] | None:
    """Parse a comma-separated --corpus-lines/--corpus-drugs value into a set.

    The documented workflow has a human copy-paste a comma-separated list printed by an
    earlier step into these flags -- strip whitespace around each entry and drop empty
    entries, so a stray space after a comma ("A, B, C") does not silently produce a corpus
    entry (" B") that can never match a real line/drug name and weakens the leakage filter
    with no error. ``None`` (flag not given) stays ``None``.
    """
    if raw is None:
        return None
    return {s.strip() for s in raw.split(",") if s.strip()}
