"""Mechanism-of-action annotation for GDSC compounds.

Selection gap@k is drug-level and mechanism-blind: two representations can post the same
delta-AUC while shortlisting mechanistically different compounds. Joining each drug to its
target pathway lets the audit ask the clinical question -- did the shortlist contain the right
pathway, not the right molecule -- and lets the interaction be split by class, since targeted
agents are line-specific by biology and broad cytotoxics are not.

Source: ``data/raw/gdsc2_sarcoma/gdsc2/screened_compounds_rel_8.5.csv`` (GDSC release 8.5,
621 compounds), columns ``TARGET`` and ``TARGET_PATHWAY``.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def normalize_drug(name: str) -> str:
    """Lowercase and strip every non-alphanumeric character.

    Tahoe, GDSC and sci-Plex spell the same compound differently (``crizotinib`` vs
    ``Crizotinib``, ``AZD-8055`` vs ``AZD8055``), so joins key on this instead of the raw name.
    """
    return _NON_ALNUM.sub("", str(name).lower())


def load_moa(path: Path) -> pd.DataFrame:
    """Load the GDSC screened-compounds table, indexed by normalized drug key.

    Duplicate keys (the same compound screened at more than one site) are deduplicated by
    retaining the row with the alphabetically-first target_pathway value, then by
    alphabetically-first target (tiebreaker). NaN pathways sort last. This ensures
    deterministic output independent of CSV row order. Target pathway annotations vary by
    screening site for some compounds; when conflicts are detected, a warning names the
    affected compounds and the alphabetically-first pathway is retained. Missing target or
    pathway values are preserved as NaN, allowing downstream code to filter them appropriately.
    """
    raw = pd.read_csv(path)

    # Preserve missing values; do not stringify NaN to "nan"
    out = pd.DataFrame(
        {
            "drug_name": raw["DRUG_NAME"],
            "target": raw["TARGET"],
            "target_pathway": raw["TARGET_PATHWAY"],
        }
    )
    out.index = pd.Index(out["drug_name"].map(normalize_drug), name="key")

    # Detect and warn about pathway disagreements for duplicate keys
    dup_keys = out.index[out.index.duplicated(keep=False)].unique()
    disagreements = []
    for key in dup_keys:
        group = out.loc[key]
        if isinstance(group, pd.Series):
            continue  # Single row, not truly a duplicate
        pathways = group["target_pathway"].dropna().unique()
        if len(pathways) > 1:
            disagreements.append(key)

    if disagreements:
        msg = (
            f"load_moa: {len(disagreements)} normalized drug keys have differing "
            f"target_pathway values across screening sites: "
            f"{', '.join(sorted(disagreements))}"
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

    # Deduplicate deterministically: for each key, keep the row with the
    # alphabetically-first target_pathway (NaN last), then alphabetically-first target.
    # This ensures reproducibility regardless of CSV row order.
    out_with_key = out.copy()
    out_with_key["__key__"] = out.index
    sorted_df = out_with_key.sort_values(
        ["__key__", "target_pathway", "target"], na_position="last"
    )
    dedup = sorted_df.drop_duplicates(subset=["__key__"], keep="first")
    dedup.index = pd.Index(dedup["__key__"], name="key")
    return dedup.loc[:, ["drug_name", "target", "target_pathway"]]


def pathway_map(moa: pd.DataFrame, drugs: Iterable[str]) -> dict[str, str]:
    """Map each drug name, as written by the caller, to its target pathway.

    Unmatched drugs (either absent from the table or lacking a pathway annotation) are
    omitted, so a caller counting coverage sees the true join rate. This contract ensures
    the returned dict never maps to NaN or the string "nan".
    """
    lookup = moa["target_pathway"].to_dict()
    pairs = (
        (d, lookup.get(normalize_drug(d)))
        for d in drugs
    )
    # Filter out None and NaN pathways
    return {
        d: pw
        for d, pw in pairs
        if pw is not None and pd.notna(pw)
    }


def moa_hit_rate_at_k(
    preds: pd.DataFrame, pathway: dict[str, str], ks: tuple[int, ...] = (1, 3, 5)
) -> dict[int, float]:
    """Share of lines whose top-k shortlist contains the true-best drug's pathway.

    ``y_pred`` is AUC-like, so shortlists rank ascending. Unlike gap@k this credits a
    mechanistically correct pick even when the compound is wrong, which is the clinical
    question and which collapses me-too compounds. Lines whose observed best drug carries no
    pathway annotation are skipped rather than counted as misses.
    """
    df = preds.copy()
    df["pathway"] = df["drug"].map(lambda d: pathway.get(d))
    best_pw = df.loc[df.groupby("patient")["y_true"].idxmin()].set_index("patient")[
        "pathway"
    ]
    ranked = df.sort_values(["patient", "y_pred"], kind="stable")
    ranked["rank"] = ranked.groupby("patient").cumcount()
    ranked["want"] = ranked["patient"].map(best_pw)
    scored = ranked[ranked["want"].notna()]
    match = scored["pathway"].eq(scored["want"])  # type: ignore[attr-defined]
    return {
        k: float(
            match.where(scored["rank"] < k, other=False)
            .groupby(scored["patient"])
            .any()
            .mean()
        )
        for k in ks
    }
