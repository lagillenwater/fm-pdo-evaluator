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

    Duplicate keys (the same compound screened at more than one site) collapse to the first
    row; the target annotation does not vary by site.
    """
    raw = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "drug_name": raw["DRUG_NAME"].astype(str),
            "target": raw["TARGET"].astype(str),
            "target_pathway": raw["TARGET_PATHWAY"].astype(str),
        }
    )
    out.index = pd.Index(out["drug_name"].map(normalize_drug), name="key")
    return out.loc[~out.index.duplicated(keep="first")]


def pathway_map(moa: pd.DataFrame, drugs: Iterable[str]) -> dict[str, str]:
    """Map each drug name, as written by the caller, to its target pathway.

    Unmatched drugs are omitted rather than mapped to a sentinel, so a caller counting
    coverage sees the true join rate.
    """
    lookup = moa["target_pathway"].to_dict()
    pairs = ((d, lookup.get(normalize_drug(d))) for d in drugs)
    return {d: pw for d, pw in pairs if pw is not None}
