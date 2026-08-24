"""Audit the check-2 shortlists: is gap@k in raw AUC a valid selection metric?

Three tables, from the per-pair dump written by ``score_generation_eval.py``:

1. The potency prior (rank drugs by training-fold mean AUC, ignore the cell line) scored with
   the same gap@k on the same folds as every representation. If the models do not beat it,
   their shortlists carry no cell-line information at all.
2. Shortlist concentration -- distinct top-1 picks, modal share, share of picks that are
   broadly active -- against the observed reference. The truth is itself concentrated, so a
   model is only collapsed if it is more concentrated than that row.
3. The same gap@k in within-drug percentile space, where a pan-cytotoxic compound carries no
   advantage.

The ``drug`` column in ``check2_preds.parquet`` holds PubChem CIDs (Tahoe's native drug key),
not names, so the MOA join goes CID -> name (``data/static/tahoe_pert_to_cid.tsv``) -> pathway
(``pathway_map``), and the resulting pathway dict is re-keyed back onto the CID so it lines up
with ``preds["drug"]``, which is what ``moa_hit_rate_at_k`` and ``interaction_by_moa_class``
index.

Usage:
  uv run python scripts/check2_selection_audit.py
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from fmharness.evaluation import regret_norm_at_k
from fmharness.moa import (
    interaction_by_moa_class,
    load_moa,
    moa_hit_rate_at_k,
    pathway_map,
    shuffled_hit_rate,
)
from fmharness.selection import shortlist_concentration, within_drug_percentile

KS = (1, 3, 5)


def _gap_row(frame: pd.DataFrame, pred_col: str, prefix: str = "gap@") -> dict[str, float]:
    # Assign rather than rename: renaming y_true -> y_pred for the prior/observed rows would
    # destroy the y_true column that regret_norm_at_k needs on the other axis.
    gaps = regret_norm_at_k(frame.assign(y_pred=frame[pred_col]), ks=KS)
    return {f"{prefix}{k}": round(gaps[k], 3) for k in KS}


def _cid_to_pathway(moa: pd.DataFrame, cid_map: Path, cids: Iterable[str]) -> dict[str, str]:
    """CID -> target pathway, via CID -> drug name -> pathway.

    ``check2_preds.parquet``'s ``drug`` column is PubChem CIDs (strings); ``pathway_map`` keys
    on drug names. ``tahoe_pert_to_cid.tsv`` has two rows mapping to CID 11707110 (``Trametinib``
    and ``Trametinib (DMSO_TF solvate)``); the plain name is the later row, so a last-one-wins
    dict keeps it -- and it is the name present in the GDSC compound table, so no special-casing
    of the solvate suffix is needed.
    """
    tsv = pd.read_csv(cid_map, sep="\t", header=None, names=["name", "cid"])
    tsv["cid"] = tsv["cid"].astype(str)
    cid_to_name = dict(zip(tsv["cid"], tsv["name"], strict=True))

    names_by_cid = {cid: cid_to_name[cid] for cid in cids if cid in cid_to_name}
    name_pathway = pathway_map(moa, sorted(set(names_by_cid.values())))
    return {cid: name_pathway[name] for cid, name in names_by_cid.items() if name in name_pathway}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", default="results/check2_preds.parquet")
    ap.add_argument(
        "--compounds",
        default="data/raw/gdsc2_sarcoma/gdsc2/screened_compounds_rel_8.5.csv",
    )
    ap.add_argument("--cid-map", default="data/static/tahoe_pert_to_cid.tsv")
    ap.add_argument("--out", default="results/check2_selection_audit.csv")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    preds = pd.read_parquet(repo / args.preds)

    moa = load_moa(repo / args.compounds)
    pathway = _cid_to_pathway(moa, repo / args.cid_map, preds["drug"].unique())
    print(f"MOA join: {len(pathway)}/{preds['drug'].nunique()} drugs annotated\n")

    def _row(name: str, method: str, frame: pd.DataFrame, col: str) -> dict[str, object]:
        scored = frame.assign(y_pred=frame[col])
        conc = shortlist_concentration(frame, pred_col=col)
        pct = within_drug_percentile(frame, cols=("y_true", col))
        moa_hits = moa_hit_rate_at_k(scored, pathway, KS)
        strat = interaction_by_moa_class(scored, pathway)
        return {
            "source": name,
            "method": method,
            **_gap_row(frame, col),
            **_gap_row(pct, col, prefix="pct_gap@"),
            **{f"moa@{k}": round(v, 3) for k, v in moa_hits.items()},
            "int_targeted": round(strat["targeted"], 3),
            "int_cytotoxic": round(strat["cytotoxic"], 3),
            "distinct": conc["distinct"],
            "modal_share": round(float(conc["modal_share"]), 3),
            "broadly_active_share": round(float(conc["broadly_active_share"]), 3),
        }

    rows = [
        _row(str(source), str(method), frame, "y_pred")
        for (source, method), frame in preds.groupby(["source", "method"], sort=True)
    ]

    # The prior and the observed reference: computed once, on any single (source, method)
    # slice, since y_prior and y_true do not vary with the representation.
    first = preds.iloc[0]
    ref = preds[(preds["source"] == first["source"]) & (preds["method"] == first["method"])]
    rows.append(_row("potency_prior", "-", ref, "y_prior"))
    rows.append(_row("observed", "-", ref, "y_true"))

    # The pan-active base rate: what a random shortlist already scores. Any moa@k at or below
    # this is saturation, not skill.
    base = shuffled_hit_rate(ref.assign(y_pred=ref["y_true"]), pathway, KS)
    print("shuffled-shortlist base rate: " + "  ".join(f"moa@{k}={base[k]:.3f}" for k in KS) + "\n")

    table = pd.DataFrame(rows)
    dest = repo / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest, index=False)
    print(table.to_string(index=False))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
