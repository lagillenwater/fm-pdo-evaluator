"""brd_to_cid must map L1000's Broad pert_id to Tahoe's own CID string format.

The rest of scripts/score_l1000_context_generation.py is exercised end to end against a
synthetic fixture during development (not checked in: needs anndata files); this pins the one
piece that is pure logic and load-bearing -- get the CID format wrong and every file silently
fails to match in build_generated_deltas (which prints "skip <file>: no pert_id match" and
continues, so a mapping bug would look like "no drugs matched" rather than crash).
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from score_l1000_context_generation import _ncid, brd_to_cid  # noqa: E402


def test_ncid_normalizes_float_like_and_int_like_forms() -> None:
    assert _ncid(444795.0) == "444795"
    assert _ncid("444795.0") == "444795"
    assert _ncid("444795") == "444795"
    assert _ncid(float("nan")) == ""
    assert _ncid("not-a-number") == ""


def test_brd_to_cid_matches_tahoes_own_clean_integer_cid_format(tmp_path: Path) -> None:
    pert_info = pd.DataFrame({
        "pert_id": ["BRD-K001", "BRD-K002"],
        "pubchem_cid": [444795.0, 6918289.0],  # L1000's own format: float
        "pert_type": ["trt_cp", "trt_cp"],
    })
    with gzip.open(tmp_path / "GSE92742_Broad_LINCS_pert_info.txt.gz", "wt") as f:
        pert_info.to_csv(f, sep="\t", index=False)

    m = brd_to_cid(tmp_path)
    # This is the exact clean-integer format context_by_drug/pert_to_cid.tsv already ships
    # (verified against the real file on Alpine: "Retinoic acid\t444795"), which
    # build_generated_deltas' Tahoe-context sources are already keyed by -- the L1000-context
    # map must match it, or the (patient, drug) join in delta_fidelity silently returns nothing.
    assert m == {"BRD-K001": "444795", "BRD-K002": "6918289"}
