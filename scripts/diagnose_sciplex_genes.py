"""Why does a 3-line human drug screen report 110,983 genes, including mouse RIKEN clones?

The panel audit read sci-Plex's gene axis as 110,983 symbols whose first entries are
0610005C13Rik, 0610006L08Rik -- RIKEN mouse cDNA clone names. sci-Plex 3 (Srivatsan et al.
2020) screens A549, K562 and MCF7, all human, and the human annotation carries roughly 60k
features. So either the file is built on a combined human+mouse reference, or the audit picked
the wrong var column. Both would corrupt the common panel, and in a way that LOOKS fine: a
contaminated axis still intersects, just wrongly, and the resulting panel would be defended by
a number nobody could trace back to a species.

Prints the full var schema with examples so the axis can be chosen on evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def decode(arr) -> list[str]:
    """HDF5 strings arrive as bytes."""
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def read_col(var, name: str) -> list[str]:
    """One var column, resolving h5ad categorical encoding."""
    node = var[name]
    if isinstance(node, h5py.Group):
        cats = decode(node["categories"][:])
        return [cats[c] if c >= 0 else "" for c in node["codes"][:]]
    return decode(node[:])


def species_split(genes: list[str]) -> dict[str, int]:
    """Split an axis by the casing conventions that separate human from mouse symbols.

    HGNC human symbols are all-caps (TP53, A1BG). MGI mouse symbols are title-case (Trp53),
    and RIKEN clones end in Rik. Counting them apart says whether an axis is one species.
    """
    human = sum(1 for g in genes if g.isupper() and not g.startswith("ENS"))
    rik = sum(1 for g in genes if g.endswith("Rik"))
    mouse = sum(1 for g in genes if g[:1].isupper() and g[1:].islower() and not g.isupper())
    ens = sum(1 for g in genes if g.upper().startswith("ENS"))
    return {"HUMAN_upper": human, "mouse_title": mouse, "RIKEN_Rik": rik, "ensembl": ens}


def main() -> None:
    """Dump every var column with its species composition."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5ad", required=True, type=Path)
    args = ap.parse_args()

    with h5py.File(args.h5ad, "r") as f:
        var = f["var"]
        idx_key = var.attrs.get("_index", "_index")
        idx_key = idx_key.decode() if isinstance(idx_key, bytes) else str(idx_key)
        print(f"var columns: {sorted(var.keys())}")
        print(f"var _index  : {idx_key}\n")
        for name in [idx_key] + sorted(k for k in var.keys() if k != idx_key):
            try:
                vals = read_col(var, name)
            except Exception as exc:  # a column that will not read is itself the answer
                print(f"{name:<24} UNREADABLE: {exc}")
                continue
            uniq = sorted(set(vals))
            comp = species_split(uniq)
            print(f"{name:<24} n={len(vals):>7} unique={len(uniq):>7}  {comp}")
            print(f"{'':24} first: {uniq[:4]}")
        n_obs_genes = f["X"].attrs.get("shape") if "X" in f else None
        print(f"\nX shape attr: {n_obs_genes}")


if __name__ == "__main__":
    main()
