"""Measure the gene axis of every dataset in the evaluation, and the panel they share.

Nothing controlled gene-set size before this. Each source was scored on whatever genes it
happened to carry -- Tahoe's full transcriptome, the Stack generation gene list, or the
top-2000-HVG-union-Hallmark panel ``learned_gene_panel`` built for pca/nmf -- and
``de_fidelity`` silently drops genes a source lacks. So every source was scored on its own
universe, which is the gene-axis twin of the (patient, drug) support bug
``restrict_common_support`` exists to fix, and it hits ``pr_auc`` hardest: average precision
depends on the positive rate, and a 5%-of-transcriptome high-variance panel is enriched for
genes that actually move.

Fixing that needs one number nobody had measured: the largest gene set EVERY dataset can
supply. This measures it, reads only each file's gene axis (parquet schema, HDF5 metadata) so
it costs seconds rather than loading matrices, and writes the panel plus a sidecar recording
every resolved path, content hash, and the git sha -- so the panel can be regenerated and
checked rather than trusted.

Namespaces are NOT assumed to match. A symbol/Ensembl mismatch would silently produce an empty
or tiny intersection that looks like a scientific finding, so each axis's namespace is detected
and reported, and Ensembl IDs are version-stripped before comparison.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def _decode(arr) -> list[str]:
    """HDF5 string arrays come back as bytes; h5ad categoricals as codes."""
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def namespace_of(genes: list[str]) -> str:
    """Which identifier namespace an axis is in, by majority of the first 200 entries."""
    head = genes[:200]
    if not head:
        return "empty"
    ens = sum(1 for g in head if g.upper().startswith("ENSG"))
    if ens > len(head) / 2:
        return "ensembl"
    num = sum(1 for g in head if g.isdigit())
    if num > len(head) / 2:
        return "entrez"
    return "symbol"


def strip_version(genes: list[str]) -> list[str]:
    """ENSG00000141510.16 and ENSG00000141510 are the same gene; only one joins."""
    return [g.split(".", 1)[0] if g.upper().startswith("ENSG") else g for g in genes]


def axis_parquet(path: Path) -> list[str]:
    """Column names from the parquet schema -- metadata only, no row groups read."""
    import pyarrow.parquet as pq

    return [c for c in pq.ParquetFile(path).schema.names if c not in ("__index_level_0__",)]


def axis_h5ad(path: Path) -> list[str]:
    """var_names from an .h5ad, preferring a symbol column when the index is Ensembl.

    Both flavors appear here: scPerturb's SrivatsanTrapnell2020_sciplex3 and chemCPA's
    subset store their gene names differently, and the index is not reliably the symbol.
    """
    import h5py

    with h5py.File(path, "r") as f:
        var = f["var"]
        idx_key = var.attrs.get("_index", "_index")
        idx_key = idx_key.decode() if isinstance(idx_key, bytes) else str(idx_key)
        index = _decode(var[idx_key][:])
        if namespace_of(index) == "symbol":
            return index
        for cand in ("gene_short_name", "gene_symbol", "symbol", "gene_name", "feature_name"):
            if cand in var:
                node = var[cand]
                if isinstance(node, h5py.Group):  # categorical
                    cats = _decode(node["categories"][:])
                    codes = node["codes"][:]
                    return [cats[c] if c >= 0 else "" for c in codes]
                vals = _decode(node[:])
                if namespace_of(vals) == "symbol":
                    return vals
        return index


def axis_gctx(path: Path) -> list[str]:
    """Row ids from a .gctx (HDF5). These are Entrez ids for the LINCS matrices."""
    import h5py

    with h5py.File(path, "r") as f:
        return _decode(f["0/META/ROW/id"][:])


def axis_gene_info(path: Path) -> list[str]:
    """Symbols from the LINCS gene_info table, so the gctx's Entrez ids can be named."""
    with gzip.open(path, "rt") as fh:
        gi = pd.read_csv(fh, sep="\t")
    return [str(s) for s in gi["pr_gene_symbol"]]


def cohort_h5ad(path: Path) -> dict:
    """Cell-line and drug cardinality of an .h5ad, read from obs metadata only.

    Gene count alone cannot decide between candidate datasets. The interaction metric this
    harness reports lives on the CELL-LINE axis -- it asks whether a prediction is matched to
    the right line, holding drug fixed -- so a dataset with a full transcriptome and three
    cell lines may be strictly worse for it than a narrow one with many. Measuring both keeps
    that trade-off explicit instead of letting a gene count decide it silently.
    """
    import h5py

    def col(obs, names):
        for n in names:
            if n in obs:
                node = obs[n]
                if isinstance(node, h5py.Group):
                    return set(_decode(node["categories"][:]))
                return set(_decode(node[:]))
        return None

    with h5py.File(path, "r") as f:
        obs = f["obs"]
        return {
            "n_cells": int(obs[list(obs.keys())[0]].shape[0]) if obs.keys() else 0,
            "obs_columns": sorted(k for k in obs.keys() if not k.startswith("_")),
            "cell_lines": sorted(col(obs, ("cell_line", "cell_type", "cell_id")) or []),
            "n_drugs": len(col(obs, ("perturbation", "condition", "product_name", "pert_iname")) or []),
        }


def cohort_l1000(inst_info: Path) -> dict:
    """Cell-line and perturbagen cardinality of the LINCS instance table."""
    with gzip.open(inst_info, "rt") as fh:
        inst = pd.read_csv(fh, sep="\t", low_memory=False)
    return {
        "n_wells": int(len(inst)),
        "cell_lines": sorted({str(c) for c in inst["cell_id"].unique()}),
        "n_drugs": int(inst["pert_iname"].nunique()),
    }


READERS = {"parquet": axis_parquet, "h5ad": axis_h5ad, "gctx": axis_gctx, "gene_info": axis_gene_info}


def sha256(path: Path, cap: int = 64 << 20) -> str:
    """Hash of up to the first `cap` bytes -- identity for files too large to hash whole."""
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as fh:
        while n < cap and (b := fh.read(1 << 20)):
            h.update(b)
            n += len(b)
    return f"sha256:{h.hexdigest()}" + ("" if n < cap else f" (first {cap} bytes)")


def git_sha() -> str:
    """The commit this panel was measured at."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    """Measure each axis, report the lattice, and write the panel with its provenance."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        action="append",
        required=True,
        help="label=kind=path, kind in {parquet,h5ad,gctx,gene_info}. Repeatable.",
    )
    ap.add_argument(
        "--require",
        default="",
        help="comma-separated labels that MUST be in the intersection. Others are measured and "
        "reported but do not constrain the panel -- that is how a dataset's cost is priced "
        "before deciding whether to pay it.",
    )
    ap.add_argument(
        "--cohort",
        action="append",
        default=[],
        help="label=kind=path for a cohort report (kind in {h5ad,l1000}). Gene count alone "
        "cannot decide between datasets; the interaction metric lives on the cell-line axis.",
    )
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    axes: dict[str, list[str]] = {}
    meta: dict[str, dict] = {}
    for spec in args.source:
        label, kind, path = spec.split("=", 2)
        p = Path(path)
        if not p.exists():
            print(f"  SKIP {label}: {p} not present")
            continue
        genes = READERS[kind](p)
        ns = namespace_of(genes)
        uniq = sorted(set(strip_version(genes)) - {"", "nan", "None"})
        axes[label] = uniq
        meta[label] = {
            "path": str(p),
            "kind": kind,
            "namespace": ns,
            "n_raw": len(genes),
            "n_unique": len(uniq),
            "duplicates": len(genes) - len(set(genes)),
            "examples": uniq[:5],
            "sha256": sha256(p),
        }
        print(f"  {label:<28} {len(uniq):>7} unique  ({ns:<7}) {uniq[:3]}")

    cohorts: dict[str, dict] = {}
    for spec in args.cohort:
        label, kind, path = spec.split("=", 2)
        p = Path(path)
        if not p.exists():
            print(f"  SKIP cohort {label}: {p} not present")
            continue
        c = cohort_h5ad(p) if kind == "h5ad" else cohort_l1000(p)
        cohorts[label] = c
        lines = c.get("cell_lines", [])
        print(f"\ncohort {label}: {len(lines)} cell lines, {c.get('n_drugs')} drugs")
        print(f"  lines: {lines if len(lines) <= 12 else lines[:12] + ['...']}")

    ns_set = {m["namespace"] for m in meta.values()}
    print(f"\nnamespaces present: {sorted(ns_set)}")
    if len(ns_set) > 1:
        print("  WARNING: axes are in different namespaces. An intersection across them is")
        print("  meaningless until they are mapped. Counts below are per-namespace.")

    print("\npairwise intersection (rows n cols):")
    labels = sorted(axes)
    tab = pd.DataFrame(
        [[len(set(axes[a]) & set(axes[b])) for b in labels] for a in labels],
        index=labels,
        columns=labels,
    )
    print(tab.to_string())

    required = [x.strip() for x in args.require.split(",") if x.strip()]
    missing_req = [r for r in required if r not in axes]
    if missing_req:
        raise SystemExit(f"--require names labels that were not measured: {missing_req}")
    if not required:
        required = labels

    panel: set[str] | None = None
    print(f"\nbuilding panel from required = {required}")
    for label in required:
        before = len(panel) if panel is not None else None
        panel = set(axes[label]) if panel is None else (panel & set(axes[label]))
        print(f"  n {label:<28} {before if before is not None else '-':>7} -> {len(panel)}")

    assert panel is not None
    final = sorted(panel)
    print(f"\nCOMMON PANEL = {len(final)} genes")
    for label in labels:
        cost = len(set(axes[label]) - panel)
        print(f"  {label:<28} contributes {len(axes[label]):>7}, loses {cost:>7} to the panel")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dest = args.out_dir / "common_gene_panel.txt"
    dest.write_text("\n".join(final) + "\n")
    (args.out_dir / "common_gene_panel.params.json").write_text(
        json.dumps(
            {
                "result": dest.name,
                "git_sha": git_sha(),
                "n_panel": len(final),
                "required": required,
                "measured": meta,
                "cohorts": cohorts,
                "pairwise_intersection": tab.to_dict(),
                "panel_sha256": "sha256:" + hashlib.sha256(dest.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {dest} and its sidecar")


if __name__ == "__main__":
    main()
