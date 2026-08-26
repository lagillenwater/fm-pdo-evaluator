"""Gene overlap AND cell-line overlap across every dataset in the evaluation, in one place.

Two axes decide whether a dataset can stand in for another, and reporting either alone is
misleading. sci-Plex carries a full transcriptome and three cell lines; L1000 carries 76 cell
lines and 978 measured genes. A gene-count table would say sci-Plex wins, a cell-line table
would say L1000 wins, and the harness needs both because its interaction metric lives on the
cell-line axis while its DE metrics live on the gene axis.

Three normalisations, each because the raw axes do not join:

  * sci-Plex is built on a COMBINED human+mouse reference -- 110,983 features, of which 45,752
    are title-case mouse symbols and 2,639 are RIKEN clones. Restricting on the ENSG prefix of
    ensembl_id is the only reliable cut; casing is not, since some mouse symbols are uppercase.
  * L1000's .gctx rows are Entrez ids and everything else is symbols, so gene_info supplies the
    mapping. Intersecting the raw axes yields exactly 0 and would look like a finding.
  * Cell lines are written inconsistently ("NCI-H1299" / "NCIH1299" / "ACH-000012"). Names are
    normalised to uppercase-alphanumeric, and DepMap ACH ids are reported separately rather
    than silently failing to match a name.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
from pathlib import Path

import pandas as pd


def load_ach_map(model_csv: Path) -> dict[str, str]:
    """DepMap ACH id -> normalised stripped cell-line name.

    Without this the cell-line matrix is nonsense in a way that reads as a result. Tahoe,
    Stack and GDSC2 key on ACH ids; L1000 and sci-Plex key on names; the organoids key on
    SARC ids. Intersecting them raw yields zeros that look like "these cohorts share no cell
    lines" when they only mean "these cohorts spell cell lines differently". Everything is
    mapped to the name namespace before any intersection is taken.
    """
    if not model_csv.exists():
        return {}
    m = pd.read_csv(model_csv, low_memory=False)
    name_col = next(
        (c for c in ("StrippedCellLineName", "CellLineName", "cell_line_name") if c in m.columns),
        None,
    )
    if name_col is None or "ModelID" not in m.columns:
        return {}
    out: dict[str, str] = {}
    for mid, nm in zip(m["ModelID"], m[name_col], strict=True):
        if isinstance(nm, str) and nm:
            out[norm_line(mid)] = norm_line(nm)
    return out


def norm_line(s: object) -> str:
    """Uppercase-alphanumeric, matching fmharness.deltas._norm so joins agree with the harness."""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def decode(arr) -> list[str]:
    """HDF5 strings arrive as bytes."""
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def h5_col(grp, name: str) -> list[str] | None:
    """One h5ad column, resolving categorical encoding; None when absent."""
    import h5py

    if name not in grp:
        return None
    node = grp[name]
    if isinstance(node, h5py.Group):
        cats = decode(node["categories"][:])
        return [cats[c] if c >= 0 else "" for c in node["codes"][:]]
    return decode(node[:])


LINE_COLS = ("cell_line", "cell_type", "cell_id", "stripped_cell_line_name", "patient", "sample_id")
SYMBOL_COLS = ("gene_symbol", "gene_short_name", "symbol", "gene_name", "feature_name")


def axes_h5ad(path: Path, *, human_only: bool = False) -> tuple[list[str], list[str]]:
    """(genes, cell lines) from an .h5ad, reading only the var/obs metadata.

    ``human_only`` keeps features whose ensembl_id is ENSG -- the sci-Plex case, where a
    combined human+mouse reference otherwise contributes 45,752 mouse features to a gene panel.
    """
    import h5py

    with h5py.File(path, "r") as f:
        var, obs = f["var"], f["obs"]
        vidx = var.attrs.get("_index", "_index")
        vidx = vidx.decode() if isinstance(vidx, bytes) else str(vidx)
        genes = h5_col(var, vidx) or []
        if not genes or genes[0].upper().startswith("ENS"):
            for c in SYMBOL_COLS:
                alt = h5_col(var, c)
                if alt and not alt[0].upper().startswith("ENS"):
                    genes = alt
                    break
        if human_only:
            ens = h5_col(var, "ensembl_id")
            if ens is None:
                raise SystemExit(f"{path}: --human-only needs an ensembl_id column")
            genes = [g for g, e in zip(genes, ens, strict=True) if e.upper().startswith("ENSG")]

        oidx = obs.attrs.get("_index", "_index")
        oidx = oidx.decode() if isinstance(oidx, bytes) else str(oidx)
        lines: list[str] = []
        for c in LINE_COLS:
            got = h5_col(obs, c)
            if got:
                lines = got
                break
        if not lines:
            lines = h5_col(obs, oidx) or []
    return genes, lines


def axes_parquet(path: Path, key_path: Path | None) -> tuple[list[str], list[str]]:
    """(gene columns, patient ids) for a delta bundle; the key is a separate small file."""
    import pyarrow.parquet as pq

    genes = [c for c in pq.ParquetFile(path).schema.names if c != "__index_level_0__"]
    lines: list[str] = []
    if key_path and key_path.exists():
        k = pd.read_parquet(key_path)
        col = "patient" if "patient" in k.columns else k.columns[0]
        lines = [str(x) for x in k[col]]
    return genes, lines


def axes_l1000(gene_info: Path, inst_info: Path) -> tuple[list[str], list[str]]:
    """(symbols from gene_info, cell ids from inst_info). The .gctx rows are Entrez, not symbols."""
    with gzip.open(gene_info, "rt") as fh:
        gi = pd.read_csv(fh, sep="\t")
    with gzip.open(inst_info, "rt") as fh:
        inst = pd.read_csv(fh, sep="\t", low_memory=False)
    return [str(s) for s in gi["pr_gene_symbol"]], [str(c) for c in inst["cell_id"].unique()]


def axes_sarcoma_counts(path: Path) -> tuple[list[str], list[str]]:
    """(genes, samples) from the Soragni normalized-count table: genes index, samples columns."""
    df = pd.read_parquet(path)
    gene_col = next((c for c in df.columns if str(c).lower() in ("gene", "gene_id", "symbol")), None)
    if gene_col is not None:
        genes = [str(g) for g in df[gene_col]]
        lines = [str(c) for c in df.columns if c != gene_col]
    else:
        genes = [str(g) for g in df.index]
        lines = [str(c) for c in df.columns]
    return genes, lines


def matrix(sets: dict[str, set[str]], labels: list[str]) -> pd.DataFrame:
    """Symmetric pairwise-intersection counts; the diagonal is each set's own size."""
    return pd.DataFrame(
        [[len(sets[a] & sets[b]) for b in labels] for a in labels], index=labels, columns=labels
    )


def git_sha() -> str:
    """The commit these overlaps were measured at."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    """Measure both axes for every dataset and write the two matrices."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    S = args.scratch

    specs: dict[str, dict] = {
        "tahoe": {"kind": "parquet", "path": "tahoe_deltas/real_delta.parquet",
                  "key": "tahoe_deltas/real_key.parquet"},
        "stack_cytokine": {"kind": "h5ad", "path": "generated_agg/5-Azacytidine.h5ad"},
        "stack_drug_aligned": {"kind": "h5ad", "path": "generated_drug_aligned_agg/5-Azacytidine.h5ad"},
        "sciplex": {"kind": "h5ad", "path": f"{S}/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad",
                    "human_only": True},
        "l1000": {"kind": "l1000", "path": "GSE92742_Broad_LINCS_gene_info.txt.gz",
                  "inst": "GSE92742_Broad_LINCS_inst_info.txt.gz"},
        "gdsc2_all": {"kind": "h5ad", "path": "data/reference/stack_input_gdscv2.h5ad"},
        "gdsc2_sarcoma": {"kind": "h5ad", "path": "data/reference/stack_input_gdscv2_sarcoma.h5ad"},
        "sarcoma_organoids": {"kind": "h5ad", "path": "data/reference/stack_input_sarcoma.h5ad"},
    }

    ach_map = load_ach_map(Path("data/raw/gdsc2_sarcoma/depmap/Model.csv"))
    print(f"ACH -> name map: {len(ach_map)} DepMap models\n")

    genes: dict[str, set[str]] = {}
    lines: dict[str, set[str]] = {}
    meta: dict[str, dict] = {}
    for label, sp in specs.items():
        p = Path(sp["path"])
        if not p.exists():
            print(f"  SKIP {label}: {p} not present")
            continue
        if sp["kind"] == "h5ad":
            g, ln = axes_h5ad(p, human_only=sp.get("human_only", False))
        elif sp["kind"] == "parquet":
            g, ln = axes_parquet(p, Path(sp["key"]) if sp.get("key") else None)
        elif sp["kind"] == "l1000":
            g, ln = axes_l1000(p, Path(sp["inst"]))
        else:
            g, ln = axes_sarcoma_counts(p)
        gs = {x.split(".", 1)[0] if x.upper().startswith("ENSG") else x for x in g} - {"", "nan"}
        raw_ls = {norm_line(x) for x in ln} - {""}
        n_ach = sum(1 for x in raw_ls if x.startswith("ACH"))
        ls = {ach_map.get(x, x) for x in raw_ls}
        unmapped = sorted(x for x in raw_ls if x.startswith("ACH") and x not in ach_map)
        genes[label], lines[label] = gs, ls
        namespace = "ACH->name" if n_ach else ("SARC" if any(x.startswith("SARC") for x in raw_ls) else "name")
        meta[label] = {"path": str(p), "n_genes": len(gs), "n_lines": len(ls),
                       "line_namespace": namespace, "n_ach_ids": n_ach,
                       "n_ach_unmapped": len(unmapped),
                       "gene_examples": sorted(gs)[:4], "line_examples": sorted(ls)[:6]}
        note = f"  [{n_ach} ACH mapped, {len(unmapped)} unmapped]" if n_ach else ""
        print(f"  {label:<24} {len(gs):>7} genes  {len(ls):>5} lines  {sorted(ls)[:4]}{note}")

    labels = list(genes)
    gm, lm = matrix(genes, labels), matrix(lines, labels)
    print("\n================ GENE OVERLAP ================")
    print(gm.to_string())
    print("\n============ CELL LINE / SAMPLE OVERLAP ============")
    print(lm.to_string())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gm.to_csv(args.out_dir / "gene_overlap.csv")
    lm.to_csv(args.out_dir / "cell_line_overlap.csv")
    (args.out_dir / "dataset_overlap.params.json").write_text(
        json.dumps({"git_sha": git_sha(), "datasets": meta}, indent=2) + "\n"
    )
    print(f"\nwrote gene_overlap.csv and cell_line_overlap.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
