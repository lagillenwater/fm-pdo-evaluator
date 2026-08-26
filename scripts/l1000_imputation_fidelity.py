"""Do L1000's imputed genes carry real delta signal, or only the 978 landmarks' shadow?

The L1000 matrix we use is INF_mlr12k: 978 measured landmark genes plus ~11,350 imputed from
them by multiple linear regression. Requiring L1000 in a common gene panel costs thousands of
genes (12,597 -> 8,600 for Path B), and that cost is only worth paying if the genes bought are
real measurements. The published accuracy figures cannot settle it: they benchmark ABSOLUTE
expression on GTEx-LINCS tissue samples, while this harness scores treated-minus-control
DELTAS -- small differences of large numbers, where imputation error is proportionally far
larger. So the fidelity of imputed genes ON DELTAS is unmeasured, and that is what this
measures, in our own data.

Design: L1000 and Tahoe share 7 cell lines and 14 drugs by PubChem CID, so the same
(line, drug) perturbation exists on both platforms -- one imputing most of its transcriptome,
one measuring all of it. For each shared pair, correlate the L1000 delta against the Tahoe
delta separately over three gene classes taken from gene_info:

  landmark      pr_is_lm=1                 directly measured on the L1000 platform
  bing          pr_is_bing=1, pr_is_lm=0   the Broad's own "best inferred" subset
  other         pr_is_bing=0               the remaining inferred genes

Comparing classes WITHIN the same pairs is what makes this work: platform differences, dose,
timepoint and cell-line effects hit every gene class in a pair equally, so they cannot explain
a gap between classes.

Two controls, because the naive comparison has two ways to be wrong:

  VARIANCE MATCHING. Landmark genes were not chosen at random -- they were selected to be
  informative and highly expressed, so they vary more, and correlation rises with dynamic
  range. A landmark-vs-imputed gap could therefore be a gene-selection artifact rather than an
  imputation one. Imputed genes are resampled to match the landmarks' Tahoe-side variance
  decile profile, and the matched comparison is the one to trust.

  PERMUTATION NULL. Correlating deltas from two platforms gives a nonzero floor from shared
  gene-level structure alone. Pairing each L1000 delta with a DIFFERENT pair's Tahoe delta
  gives that floor per gene class, so "imputed genes correlate at 0.2" can be read against
  what mismatched pairs already produce.

Note what a positive result would and would not mean. Imputed genes are a deterministic
function of the landmarks, so they SHOULD correlate somewhat whenever the landmarks do -- a
linear function of a partly-correct signal is partly correct. Correlation above null is
therefore expected and is not evidence the imputed genes add information. The question this
answers is how much fidelity is retained; the stronger question -- whether they add anything
beyond the landmarks -- is flagged in the writeup as the follow-up.
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
from scipy import stats

from fmharness.statistics import bootstrap_aggregate_pvalue

GENE_CLASSES = ("landmark", "bing", "other")


def norm_name(s: object) -> str:
    """Lowercase alphanumeric, for joining drug names across cohorts."""
    return "".join(c for c in str(s).lower() if c.isalnum())


def norm_line(s: object) -> str:
    """Uppercase alphanumeric, matching fmharness.deltas._norm."""
    import re

    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def gene_class_map(gene_info: Path) -> dict[str, str]:
    """symbol -> one of landmark / bing / other, from the LINCS gene table."""
    with gzip.open(gene_info, "rt") as fh:
        gi = pd.read_csv(fh, sep="\t")
    out: dict[str, str] = {}
    for sym, is_lm, is_bing in zip(
        gi["pr_gene_symbol"], gi["pr_is_lm"], gi["pr_is_bing"], strict=True
    ):
        s = str(sym)
        if int(is_lm) == 1:
            out[s] = "landmark"
        elif int(is_bing) == 1:
            out[s] = "bing"
        else:
            out[s] = "other"
    return out


def variance_matched_sample(
    pool: list[str], target: list[str], var: pd.Series, rng: np.random.Generator, n_bins: int = 10
) -> list[str]:
    """Draw from `pool` so its `var` distribution matches `target`'s, decile by decile.

    Without this, a landmark-vs-imputed gap is confounded with the fact that landmarks were
    SELECTED to be informative and highly expressed. Matching on the Tahoe-side variance --
    the platform that measured everything -- removes the part of the gap attributable to
    which genes were chosen rather than to how they were produced.
    """
    tv = var.reindex(target).dropna()
    pv = var.reindex(pool).dropna()
    if tv.empty or pv.empty:
        return []
    edges = np.quantile(tv.to_numpy(), np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    picked: list[str] = []
    for i in range(n_bins):
        want = int(((tv >= edges[i]) & (tv < edges[i + 1])).sum())
        cand = pv[(pv >= edges[i]) & (pv < edges[i + 1])].index.tolist()
        if not cand or want == 0:
            continue
        take = min(want, len(cand))
        picked.extend(rng.choice(cand, size=take, replace=False).tolist())
    return picked


def corr_by_class(
    l_vec: np.ndarray, t_vec: np.ndarray, genes: list[str], cls: dict[str, str]
) -> dict[str, float]:
    """Spearman correlation of two delta vectors, computed within each gene class."""
    out: dict[str, float] = {}
    arr = np.array([cls.get(g, "other") for g in genes])
    for c in GENE_CLASSES:
        m = arr == c
        if m.sum() < 20:
            out[c] = float("nan")
            continue
        a, b = l_vec[m], t_vec[m]
        ok = np.isfinite(a) & np.isfinite(b)
        out[c] = float(stats.spearmanr(a[ok], b[ok]).statistic) if ok.sum() >= 20 else float("nan")
    return out


def sha256(path: Path, cap: int = 64 << 20) -> str:
    """Hash of up to `cap` bytes -- identity for files too large to hash whole."""
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as fh:
        while n < cap and (b := fh.read(1 << 20)):
            h.update(b)
            n += len(b)
    return f"sha256:{h.hexdigest()}" + ("" if n < cap else f" (first {cap} bytes)")


def git_sha() -> str:
    """The commit this was measured at."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    """Build matched L1000/Tahoe deltas and score imputed-gene fidelity against landmarks."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l1000-dir", required=True, type=Path)
    ap.add_argument("--gctx", required=True)
    ap.add_argument("--deltas-bundle", required=True, type=Path)
    ap.add_argument("--pert-map", required=True, type=Path, help="Tahoe drug_name -> pubchem_cid")
    ap.add_argument("--model-csv", type=Path, default=Path("data/raw/gdsc2_sarcoma/depmap/Model.csv"))
    ap.add_argument("--time", type=float, default=24.0)
    ap.add_argument("--treated-cap", type=int, default=8)
    ap.add_argument("--dmso-cap", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    from cmapPy.pandasGEXpress.parse_gctx import parse  # Alpine-only dep

    # Created up front: the paired-test artifacts are written before the summary, and doing
    # this only at the end cost job 31661545 on a non-existent-directory error after all the
    # compute had finished.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cls = gene_class_map(args.l1000_dir / "GSE92742_Broad_LINCS_gene_info.txt.gz")
    print(f"gene classes: {pd.Series(list(cls.values())).value_counts().to_dict()}")

    # ---- Tahoe side -------------------------------------------------------------------
    t_delta = pd.read_parquet(args.deltas_bundle / "real_delta.parquet")
    t_key = pd.read_parquet(args.deltas_bundle / "real_key.parquet")
    model = pd.read_csv(args.model_csv, low_memory=False)
    ach2name = {
        norm_line(m): norm_line(n)
        for m, n in zip(model["ModelID"], model["StrippedCellLineName"], strict=True)
        if isinstance(n, str) and n
    }
    pm = pd.read_csv(args.pert_map, sep="\t", header=None, names=["drug_name", "pubchem_cid"])
    cid2name = {str(c).split(".")[0]: str(n) for n, c in zip(pm["drug_name"], pm["pubchem_cid"], strict=True)}

    t_key = t_key.assign(
        line=[ach2name.get(norm_line(p), norm_line(p)) for p in t_key["patient"]],
        dname=[norm_name(cid2name.get(str(d).split(".")[0], "")) for d in t_key["drug"]],
    )
    print(f"tahoe: {len(t_key)} pairs, {t_key['line'].nunique()} lines, {t_key['dname'].nunique()} drugs")

    # ---- L1000 side: restrict wells to the shared lines and drugs ----------------------
    with gzip.open(args.l1000_dir / "GSE92742_Broad_LINCS_inst_info.txt.gz", "rt") as fh:
        inst = pd.read_csv(fh, sep="\t", low_memory=False)
    inst["line"] = [norm_line(c) for c in inst["cell_id"]]
    inst["dname"] = [norm_name(n) for n in inst["pert_iname"]]

    shared_lines = sorted(set(t_key["line"]) & set(inst["line"]))
    shared_drugs = sorted({d for d in t_key["dname"] if d} & set(inst["dname"]))
    print(f"shared: {len(shared_lines)} lines {shared_lines}")
    print(f"shared: {len(shared_drugs)} drugs {shared_drugs}")
    if not shared_lines or not shared_drugs:
        raise SystemExit("no shared (line, drug) support -- nothing to compare")

    t_wells = inst[
        inst["line"].isin(shared_lines)
        & inst["dname"].isin(shared_drugs)
        & (inst["pert_time"] == args.time)
    ].copy()
    c_wells = inst[
        (inst["pert_iname"] == "DMSO")
        & inst["line"].isin(shared_lines)
        & (inst["pert_time"] == args.time)
    ].copy()
    t_wells = t_wells.sort_values("inst_id").groupby(["line", "dname"], sort=False).head(args.treated_cap)
    c_wells = c_wells.sort_values("inst_id").groupby("line", sort=False).head(args.dmso_cap)
    print(f"wells: {len(t_wells)} treated + {len(c_wells)} DMSO (time={args.time})")
    if t_wells.empty or c_wells.empty:
        raise SystemExit("no wells at this timepoint for the shared support")

    with gzip.open(args.l1000_dir / "GSE92742_Broad_LINCS_gene_info.txt.gz", "rt") as fh:
        sym = pd.read_csv(fh, sep="\t").set_index("pr_gene_id")["pr_gene_symbol"].astype(str)

    def group_means(ids: list[str], lab: dict[str, str]) -> pd.DataFrame:
        """Mean profile per label, reading the .gctx in column chunks so memory stays bounded."""
        tot: pd.DataFrame | None = None
        cnt: pd.Series | None = None
        for i in range(0, len(ids), args.chunk):
            block = parse(args.gctx, cid=ids[i : i + args.chunk]).data_df.T
            block.index = block.index.map(lab)
            s, n = block.groupby(level=0).sum(), block.groupby(level=0).size()
            tot = s if tot is None else tot.add(s, fill_value=0.0)
            cnt = n if cnt is None else cnt.add(n, fill_value=0)
        assert tot is not None and cnt is not None
        return tot.div(cnt, axis=0)

    t_lab = dict(zip(t_wells["inst_id"], t_wells["line"] + "\t" + t_wells["dname"], strict=True))
    c_lab = dict(zip(c_wells["inst_id"], c_wells["line"], strict=True))
    tmean = group_means(t_wells["inst_id"].tolist(), t_lab)
    dmean = group_means(c_wells["inst_id"].tolist(), c_lab)
    tmean.columns = pd.Index([str(sym.get(int(i), "")) for i in tmean.columns])
    dmean.columns = pd.Index([str(sym.get(int(i), "")) for i in dmean.columns])
    tmean = tmean.loc[:, (tmean.columns != "") & ~tmean.columns.duplicated()]
    dmean = dmean.loc[:, (dmean.columns != "") & ~dmean.columns.duplicated()]

    # ---- match pairs across platforms --------------------------------------------------
    t_index = {(r.line, r.dname): i for i, r in enumerate(t_key.itertuples())}
    genes = [g for g in tmean.columns if g in set(t_delta.columns)]
    print(f"genes shared between L1000 and Tahoe: {len(genes)}")
    var_t = t_delta[genes].var(axis=0)

    rows: list[dict[str, object]] = []
    l_vecs: list[np.ndarray] = []
    t_vecs: list[np.ndarray] = []
    labels: list[tuple[str, str]] = []
    for lab_str in tmean.index:
        line, dname = str(lab_str).split("\t")
        if line not in dmean.index or (line, dname) not in t_index:
            continue
        l_vec = (tmean.loc[lab_str, genes].to_numpy(dtype=float)
                 - dmean.loc[line, genes].to_numpy(dtype=float))
        t_vec = t_delta.iloc[t_index[(line, dname)]][genes].to_numpy(dtype=float)
        l_vecs.append(l_vec)
        t_vecs.append(t_vec)
        labels.append((line, dname))
        r = corr_by_class(l_vec, t_vec, genes, cls)
        rows.append({"line": line, "drug": dname, "matched": "observed", **r})
        print(f"  {line:<10} {dname:<16} " + "  ".join(f"{c}={r[c]:+.3f}" for c in GENE_CLASSES))

    if not rows:
        raise SystemExit("no (line, drug) pair present on both platforms at this timepoint")
    print(f"\n{len(rows)} matched (line, drug) pairs")

    # ---- control 1: variance-matched imputed genes -------------------------------------
    lm_genes = [g for g in genes if cls.get(g) == "landmark"]
    imp_genes = [g for g in genes if cls.get(g) in ("bing", "other")]
    matched = variance_matched_sample(imp_genes, lm_genes, var_t, rng)
    print(f"variance-matched imputed sample: {len(matched)} genes vs {len(lm_genes)} landmarks")
    gi_pos = {g: i for i, g in enumerate(genes)}
    m_idx = np.array([gi_pos[g] for g in matched], dtype=int)
    l_idx = np.array([gi_pos[g] for g in lm_genes], dtype=int)
    for (line, dname), lv, tv in zip(labels, l_vecs, t_vecs, strict=True):
        for name, idx in (("landmark", l_idx), ("imputed_varmatched", m_idx)):
            if idx.size < 20:
                continue
            a, b = lv[idx], tv[idx]
            ok = np.isfinite(a) & np.isfinite(b)
            rows.append({
                "line": line, "drug": dname, "matched": name,
                "spearman": round(float(stats.spearmanr(a[ok], b[ok]).statistic), 4),
                "n_genes": int(ok.sum()),
            })

    # ---- control 2: permutation null (mismatched pairs) --------------------------------
    null: dict[str, list[float]] = {c: [] for c in GENE_CLASSES}
    n = len(l_vecs)
    if n > 1:
        for _ in range(args.n_perm):
            i, j = rng.choice(n, size=2, replace=False)
            r = corr_by_class(l_vecs[i], t_vecs[j], genes, cls)
            for c in GENE_CLASSES:
                if np.isfinite(r[c]):
                    null[c].append(r[c])

    obs = pd.DataFrame([r for r in rows if r.get("matched") == "observed"])
    summary_rows = []
    print("\n================ SUMMARY ================")
    for c in GENE_CLASSES:
        o = obs[c].dropna()
        nl = np.asarray(null[c], dtype=float)
        n_cls = int(sum(1 for g in genes if cls.get(g, "other") == c))
        # p compares the observed MEAN over len(o) pairs against the bootstrapped sampling
        # distribution of the null MEAN at that same pair count -- not against the spread of
        # individual mismatched-pair draws, which is a different, much wider quantity and
        # inflates p by roughly sqrt(len(o)). See fmharness.statistics for why.
        p, _, _ = bootstrap_aggregate_pvalue(float(o.mean()), nl, len(o)) if len(o) else (float("nan"),) * 3
        summary_rows.append({
            "gene_class": c, "n_genes": n_cls, "n_pairs": int(len(o)),
            "mean_spearman": round(float(o.mean()), 4) if len(o) else float("nan"),
            "sd_spearman": round(float(o.std(ddof=1)), 4) if len(o) > 1 else float("nan"),
            "null_mean": round(float(nl.mean()), 4) if nl.size else float("nan"),
            "null_sd": round(float(nl.std(ddof=1)), 4) if nl.size > 1 else float("nan"),
            "lift_over_null": round(float(o.mean() - nl.mean()), 4) if nl.size and len(o) else float("nan"),
            "p_vs_null": round(p, 4),
        })
        print(f"  {c:<10} n_genes={n_cls:>6}  mean rho={summary_rows[-1]['mean_spearman']}"
              f"  null={summary_rows[-1]['null_mean']}  lift={summary_rows[-1]['lift_over_null']}"
              f"  p={summary_rows[-1]['p_vs_null']}")

    # ---- the PAIRED test, which is what the design actually calls for -------------------
    # The summary above compares each class's MEAN against a null of mismatched pairs. That
    # answers "is there cross-platform agreement at all", and it is swamped by pair-level noise:
    # dose, replicate count and effect size vary enormously between pairs and none of that is
    # about imputation. The stated design is a WITHIN-PAIR comparison, and its correct
    # aggregation is a paired signed-rank test over pairs, where each pair is its own control.
    # Added after the marginal aggregation proved uninformative; the within-pair design was
    # fixed in advance, the choice of paired statistic was not.
    paired_rows = []
    for a, b in (("landmark", "bing"), ("landmark", "other"), ("bing", "other")):
        d = (obs[a] - obs[b]).dropna()
        if len(d) < 6 or not np.any(d != 0):
            continue
        w = stats.wilcoxon(d)
        paired_rows.append({
            "comparison": f"{a} - {b}", "n_pairs": int(len(d)),
            "n_favoring_first": int((d > 0).sum()),
            "median_delta_rho": round(float(d.median()), 4),
            "wilcoxon_p": round(float(w.pvalue), 4),
        })
    if paired_rows:
        print("\n  paired within-pair comparison (each pair is its own control):")
        for r in paired_rows:
            print(f"    {r['comparison']:<20} {r['n_favoring_first']}/{r['n_pairs']} pairs"
                  f"  median drho={r['median_delta_rho']:+.4f}  Wilcoxon p={r['wilcoxon_p']}")
        pd.DataFrame(paired_rows).to_csv(args.out_dir / "l1000_imputation_fidelity_paired.csv", index=False)

    vm = pd.DataFrame([r for r in rows if r.get("matched") in ("landmark", "imputed_varmatched")])
    if not vm.empty:
        print("\n  variance-matched comparison (equal gene counts, matched variance profile):")
        for name, grp in vm.groupby("matched"):
            print(f"    {name:<22} mean rho={grp['spearman'].mean():+.4f}  n={len(grp)} pairs")
        piv = vm.pivot_table(index=["line", "drug"], columns="matched", values="spearman")
        if {"landmark", "imputed_varmatched"} <= set(piv.columns):
            dv = (piv["landmark"] - piv["imputed_varmatched"]).dropna()
            if len(dv) >= 6 and np.any(dv != 0):
                wv = stats.wilcoxon(dv)
                print(f"    paired, variance-matched: landmark higher in {int((dv > 0).sum())}/{len(dv)}"
                      f" pairs, median drho={float(dv.median()):+.4f}, Wilcoxon p={float(wv.pvalue):.4f}")
                paired_rows.append({
                    "comparison": "landmark - imputed_varmatched", "n_pairs": int(len(dv)),
                    "n_favoring_first": int((dv > 0).sum()),
                    "median_delta_rho": round(float(dv.median()), 4),
                    "wilcoxon_p": round(float(wv.pvalue), 4),
                })
                pd.DataFrame(paired_rows).to_csv(
                    args.out_dir / "l1000_imputation_fidelity_paired.csv", index=False
                )
        summary_rows.extend(
            {"gene_class": f"varmatched::{name}", "n_genes": int(grp["n_genes"].iloc[0]),
             "n_pairs": int(len(grp)), "mean_spearman": round(float(grp["spearman"].mean()), 4),
             "sd_spearman": round(float(grp["spearman"].std(ddof=1)), 4) if len(grp) > 1 else float("nan"),
             "null_mean": float("nan"), "null_sd": float("nan"),
             "lift_over_null": float("nan"), "p_vs_null": float("nan")}
            for name, grp in vm.groupby("matched")
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "l1000_imputation_fidelity.csv", index=False)
    pd.DataFrame(rows).to_csv(args.out_dir / "l1000_imputation_fidelity_per_pair.csv", index=False)
    (args.out_dir / "l1000_imputation_fidelity.params.json").write_text(
        json.dumps({
            "git_sha": git_sha(),
            "args": {k: str(v) for k, v in vars(args).items()},
            "shared_lines": shared_lines, "shared_drugs": shared_drugs,
            "n_matched_pairs": len(labels),
            "matched_pairs": [f"{a}|{b}" for a, b in labels],
            "n_genes_shared": len(genes),
            "gene_class_counts": {c: int(sum(1 for g in genes if cls.get(g, "other") == c))
                                  for c in GENE_CLASSES},
            "n_varmatched_genes": len(matched),
            "inputs": {
                "gctx": sha256(Path(args.gctx)),
                "gene_info": sha256(args.l1000_dir / "GSE92742_Broad_LINCS_gene_info.txt.gz"),
                "real_delta": sha256(args.deltas_bundle / "real_delta.parquet"),
            },
        }, indent=2) + "\n"
    )
    print(f"\nwrote {args.out_dir}/l1000_imputation_fidelity.csv and sidecar")


if __name__ == "__main__":
    main()
