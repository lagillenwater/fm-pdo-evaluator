"""Why is L1000-vs-Tahoe delta agreement ~zero even for the 978 MEASURED genes?

Job 31661570 found landmark genes -- measured on both platforms -- at mean Spearman 0.041,
p=0.2438 against a mismatched-pair null. That is the positive control failing, and it makes
every other number in that run uninterpretable. This establishes WHY, which should have been
done before that test ran, not after.

Spearman answers only whether the gene RANKING agrees. Three distinct things could each drive
it to zero, and they call for different responses:

  NOISE CEILING. If L1000's own delta is not reproducible across its replicate wells, no
  cross-platform correlation is possible and the experiment was doomed regardless of biology.
  Splitting each pair's treated and DMSO wells in half gives two independent deltas from the
  same data; their correlation is the ceiling any cross-platform comparison could reach. This
  is measured FIRST because it bounds everything else.

  DIRECTION. Platforms could agree on which genes move up or down while disagreeing on rank
  order. Sign concordance against a 50% chance rate tests that directly, and is reported both
  over all landmark genes and over the genes Tahoe says actually moved -- near-zero genes have
  essentially random sign and dilute the statistic toward 50%.

  MAGNITUDE. L1000 deltas could be near-zero -- wrong dose, wrong timepoint, or a compressed
  assay scale -- in which case the comparison is between a real signal and nothing. Median
  absolute delta per platform, and their ratio, separates a compressed signal from an absent
  one.

Emits per-pair and summary artifacts. No number here is reported without one.
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


def norm_name(s: object) -> str:
    """Lowercase alphanumeric, for joining drug names across cohorts."""
    return "".join(c for c in str(s).lower() if c.isalnum())


def norm_line(s: object) -> str:
    """Uppercase alphanumeric, matching fmharness.deltas._norm."""
    import re

    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


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
    """Measure the noise ceiling, sign concordance and magnitude for each shared pair."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l1000-dir", required=True, type=Path)
    ap.add_argument("--gctx", required=True)
    ap.add_argument("--deltas-bundle", required=True, type=Path)
    ap.add_argument("--pert-map", required=True, type=Path)
    ap.add_argument("--model-csv", type=Path, default=Path("data/raw/gdsc2_sarcoma/depmap/Model.csv"))
    ap.add_argument("--time", type=float, default=24.0)
    ap.add_argument("--treated-cap", type=int, default=8)
    ap.add_argument("--dmso-cap", type=int, default=60)
    ap.add_argument("--top-n", type=int, default=100, help="genes by |Tahoe delta| for the moved-gene arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    from cmapPy.pandasGEXpress.parse_gctx import parse

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with gzip.open(args.l1000_dir / "GSE92742_Broad_LINCS_gene_info.txt.gz", "rt") as fh:
        gi = pd.read_csv(fh, sep="\t")
    sym = gi.set_index("pr_gene_id")["pr_gene_symbol"].astype(str)
    landmark = {str(s) for s, lm in zip(gi["pr_gene_symbol"], gi["pr_is_lm"], strict=True) if int(lm) == 1}
    print(f"landmark genes declared: {len(landmark)}")

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

    with gzip.open(args.l1000_dir / "GSE92742_Broad_LINCS_inst_info.txt.gz", "rt") as fh:
        inst = pd.read_csv(fh, sep="\t", low_memory=False)
    inst["line"] = [norm_line(c) for c in inst["cell_id"]]
    inst["dname"] = [norm_name(n) for n in inst["pert_iname"]]

    shared_lines = sorted(set(t_key["line"]) & set(inst["line"]))
    shared_drugs = sorted({d for d in t_key["dname"] if d} & set(inst["dname"]))
    t_wells = inst[
        inst["line"].isin(shared_lines) & inst["dname"].isin(shared_drugs)
        & (inst["pert_time"] == args.time)
    ].sort_values("inst_id").groupby(["line", "dname"], sort=False).head(args.treated_cap)
    c_wells = inst[
        (inst["pert_iname"] == "DMSO") & inst["line"].isin(shared_lines)
        & (inst["pert_time"] == args.time)
    ].sort_values("inst_id").groupby("line", sort=False).head(args.dmso_cap)
    print(f"{len(shared_lines)} lines x {len(shared_drugs)} drugs -> "
          f"{len(t_wells)} treated + {len(c_wells)} DMSO wells")

    def well_matrix(ids: list[str]) -> pd.DataFrame:
        """wells x gene-symbols for the given inst_ids."""
        blocks = []
        for i in range(0, len(ids), 2000):
            blocks.append(parse(args.gctx, cid=ids[i : i + 2000]).data_df.T)
        m = pd.concat(blocks)
        m.columns = pd.Index([str(sym.get(int(c), "")) for c in m.columns])
        return m.loc[:, (m.columns != "") & ~m.columns.duplicated()]

    tm = well_matrix(t_wells["inst_id"].tolist())
    cm = well_matrix(c_wells["inst_id"].tolist())
    t_index = {(r.line, r.dname): i for i, r in enumerate(t_key.itertuples())}

    rows: list[dict[str, object]] = []
    for (line, dname), grp in t_wells.groupby(["line", "dname"], sort=False):
        if (line, dname) not in t_index:
            continue
        ctrl_ids = c_wells.loc[c_wells["line"] == line, "inst_id"].tolist()
        if len(ctrl_ids) < 2 or len(grp) < 2:
            continue
        tw, cw = grp["inst_id"].tolist(), ctrl_ids
        genes = [g for g in tm.columns if g in landmark and g in set(t_delta.columns)]
        tv = t_delta.iloc[t_index[(line, dname)]][genes].to_numpy(dtype=float)

        full = tm.loc[tw, genes].mean(axis=0).to_numpy() - cm.loc[cw, genes].mean(axis=0).to_numpy()

        # noise ceiling: two independent deltas from disjoint halves of the SAME wells
        ti, ci = rng.permutation(len(tw)), rng.permutation(len(cw))
        ta, tb = [tw[i] for i in ti[: len(tw) // 2]], [tw[i] for i in ti[len(tw) // 2 :]]
        ca, cb = [cw[i] for i in ci[: len(cw) // 2]], [cw[i] for i in ci[len(cw) // 2 :]]
        d_a = tm.loc[ta, genes].mean(axis=0).to_numpy() - cm.loc[ca, genes].mean(axis=0).to_numpy()
        d_b = tm.loc[tb, genes].mean(axis=0).to_numpy() - cm.loc[cb, genes].mean(axis=0).to_numpy()
        split = float(stats.spearmanr(d_a, d_b).statistic)

        # direction: over all landmarks, and over the genes Tahoe says actually moved
        top = np.argsort(-np.abs(tv))[: args.top_n]
        sign_all = float(np.mean(np.sign(full) == np.sign(tv)))
        sign_top = float(np.mean(np.sign(full[top]) == np.sign(tv[top])))

        rows.append({
            "line": line, "drug": dname,
            "n_treated_wells": int(len(tw)), "n_dmso_wells": int(len(cw)),
            "n_landmark_genes": int(len(genes)),
            "splithalf_rho_l1000": round(split, 4),
            "cross_platform_rho": round(float(stats.spearmanr(full, tv).statistic), 4),
            "sign_agree_all": round(sign_all, 4),
            "sign_agree_top": round(sign_top, 4),
            "median_abs_l1000": round(float(np.median(np.abs(full))), 4),
            "median_abs_tahoe": round(float(np.median(np.abs(tv))), 4),
            "magnitude_ratio": round(float(np.median(np.abs(full)) / max(np.median(np.abs(tv)), 1e-9)), 3),
        })
        print(f"  {line:<9} {dname:<14} wells={len(tw)}/{len(cw)} splithalf={split:+.3f} "
              f"cross={rows[-1]['cross_platform_rho']:+.3f} sign_all={sign_all:.3f} "
              f"sign_top={sign_top:.3f} |L1000|={rows[-1]['median_abs_l1000']:.3f} "
              f"|Tahoe|={rows[-1]['median_abs_tahoe']:.3f}")

    if not rows:
        raise SystemExit("no pair had enough replicate wells to split -- nothing measured")
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "l1000_tahoe_agreement_per_pair.csv", index=False)

    def ci(x: pd.Series) -> str:
        """Mean with a one-sample t interval, so a mean is never printed bare."""
        m, se = float(x.mean()), float(x.sem())
        lo, hi = stats.t.interval(0.95, len(x) - 1, loc=m, scale=se) if len(x) > 1 else (m, m)
        return f"{m:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}]"

    print("\n================ SUMMARY ================")
    summary = []
    for col, label in (
        ("splithalf_rho_l1000", "L1000 split-half (NOISE CEILING)"),
        ("cross_platform_rho", "cross-platform rho"),
        ("sign_agree_all", "sign agreement, all landmarks"),
        ("sign_agree_top", f"sign agreement, top-{args.top_n} moved"),
        ("median_abs_l1000", "median |delta| L1000"),
        ("median_abs_tahoe", "median |delta| Tahoe"),
        ("magnitude_ratio", "magnitude ratio L1000/Tahoe"),
    ):
        print(f"  {label:<36} {ci(df[col])}")
        summary.append({"metric": col, "label": label, "mean": round(float(df[col].mean()), 4),
                        "sd": round(float(df[col].std(ddof=1)), 4), "n_pairs": int(len(df))})

    # Sign agreement has a known chance rate; a mean near 0.5 is the null, not a weak signal.
    for col in ("sign_agree_all", "sign_agree_top"):
        tt = stats.ttest_1samp(df[col], 0.5)
        print(f"  {col} vs 0.5 chance: t={tt.statistic:+.3f} p={tt.pvalue:.4f}")
        summary.append({"metric": f"{col}_vs_chance", "label": "one-sample t vs 0.5",
                        "mean": round(float(tt.statistic), 4), "sd": float("nan"),
                        "n_pairs": int(len(df)), "p": round(float(tt.pvalue), 4)})

    pd.DataFrame(summary).to_csv(args.out_dir / "l1000_tahoe_agreement_summary.csv", index=False)
    (args.out_dir / "l1000_tahoe_agreement.params.json").write_text(
        json.dumps({
            "git_sha": git_sha(), "args": {k: str(v) for k, v in vars(args).items()},
            "n_pairs": int(len(df)), "shared_lines": shared_lines, "shared_drugs": shared_drugs,
            "inputs": {"gctx": sha256(Path(args.gctx)),
                       "real_delta": sha256(args.deltas_bundle / "real_delta.parquet")},
        }, indent=2) + "\n"
    )
    print(f"\nwrote per-pair and summary artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
