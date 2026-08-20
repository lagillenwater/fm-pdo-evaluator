"""Summary figure for Path B's faithful-generation rerun: the fair source (additive/PCA/NMF/
Stack) x readout-adapter (hallmark/l1/l2) grid against real Soragni organoid viability.

Reads docs/soragni_pathb_results.csv (written by scripts/score_viability_adapters.py's
--out-csv against 24-synthetic-replicate/patient, --mode mdm generation -- see
scripts/alpine/10-12_soragni_*.sbatch); this script only lays the numbers out, so the figure
can be regenerated without re-running the Alpine pipeline.

    python3 scripts/plot_soragni_pathb_summary.py [--out docs/figures/soragni_pathb_summary.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOURCES = ("additive", "pca", "nmf", "stack")
METHODS = ("hallmark", "l1", "l2")
METHOD_COLOR = {"hallmark": "#2C6FAF", "l1": "#D98C29", "l2": "#7B52A8"}
STACK_RED = "#C0392B"
BONFERRONI = 0.05 / (len(SOURCES) * len(METHODS))  # 12 independent-ish cells in the grid


def _pivot(df: pd.DataFrame, col: str) -> np.ndarray:
    return (
        df.pivot(index="source", columns="method", values=col)
        .reindex(index=SOURCES, columns=METHODS)
        .to_numpy()
    )


def panel(
    ax,
    df: pd.DataFrame,
    col: str,
    *,
    title: str,
    ylabel: str,
    fmt: str,
    better: str,
    legend_loc: str,
) -> None:
    vals = _pivot(df, col)
    p_label = _pivot(df, "p_label")
    n_sources = len(SOURCES)
    width = 0.8 / len(METHODS)
    x = np.arange(n_sources)
    for j, method in enumerate(METHODS):
        xs = x - 0.4 + width * (j + 0.5)
        sig = p_label[:, j] < 0.05
        edge = [STACK_RED if s else "none" for s in sig]
        lw = [2.4 if s else 0.0 for s in sig]
        ax.bar(
            xs, vals[:, j], width * 0.92, color=METHOD_COLOR[method], edgecolor=edge, linewidth=lw
        )
        for xi, v, s in zip(xs, vals[:, j], sig, strict=True):
            ax.annotate(
                fmt.format(v) + ("*" if s else ""),
                (xi, v + (0.012 if v >= 0 else -0.012)),
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8,
                fontweight="bold" if s else "normal",
                color=STACK_RED if s else "black",
            )
    ax.axhline(0, color="#666666", lw=0.8)
    pad = (vals.max() - vals.min()) * 0.18
    ax.set_ylim(min(0, vals.min()) - pad, max(0, vals.max()) + pad)
    ax.set_xticks(x, SOURCES)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(f"{better}  |  * = p_label < 0.05 (uncorrected)", fontsize=8.5, color="#444444")
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLOR[m]) for m in METHODS]
    ax.legend(handles, METHODS, fontsize=8, loc=legend_loc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("docs/soragni_pathb_results.csv"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures/soragni_pathb_summary.png"))
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    best = df.loc[df["p_label"].idxmin()]
    any_sig = bool((df["p_label"] < 0.05).any())

    fig = plt.figure(figsize=(13.5, 6.4), dpi=150)
    grid = fig.add_gridspec(1, 2, wspace=0.3)
    panel(
        fig.add_subplot(grid[0, 0]),
        df,
        "interaction",
        title="Cell-line (patient)-specific response",
        ylabel="Spearman interaction (higher = better)",
        fmt="{:+.2f}",
        better="higher is better",
        legend_loc="upper left",
    )
    panel(
        fig.add_subplot(grid[0, 1]),
        df,
        "regret@1",
        title="Best-drug shortlisting (top-1 pick)",
        ylabel="selection gap@1 (lower = better; 0 = best drug)",
        fmt="{:.2f}",
        better="lower is better",
        legend_loc="lower right",
    )

    if any_sig:
        headline = (
            f"{best['source']}+{best['method']} is the only significant cell of 12 "
            f"(interaction {best['interaction']:+.2f}, p={best['p_label']:.3f}) -- "
            f"does not survive Bonferroni across the grid (p<{BONFERRONI:.4f})"
        )
    else:
        headline = (
            f"NULL -- no cell of 12 reaches p<0.05; best is {best['source']}+{best['method']} "
            f"(interaction {best['interaction']:+.2f}, p={best['p_label']:.3f})"
        )
    fig.suptitle(
        f"Path B faithful generation (24 synthetic replicates/patient, --mode mdm): {headline}",
        fontsize=12.5,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.text(
        0.01,
        0.005,
        "Matches the June (vanilla-mode) Path B run's null; n=17 patients underlies every "
        "interaction/regret estimate regardless of pair count. Corrected 2026-08-20: an earlier "
        "run of this pipeline scored against a stale, mismatched (organoid-RNA) baseline file "
        "and found a significant stack+hallmark cell that did not survive fixing the baseline.",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
