"""Four-panel summary of the Tahoe generation eval (slide 6 of the harness overview).

Numbers are the reported values in ``docs/tahoe_generation_results.md`` (branch
``tahoe-generation-eval``); this script only lays them out, so the slide figure can be
regenerated without re-running the Alpine pipeline.

    python3 scripts/plot_generation_eval_summary.py [--out docs/figures/generation_eval_summary.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- reported values -------------------------------------------------------------------
# Check 1: delta-Pearson of each source against the real Tahoe delta, and the off-diagonal
# (wrong-condition) control.
CHECK1 = {
    "additive": (0.225, 0.095),
    "knn": (0.178, 0.067),
    "pca": (0.207, 0.083),
    "nmf": (0.221, 0.088),
    "stack\n(generated)": (0.012, -0.002),
}
DELTA_CEILING, DELTA_SPLIT_HALF = 0.46, 0.30

INTERACTION_CEILING = 0.47  # GDSC2 vs CTRPv2 agreement on the cell-line-specific axis

# Check 2, ridge (L2): overall potency (global) and cell-line-specific response (interaction).
CHECK2_RIDGE = {
    "expr": (0.475, -0.037),
    "additive": (0.628, -0.095),
    "knn": (0.547, -0.068),
    "pca": (0.585, 0.007),
    "nmf": (0.550, 0.007),
    "stack\n(gen delta)": (0.540, -0.003),
    "base\n(embed)": (0.644, 0.119),
    "aligned\n(embed)": (0.618, 0.045),
}
# Selection gap@k, lowest across the L1/L2/EN sweep (each k minimized independently).
SEL_GAP = {
    "expr": (0.360, 0.114),
    "additive": (0.264, 0.091),
    "knn": (0.250, 0.101),
    "pca": (0.219, 0.102),
    "nmf": (0.251, 0.082),
    "stack\n(gen delta)": (0.320, 0.140),
    "base\n(embed)": (0.273, 0.102),
    "aligned\n(embed)": (0.240, 0.096),
}
RANDOM_TOP1 = 0.70  # range-normalized gap of a random ranking on a right-skewed AUC panel

# How often the line's genuinely best drug is already one of the broadly-active (pan-toxic)
# compounds, measured on real GDSC2 AUC resampled to the Check-2 geometry (50 lines x 26 drugs).
# Produced by scripts/pick_concentration_reference.py -> results/pick_concentration_reference.json.
TOXIC_SHARE_OBSERVED = {1: (0.890, 0.489, 1.000), 3: (0.994, 0.950, 1.000)}  # mean, lo95, hi95
TOXIC_SHARE_PRIOR = {1: 1.0, 3: 1.0}  # potency prior picks by mean AUC -> pan-actives by definition
DISTINCT_TOP1_OBSERVED = (6.7, 2.0, 13.0)  # distinct drugs ever best, out of 26

BLUE, BLUE_LT = "#2C6FAF", "#A8C6E2"
ORANGE, PURPLE, PURPLE_LT = "#D98C29", "#7B52A8", "#C3AEDA"
STACK_RED, BASE_GREEN = "#C0392B", "#2E7D50"


def _edges(labels: list[str]) -> tuple[list[str], list[float]]:
    """Outline (not refill) the Stack rows, so the series legend stays truthful."""
    colors, widths = [], []
    for lab in labels:
        if "stack" in lab:
            colors.append(STACK_RED)
            widths.append(2.2)
        elif "embed" in lab:
            colors.append(BASE_GREEN)
            widths.append(2.2)
        else:
            colors.append("none")
            widths.append(0.0)
    return colors, widths


def _annotate(ax, xs, values, *, fmt="{:.2f}", pad=0.012, bold_idx=()):
    for i, (x, v) in enumerate(zip(xs, values)):
        v = 0.0 if abs(v) < 0.005 else v
        ax.annotate(
            fmt.format(v),
            (x, v + pad if v >= 0 else v - pad),
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9,
            fontweight="bold" if i in bold_idx else "normal",
        )


def panel_check1(ax) -> None:
    labels = list(CHECK1)
    vals = np.array(list(CHECK1.values()))
    x = np.arange(len(labels))
    ec, lw = _edges(labels)
    ax.bar(
        x - 0.19, vals[:, 0], 0.36, color=BLUE, edgecolor=ec, linewidth=lw,
        label="predicted vs real change",
    )
    ax.bar(x + 0.19, vals[:, 1], 0.36, color=BLUE_LT, label="match to the WRONG condition")
    ax.axhline(DELTA_CEILING, ls="--", color="#C0392B", lw=1.4, label=f"ceiling {DELTA_CEILING:.2f}")
    ax.axhline(DELTA_SPLIT_HALF, ls=":", color="#E08214", lw=1.4, label=f"split-half {DELTA_SPLIT_HALF:.2f}")
    _annotate(ax, x - 0.19, vals[:, 0], bold_idx=(len(labels) - 1,))
    ax.set_xticks(x, labels)
    ax.set_ylim(-0.05, 0.55)
    ax.set_ylabel("correlation")
    ax.set_title("Check 1: predicted vs. real drug-induced change", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")


def panel_toxic_share(ax) -> None:
    """How often a pan-toxic drug is in the top k -- for the truth and for a line-blind ranker.

    The model bars are deliberately absent: this panel predates `check2_selection_audit.py`,
    which now dumps per-pair predictions to `results/check2_preds.parquet` and each
    representation's shortlist concentration is recoverable and reported in
    `results/check2_selection_audit.csv` and `docs/tahoe_generation_results.md`. This panel's
    hatched placeholder bars have not been backfilled with those numbers.
    """
    ks = (1, 3)
    x = np.arange(len(ks))
    obs = [TOXIC_SHARE_OBSERVED[k][0] for k in ks]
    err = np.array([[TOXIC_SHARE_OBSERVED[k][0] - TOXIC_SHARE_OBSERVED[k][1] for k in ks],
                    [TOXIC_SHARE_OBSERVED[k][2] - TOXIC_SHARE_OBSERVED[k][0] for k in ks]])
    ax.bar(x - 0.26, obs, 0.24, color=BLUE, yerr=err, capsize=4,
           label="observed best drug (the truth)")
    ax.bar(x, [TOXIC_SHARE_PRIOR[k] for k in ks], 0.24, color=ORANGE,
           label="potency prior (ignores the cell line)")
    ax.bar(x + 0.26, [1.0, 1.0], 0.24, facecolor="none", edgecolor="#999999", hatch="///",
           linewidth=1.2, label="each representation (not yet computable)")
    for xi, k in zip(x, ks):
        ax.annotate(f"{TOXIC_SHARE_OBSERVED[k][0]:.2f}",
                    (xi - 0.26, TOXIC_SHARE_OBSERVED[k][2] + 0.03),
                    ha="center", fontsize=9)
    for xi in x:
        ax.annotate("?", (xi + 0.26, 0.46), ha="center", fontsize=15, color="#777777",
                    fontweight="bold")
    ax.set_xticks(x, [f"top-{k}" for k in ks])
    ax.set_ylim(0, 1.32)
    ax.set_ylabel("share of picks containing a pan-toxic drug")
    ax.set_title("Are the top picks just the pan-toxic drugs?", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    ax.set_xlabel(
        f"even the TRUTH is {obs[0]:.0%} pan-toxic at top-1 and {obs[1]:.0%} at top-3, so this share\n"
        f"saturates; the separating view is how many DISTINCT drugs get picked #1\n"
        f"(truth {DISTINCT_TOP1_OBSERVED[0]:.1f} of 26, 95% {DISTINCT_TOP1_OBSERVED[1]:.0f}-"
        f"{DISTINCT_TOP1_OBSERVED[2]:.0f}; prior 1 by construction)",
        fontsize=8.5, color="#444444", labelpad=8,
    )


def panel_check2(ax) -> None:
    labels = list(CHECK2_RIDGE)
    vals = np.array(list(CHECK2_RIDGE.values()))
    x = np.arange(len(labels))
    ax.bar(x - 0.19, vals[:, 0], 0.36, color=BLUE, label="overall potency (global)")
    ec, lw = _edges(labels)
    ax.bar(
        x + 0.19, vals[:, 1], 0.36, color=ORANGE, edgecolor=ec, linewidth=lw,
        label="cell-line-specific (interaction)",
    )
    ax.axhline(INTERACTION_CEILING, ls="--", color="#C0392B", lw=1.4, label=f"interaction ceiling {INTERACTION_CEILING:.2f}")
    ax.axhline(0, color="#666666", lw=0.8)
    _annotate(ax, x - 0.19, vals[:, 0])
    _annotate(ax, x + 0.19, vals[:, 1], fmt="{:+.2f}", pad=0.02, bold_idx=(labels.index("base\n(embed)"),))
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylim(-0.22, 0.95)
    ax.set_ylabel("Spearman (higher = better)")
    ax.set_title("Check 2: potency vs. personalization  (trained ridge model)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)


def panel_selection_gap(ax) -> None:
    labels = list(SEL_GAP)
    vals = np.array(list(SEL_GAP.values()))
    x = np.arange(len(labels))
    ec, lw = _edges(labels)
    ax.bar(
        x - 0.19, vals[:, 0], 0.36, color=PURPLE, edgecolor=ec, linewidth=lw,
        label="selection gap@1 (top pick)",
    )
    ax.bar(x + 0.19, vals[:, 1], 0.36, color=PURPLE_LT, label="selection gap@3 (top-3 list)")
    ax.axhline(RANDOM_TOP1, ls="--", color="#C0392B", lw=1.4, label=f"random top-1 ~ {RANDOM_TOP1:.2f}")
    _annotate(ax, x - 0.19, vals[:, 0])
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylim(0, 0.86)
    ax.set_ylabel("gap (lower = better; 0 = best drug)")
    ax.set_title("Check 2: best-drug shortlisting  (best of L1/L2/EN)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/generation_eval_summary.png"))
    args = ap.parse_args()

    fig = plt.figure(figsize=(17.8, 9.6), dpi=150)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.12))
    axes = {
        "check1": fig.add_subplot(grid[0, 0]),
        "toxic": fig.add_subplot(grid[0, 1]),
        "check2": fig.add_subplot(grid[1, 0]),
        "gap": fig.add_subplot(grid[1, 1]),
    }
    panel_check1(axes["check1"])
    panel_toxic_share(axes["toxic"])
    panel_check2(axes["check2"])
    panel_selection_gap(axes["gap"])

    axes["check2"].set_xlabel(
        "overall potency ~ 0.6 (near its own ceiling); personalization ~ 0 everywhere except the "
        "base Stack embedding (interaction +0.12, p = 0.001)",
        fontsize=9,
        color="#444444",
        labelpad=8,
    )
    axes["gap"].set_xlabel(
        "beats a random pick, but the six lowest bars span 0.054 — below the 0.047-0.106\n"
        "detectable difference, so they are mutually indistinguishable",
        fontsize=9,
        color="#444444",
        labelpad=8,
    )
    fig.suptitle(
        "Current results — grouped 5-fold-by-cell-line ladder, CV-tuned penalties "
        "(Stack generation null; base embedding carries the personalization signal)",
        fontsize=15,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
