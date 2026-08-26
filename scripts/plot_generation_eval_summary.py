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
    "observed_delta": (0.225, 0.095),
    "knn": (0.178, 0.067),
    "pca": (0.207, 0.083),
    "nmf": (0.221, 0.088),
    "stack\n(cytokine)": (-0.001, -0.005),
    "stack\n(drug)": (0.006, -0.002),
}
DELTA_CEILING, DELTA_SPLIT_HALF = 0.46, 0.30

# Check 1b (2026-08-19): DE-restricted metrics (fmharness.evaluation.score_de_metrics), scored
# against ground-truth Wilcoxon DE calls -- the sparse counterpart to CHECK1's dense Pearson-Delta.
CHECK1B = {
    "observed_delta": (0.389, 0.012, 0.017, 0.009),
    "knn": (0.382, 0.010, 0.009, 0.005),
    "pca": (0.414, 0.034, 0.023, 0.014),
    "nmf": (0.414, 0.041, 0.029, 0.019),
    "stack\n(cytokine)": (0.357, 0.030, 0.049, 0.026),
    "stack\n(drug)": (0.466, 0.075, 0.076, 0.047),
}
# Permutation-null-corrected specific lift (observed - null_mean, n_perm=200/checkpoint) for the
# two stack rows only -- baselines have no null computed (see docs/tahoe_generation_results.md's
# Check 1b for the full null_mean/null_std/p table). Used to annotate the stack bars instead of
# their raw value, since the raw value conflates real pair-specific signal with a nonzero generic-
# correlation floor.
CHECK1B_STACK_LIFT = {
    "stack\n(cytokine)": (0.228, 0.0072, 0.0137, 0.0076),
    "stack\n(drug)": (0.273, 0.0119, 0.0123, 0.0075),
}
CHECK1B_METRICS = ("de_spearman_lfc", "pr_auc", "overlap_accuracy", "jaccard")

INTERACTION_CEILING = 0.47  # GDSC2 vs CTRPv2 agreement on the cell-line-specific axis

# Check 2, ridge (L2): overall potency (global) and cell-line-specific response (interaction).
CHECK2_RIDGE = {
    "expr": (0.475, -0.037),
    "observed_delta": (0.628, -0.095),
    "knn": (0.547, -0.068),
    "pca": (0.585, 0.007),
    "nmf": (0.550, 0.007),
    "stack\n(cytokine)": (0.539, -0.003),
    "stack\n(drug,\nunfilt.)":(0.561, -0.082),
    "stack\n(drug)": (0.561, -0.079),
    "base\n(embed)": (0.644, 0.119),
    "aligned\n(embed)": (0.618, 0.045),
}
# Selection gap@k, lowest across the L1/L2/EN sweep (each k minimized independently).
SEL_GAP = {
    "expr": (0.354, 0.119),
    "observed_delta": (0.264, 0.091),
    "knn": (0.250, 0.101),
    "pca": (0.219, 0.102),
    "nmf": (0.251, 0.082),
    "stack\n(cytokine)": (0.320, 0.144),
    "stack\n(drug,\nunfilt.)":(0.343, 0.133),
    "stack\n(drug)": (0.324, 0.122),
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

# Per-representation top-1 pick concentration (2026-08-19): fmharness.check2.penalized_preds
# already returns full per-pair predictions -- score_check2 just discards them after scoring.
# Computed by replicating check2_registry_driver.run_check2's representation construction,
# picking each representation's best-gap@1 penalty (matching how SEL_GAP's own "best of
# L1/L2/EN" was built: gap@1's own values here reproduce SEL_GAP to 3 decimals for every
# non-stack representation), then scoring top-1 picks against scripts/pick_concentration_
# reference.py's own broad-drug definition (top quartile by breadth, observed GDSC2 AUC,
# Tahoe-matched CIDs). top-3 broad-share is omitted -- saturates at 0.98-1.00 for every
# representation, same as the truth's own 99%, so it carries no separating information (see the
# panel's xlabel). Caveat: the stack rows pair the Aug-12 vanilla-mode generated deltas
# (whatever produced the currently-published CHECK2_RIDGE/SEL_GAP numbers -- the original
# 50-row pseudobulk query baseline wasn't preserved) with the newer tahoe_query_baseline.h5ad --
# an approximation on baseline MAGNITUDE, not on per-line RANKING, which is what top-1-pick
# concentration actually depends on.
TOXIC_SHARE_BY_REPR = {
    # label: (broad_share_top1, distinct_top1, n_lines)
    "expr": (0.886, 7, 44),
    "observed_delta": (1.000, 3, 44),
    "knn": (1.000, 3, 44),
    "pca": (0.977, 4, 44),
    "nmf": (1.000, 3, 44),
    "stack\n(cytokine)": (0.886, 5, 44),
    "stack\n(drug,\nunfilt.)": (0.932, 7, 44),
    "base\n(embed)": (0.977, 4, 44),
    "aligned\n(embed)": (0.977, 4, 44),
}

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


def panel_check1b(axes) -> None:
    """One small subplot per DE metric -- de_spearman_lfc (~0.35-0.47) and pr_auc/overlap/jaccard
    (~0.01-0.08) differ by 5-10x, so a single shared y-axis would flatten three of the four."""
    labels = list(CHECK1B)
    x = np.arange(len(labels))
    ec, lw = _edges(labels)
    for ax, metric in zip(axes, CHECK1B_METRICS, strict=True):
        vals = np.array([CHECK1B[lab][CHECK1B_METRICS.index(metric)] for lab in labels])
        ax.bar(x, vals, 0.6, color=BLUE, edgecolor=ec, linewidth=lw)
        for xi, v, lab in zip(x, vals, labels, strict=True):
            lift = CHECK1B_STACK_LIFT.get(lab)
            txt = f"+{lift[CHECK1B_METRICS.index(metric)]:.3f}" if lift else f"{v:.3f}"
            ax.annotate(
                txt, (xi, v + vals.max() * 0.03), ha="center", va="bottom", fontsize=8,
                fontweight="bold" if lift else "normal",
                color=STACK_RED if lift else "black",
            )
        ax.set_xticks(x, labels, fontsize=7.5)
        ax.set_ylim(0, vals.max() * 1.28)
        ax.set_title(metric, fontsize=9.5, fontweight="bold")
    axes[0].set_ylabel("point estimate\n(red label = stack's null-corrected lift)", fontsize=8)


def panel_toxic_share(ax) -> None:
    """Top-1 pick concentration: truth, potency prior, and every representation (2026-08-19 --
    see TOXIC_SHARE_BY_REPR's own comment for how the model bars were computed). top-3 omitted:
    saturates near 1.0 for everyone (truth included), no separating information."""
    labels = ["truth", "prior", *TOXIC_SHARE_BY_REPR]
    x = np.arange(len(labels))
    vals = [TOXIC_SHARE_OBSERVED[1][0], TOXIC_SHARE_PRIOR[1]] + [
        TOXIC_SHARE_BY_REPR[lab][0] for lab in TOXIC_SHARE_BY_REPR
    ]
    distinct = [DISTINCT_TOP1_OBSERVED[0], 1] + [
        TOXIC_SHARE_BY_REPR[lab][1] for lab in TOXIC_SHARE_BY_REPR
    ]
    ec, lw = _edges(labels)
    colors = [BLUE, ORANGE, *([BLUE] * len(TOXIC_SHARE_BY_REPR))]
    err_lo = TOXIC_SHARE_OBSERVED[1][0] - TOXIC_SHARE_OBSERVED[1][1]
    err_hi = TOXIC_SHARE_OBSERVED[1][2] - TOXIC_SHARE_OBSERVED[1][0]
    yerr = np.zeros((2, len(labels)))
    yerr[0, 0], yerr[1, 0] = err_lo, err_hi
    ax.bar(x, vals, 0.62, color=colors, edgecolor=ec, linewidth=lw, yerr=yerr, capsize=4)
    for xi, v, n in zip(x, vals, distinct, strict=True):
        ax.annotate(f"{v:.2f}", (xi, v + 0.05), ha="center", fontsize=8.5, fontweight="bold")
        ax.annotate(f"n={n:.0f}", (xi, -0.10), ha="center", fontsize=7.5, color="#555555")
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylim(-0.18, 1.28)
    ax.set_ylabel("share of top-1 picks that are pan-toxic")
    ax.set_title("Are the top picks just the pan-toxic drugs?", fontweight="bold")
    ax.set_xlabel(
        "n = distinct drugs ever picked #1 (of 26) -- the separating view once the top-1 share "
        "itself saturates near 1.0.\nExpr/stack show more diverse picks (n=5-7) than the "
        "regression baselines (n=3-4), which lean almost purely on potency.",
        fontsize=8.5, color="#444444", labelpad=18,
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

    fig = plt.figure(figsize=(17.8, 13.4), dpi=150)
    # A dedicated thin header row (index 1) reserves its own geometric space for the "Check 1b"
    # label, rather than a fig.text() at a guessed y-fraction that tight_layout's later re-flow
    # would silently invalidate -- row 0/2/3 hold the real panels, row 1 is header-only.
    grid = fig.add_gridspec(4, 4, height_ratios=(1.0, 0.08, 1.2, 0.8), hspace=0.55, wspace=0.32)
    axes = {
        "check1": fig.add_subplot(grid[0, 0:2]),
        "toxic": fig.add_subplot(grid[0, 2:4]),
        "check2": fig.add_subplot(grid[2, 0:2]),
        "gap": fig.add_subplot(grid[2, 2:4]),
        "check1b": [fig.add_subplot(grid[3, i]) for i in range(4)],
    }
    header_ax = fig.add_subplot(grid[1, :])
    header_ax.axis("off")
    header_ax.text(
        0.0, 0.0,
        "Check 1b (2026-08-19) -- DE-restricted metrics vs ground-truth Wilcoxon DE calls; "
        "both stack checkpoints permutation-significant (p < 0.005) beyond every baseline",
        fontsize=10.5, fontweight="bold", ha="left", va="bottom", transform=header_ax.transAxes,
    )
    panel_check1(axes["check1"])
    panel_toxic_share(axes["toxic"])
    panel_check2(axes["check2"])
    panel_selection_gap(axes["gap"])
    panel_check1b(axes["check1b"])

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
