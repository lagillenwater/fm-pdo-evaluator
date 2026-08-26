"""Figures for every ladder rung, built ONLY from promoted artifacts.

Reading from docs/results/ rather than from a run directory is the point: a figure sourced from
a scratch file can show a number that no longer exists anywhere checkable, and this project has
already had numbers live in prose with nothing behind them. Every panel here traces to a CSV
with a provenance sidecar naming the job that produced it, and each figure prints its sources.

A rung with no promoted result is drawn as an explicit "not yet available" panel rather than
skipped, so a missing rung is visible in the figure instead of being absent from it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = {
    "control": "#8c8c8c",
    "baseline": "#4c72b0",
    "model": "#c44e52",
    "reference": "#55a868",
    "ceiling": "#000000",
}


def kind_of(source: str) -> str:
    """Colour class for a source name, so controls never look like results."""
    s = str(source).lower()
    if s in ("prior", "planted", "shuffled") or s.endswith("_random"):
        return "control"
    if s.startswith("stack") or s in ("base", "aligned"):
        return "model"
    if s in ("measured_delta", "observed_delta"):
        return "reference"
    return "baseline"


def load(results: Path, name: str) -> tuple[pd.DataFrame | None, str]:
    """A promoted result plus the job id from its sidecar, or (None, why-not)."""
    f = results / f"{name}.csv"
    if not f.exists():
        return None, f"{name}.csv not promoted"
    side = results / f"{name}.provenance.json"
    job = ""
    if side.exists():
        try:
            job = str(json.loads(side.read_text()).get("job_id", ""))
        except Exception:
            job = ""
    return pd.read_csv(f), job


def _unavailable(ax, title: str, why: str) -> None:
    """Draw the absence rather than omitting the panel."""
    ax.text(0.5, 0.5, f"not yet available\n{why}", ha="center", va="center",
            fontsize=10, color="#999999", transform=ax.transAxes)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])


def fig_rung0(results: Path, out: Path) -> str:
    """Replicate ceiling: split-half against its mismatched-pair null."""
    df, job = load(results, "rung0_delta_reproducibility")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if df is None or df.empty:
        _unavailable(ax, "Rung 0 — replicate ceiling", job)
    else:
        r = df.iloc[0]
        med = float(r.get("splithalf_median_r", np.nan))
        sb = float(r.get("spearman_brown_full", np.nan))
        nullv = float(r.get("null_median_r", np.nan))
        bars = {"mismatched-pair null": nullv, "split-half (half data)": med,
                "Spearman-Brown (full data)": sb}
        cols = [PALETTE["control"], PALETTE["baseline"], PALETTE["ceiling"]]
        ax.barh(list(bars), list(bars.values()), color=cols)
        for i, v in enumerate(bars.values()):
            if np.isfinite(v):
                ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
        ax.set_xlabel("Pearson r between plate halves")
        p = r.get("p_vs_null", "")
        ax.set_title(f"Rung 0 — replicate ceiling  (n={int(r.get('n_pairs', 0))} pairs, "
                     f"p vs null {p})", fontsize=11, loc="left")
        ax.axvline(0, color="#cccccc", lw=0.8)
    fig.tight_layout()
    fig.savefig(out / "rung0_ceiling.png", dpi=200)
    plt.close(fig)
    return f"rung0_ceiling.png (job {job})"


def fig_rung1(results: Path, out: Path) -> str:
    """Check-1b: per-source p under both nulls, the within-drug one being the real test."""
    df, job = load(results, "rung1_de_permutation_null")
    if df is None:
        df, job = load(results, "de_permutation_null_both_checkpoints")
    fig, ax = plt.subplots(figsize=(9, 5))
    if df is None or df.empty or "null_kind" not in df.columns:
        _unavailable(ax, "Rung 1 — held-out line, DE fidelity", job or "no null table")
    else:
        piv = df.pivot_table(index="source", columns="null_kind", values="p", aggfunc="mean")
        piv = piv.sort_values(piv.columns[-1])
        y = np.arange(len(piv))
        h = 0.38
        for i, col in enumerate(piv.columns):
            ax.barh(y + (i - 0.5) * h, piv[col], height=h, label=col,
                    color=PALETTE["control"] if "all" in col else PALETTE["baseline"])
        ax.set_yticks(y)
        ax.set_yticklabels(piv.index)
        ax.axvline(0.05, color=PALETTE["model"], ls="--", lw=1, label="p = 0.05")
        ax.set_xlabel("permutation p  (lower = clears the null)")
        ax.set_title("Rung 1 — DE fidelity vs two nulls; within_drug tests LINE specificity",
                     fontsize=11, loc="left")
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "rung1_de_null.png", dpi=200)
    plt.close(fig)
    return f"rung1_de_null.png (job {job})"


def fig_rung2(results: Path, out: Path) -> str:
    """Transfer penalty: each source under each arm, and the difference."""
    df, job = load(results, "rung2_transfer_penalty")
    grid, gjob = load(results, "rung2_grid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    if grid is None or grid.empty:
        _unavailable(axes[0], "Rung 2 — arms", gjob)
        _unavailable(axes[1], "Rung 2 — transfer penalty", gjob)
    else:
        piv = grid.pivot_table(index="source", columns="arm", values="mean_rho")
        piv.plot.barh(ax=axes[0], width=0.8,
                      color=[PALETTE["baseline"], PALETTE["model"], PALETTE["reference"]])
        axes[0].axvline(0, color="#cccccc", lw=0.8)
        axes[0].set_xlabel("mean Spearman rho vs Tahoe truth")
        axes[0].set_title("Rung 2 — each source under each arm", fontsize=11, loc="left")
        axes[0].legend(fontsize=8)
        if df is not None and "transfer_penalty" in df.columns:
            d = df.dropna(subset=["transfer_penalty"]).sort_values("transfer_penalty")
            axes[1].barh(d["source"], d["transfer_penalty"],
                         color=[PALETTE[kind_of(s)] for s in d["source"]])
            axes[1].axvline(0, color="#cccccc", lw=0.8)
            axes[1].set_xlabel("cross_platform - in_platform")
            axes[1].set_title("Rung 2 — transfer penalty (negative = transfer costs)",
                              fontsize=11, loc="left")
        else:
            _unavailable(axes[1], "Rung 2 — transfer penalty", "penalty table absent")
    fig.tight_layout()
    fig.savefig(out / "rung2_transfer.png", dpi=200)
    plt.close(fig)
    return f"rung2_transfer.png (jobs {gjob}/{job})"


def fig_rung3(results: Path, out: Path, ceiling: float | None) -> str:
    """Check 2: interaction by representation, controls marked, ceiling drawn."""
    df, job = load(results, "rung3_check2_grid")
    fig, ax = plt.subplots(figsize=(9, 6))
    if df is None or df.empty or "interaction" not in df.columns:
        _unavailable(ax, "Rung 3 — GDSC2 viability", job or "grid not promoted")
    else:
        best = (df.sort_values("interaction", ascending=False)
                  .drop_duplicates("source").sort_values("interaction"))
        ax.barh(best["source"], best["interaction"],
                color=[PALETTE[kind_of(s)] for s in best["source"]])
        if "null_p95" in best.columns:
            ax.scatter(best["null_p95"], np.arange(len(best)), marker="|", s=90,
                       color="#333333", label="null p95", zorder=3)
        if ceiling and np.isfinite(ceiling):
            ax.axvline(ceiling, color=PALETTE["ceiling"], ls="--", lw=1.2,
                       label=f"screen-agreement ceiling {ceiling:.3f}")
        ax.axvline(0, color="#cccccc", lw=0.8)
        ax.set_xlabel("interaction rho (cell-line-specific drug response)")
        ax.set_title("Rung 3 — GDSC2 viability; grey = controls, red = Stack",
                     fontsize=11, loc="left")
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "rung3_check2.png", dpi=200)
    plt.close(fig)
    return f"rung3_check2.png (job {job})"


def fig_ladder(results: Path, out: Path, summary: pd.DataFrame) -> str:
    """The ladder itself: each rung as a fraction of its own ceiling."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    if summary.empty:
        _unavailable(ax, "Ladder summary", "no rung has both a score and a ceiling")
    else:
        ax.bar(summary["rung"], summary["fraction_of_ceiling"],
               color=[PALETTE["model"] if f < 0.25 else PALETTE["baseline"]
                      for f in summary["fraction_of_ceiling"]])
        for i, row in enumerate(summary.itertuples()):
            ax.text(i, row.fraction_of_ceiling + 0.02, f"{row.fraction_of_ceiling:.2f}",
                    ha="center", fontsize=9)
        ax.set_ylabel("best score / that rung's ceiling")
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color=PALETTE["ceiling"], ls="--", lw=1)
        ax.set_title("Transfer ladder — how far prediction survives, per rung",
                     fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out / "ladder_summary.png", dpi=200)
    plt.close(fig)
    return "ladder_summary.png"


def main() -> None:
    """Draw every rung and the ladder summary from promoted artifacts."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("docs/results"))
    ap.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lc, _ = load(args.results_dir, "rung3_label_ceiling")
    ceiling = None
    if lc is not None and "interaction" in lc.columns:
        ceiling = float(lc["interaction"].max())

    made = [
        fig_rung0(args.results_dir, args.out_dir),
        fig_rung1(args.results_dir, args.out_dir),
        fig_rung2(args.results_dir, args.out_dir),
        fig_rung3(args.results_dir, args.out_dir, ceiling),
    ]

    # Ladder summary needs a score AND a ceiling per rung; rungs lacking either are omitted from
    # the bar chart and named in the caption rather than silently dropped.
    rows = []
    r0, _ = load(args.results_dir, "rung0_delta_reproducibility")
    r3, _ = load(args.results_dir, "rung3_check2_grid")
    if r3 is not None and ceiling and "interaction" in r3.columns:
        real = r3[~r3["source"].astype(str).str.lower().isin(
            ["planted", "prior", "measured_delta", "observed_delta"])]
        real = real[~real["source"].astype(str).str.endswith("_random")]
        if len(real):
            rows.append({"rung": "3 GDSC2", "best": float(real["interaction"].max()),
                         "ceiling": ceiling,
                         "fraction_of_ceiling": float(real["interaction"].max()) / ceiling})
    summary = pd.DataFrame(rows)
    made.append(fig_ladder(args.results_dir, args.out_dir, summary))
    if len(summary):
        summary.to_csv(args.out_dir / "ladder_summary.csv", index=False)

    print("figures written to", args.out_dir)
    for m in made:
        print("  ", m)


if __name__ == "__main__":
    main()
