"""Figures for every ladder rung, built ONLY from promoted artifacts.

Reading from docs/results/ rather than from a run directory is the point: a figure sourced from
a scratch file can show a number that no longer exists anywhere checkable, and this project has
already had numbers live in prose with nothing behind them. Every panel here traces to a CSV
with a provenance sidecar naming the job that produced it, and each figure prints its sources.

A rung with no promoted result is drawn as an explicit "not yet available" panel rather than
skipped, so a missing rung is visible in the figure instead of being absent from it. The same
principle extends to the ladder summary: a rung whose ratio is not yet a fair comparison (an
open invariant mismatch, not a missing result) is drawn as a labelled "blocked" bar rather than
silently omitted or faked.

Color follows one fixed identity, everywhere in this file: baseline blue, Stack/model red,
reference green, control grey (recedes -- it's what a result has to beat, not a result), ceiling
near-black. Every source-identity color is validated against the dataviz skill's CVD/contrast
checks (fixed hues #2a78d6 / #e34948 / #008300, #52514e for the muted control role).
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
    "control": "#52514e",
    "baseline": "#2a78d6",
    "model": "#e34948",
    "reference": "#008300",
    "ceiling": "#0b0b0b",
    "blocked": "#c3c2b7",
}

# Fixed categorical order for the arm/quantity dimension (rung 2's arms, rung 0's three bars) --
# a SEPARATE identity axis from source-kind above, so it gets its own fixed, validated slots
# rather than reusing baseline/model/reference (those mean something different here).
ARM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


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


def _legend_once(ax, **kw) -> None:
    """De-duplicate repeated labels (e.g. from a per-row loop) before drawing the legend."""
    h, l = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for hi, li in zip(h, l, strict=True):
        seen.setdefault(li, hi)
    if seen:
        ax.legend(list(seen.values()), list(seen.keys()), **kw)


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
        bars = {"mismatched-pair null\n(the floor)": nullv,
                "split-half\n(half the data)": med,
                "Spearman-Brown\n(the ceiling)": sb}
        cols = [PALETTE["control"], PALETTE["baseline"], PALETTE["ceiling"]]
        ax.barh(list(bars), list(bars.values()), color=cols)
        for i, v in enumerate(bars.values()):
            if np.isfinite(v):
                ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
        ax.set_xlabel("Pearson r between plate halves")
        p = r.get("p_vs_null", "")
        ax.set_title(f"Rung 0 — replicate ceiling  (n={int(r.get('n_pairs', 0))} pairs, "
                     f"result vs null p={p})", fontsize=11, loc="left")
        ax.axvline(0, color="#cccccc", lw=0.8)
        ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(out / "rung0_ceiling.png", dpi=200)
    plt.close(fig)
    return f"rung0_ceiling.png (job {job})"


def fig_rung1(results: Path, out: Path) -> str:
    """Check-1b: per-source DE fidelity vs its null. within_drug is the real test; shuffle_all
    is a floor every source already clears (recorded as uninformative in the write-up), so it's
    drawn as a thin reference tick rather than a full bar competing for attention with the
    result that matters."""
    df, job = load(results, "rung1_de_permutation_null")
    if df is None:
        df, job = load(results, "de_permutation_null_both_checkpoints")
    fig, ax = plt.subplots(figsize=(9, 5))
    if df is None or df.empty or "null_kind" not in df.columns:
        _unavailable(ax, "Rung 1 — held-out line, DE fidelity", job or "no null table")
    else:
        piv = df.pivot_table(index="source", columns="null_kind", values="p", aggfunc="mean")
        wd_col = next((c for c in piv.columns if "within" in str(c)), piv.columns[-1])
        piv = piv.sort_values(wd_col, ascending=False)  # worst (highest p) at top -> best at bottom
        y = np.arange(len(piv))
        colors = [PALETTE[kind_of(s)] for s in piv.index]
        ax.barh(y, piv[wd_col], height=0.62, color=colors, zorder=3)
        for c in piv.columns:
            if c == wd_col:
                continue
            ax.scatter(piv[c], y, marker="|", s=140, color=PALETTE["control"],
                       linewidth=1.6, zorder=4, label=str(c).replace("_", " ") + " (floor)")
        for i, v in enumerate(piv[wd_col]):
            ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=8.5)
        ax.set_yticks(y)
        ax.set_yticklabels(piv.index)
        ax.axvline(0.05, color=PALETTE["ceiling"], ls="--", lw=1, label="p = 0.05", zorder=2)
        ax.set_xlabel("permutation p vs within_drug null  (lower = clears it, i.e. LINE-specific)")
        ax.set_title("Rung 1 — DE fidelity: models (red) and reference (green) vs baselines (blue)",
                     fontsize=11, loc="left")
        ax.set_xlim(0, max(0.08, float(piv[wd_col].max()) * 1.25))
        _legend_once(ax, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "rung1_de_null.png", dpi=200)
    plt.close(fig)
    return f"rung1_de_null.png (job {job})"


def fig_rung2(results: Path, out: Path) -> str:
    """Transfer penalty: each source under each arm, and the difference."""
    df, job = load(results, "rung2_transfer_penalty")
    grid, gjob = load(results, "rung2_grid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    if grid is None or grid.empty:
        _unavailable(axes[0], "Rung 2 — arms", gjob)
        _unavailable(axes[1], "Rung 2 — transfer penalty", gjob)
    else:
        piv = grid.pivot_table(index="source", columns="arm", values="mean_rho")
        order = piv["in_platform"].sort_values(ascending=True).index if "in_platform" in piv else piv.index
        piv = piv.loc[order]
        arm_order = [a for a in ("in_platform", "cross_platform", "bulk_target") if a in piv.columns]
        piv[arm_order].plot.barh(ax=axes[0], width=0.78, color=ARM_COLORS[: len(arm_order)])
        for i, src in enumerate(piv.index):
            axes[0].scatter([], [])  # keep color cycle untouched
            if kind_of(src) == "control":
                axes[0].axhspan(i - 0.45, i + 0.45, color="#00000006", zorder=0)
        axes[0].axvline(0, color="#cccccc", lw=0.8)
        axes[0].set_xlabel("mean Spearman rho vs Tahoe truth")
        axes[0].set_title("Rung 2 — each source, in-platform vs transferred", fontsize=11, loc="left")
        axes[0].legend(fontsize=8, loc="lower right")
        axes[0].set_ylabel("")
        # controls (grey) recede via a shaded row-band rather than color, since arm identity
        # already owns the color channel in this panel.
        for lbl in axes[0].get_yticklabels():
            if kind_of(lbl.get_text()) == "control":
                lbl.set_color(PALETTE["control"])
                lbl.set_style("italic")

        if df is not None and "transfer_penalty" in df.columns:
            d = df.dropna(subset=["transfer_penalty"]).set_index("source").loc[
                [s for s in order if s in df["source"].to_numpy()]
            ].reset_index()
            colors = [PALETTE[kind_of(s)] for s in d["source"]]
            axes[1].barh(d["source"], d["transfer_penalty"], color=colors)
            for i, v in enumerate(d["transfer_penalty"]):
                axes[1].text(v - 0.012 if v < 0 else v + 0.012, i, f"{v:+.2f}",
                             va="center", ha="right" if v < 0 else "left", fontsize=8.5)
            axes[1].axvline(0, color="#cccccc", lw=0.8)
            axes[1].margins(x=0.18)
            if "shuffled" in d["source"].to_numpy():
                sv = float(d.loc[d["source"] == "shuffled", "transfer_penalty"].iloc[0])
                axes[1].axvline(sv, color=PALETTE["control"], ls=":", lw=1.2,
                                 label="shuffled (wrong-line) control")
            axes[1].set_xlabel("cross_platform − in_platform  (negative = transfer costs)")
            axes[1].set_title("Rung 2 — transfer penalty per source", fontsize=11, loc="left")
            axes[1].set_ylabel("")
            for lbl in axes[1].get_yticklabels():
                if kind_of(lbl.get_text()) == "control":
                    lbl.set_color(PALETTE["control"])
                    lbl.set_style("italic")
            _legend_once(axes[1], fontsize=8, loc="lower left")
        else:
            _unavailable(axes[1], "Rung 2 — transfer penalty", "penalty table absent")
    fig.tight_layout()
    fig.savefig(out / "rung2_transfer.png", dpi=200)
    plt.close(fig)
    return f"rung2_transfer.png (jobs {gjob}/{job})"


def fig_rung3(results: Path, out: Path, ceiling: float | None) -> tuple[str, dict | None]:
    """Check 2: interaction by representation, controls marked, ceiling annotated (not
    stretching the axis to it -- every real point sits under 0.15, so an axis reaching 0.457
    would spend 70% of the panel on empty space). Returns the (source, interaction, method)
    of the row that actually clears its declared-variant significance test, if any, so the
    ladder summary can cite the SAME row rather than an unfiltered max."""
    df, job = load(results, "rung3_check2_grid")
    variants, _ = load(results, "rung3_declared_variants")
    fig, ax = plt.subplots(figsize=(9, 6.4))
    winner = None
    if df is None or df.empty or "interaction" not in df.columns:
        _unavailable(ax, "Rung 3 — GDSC2 viability", job or "grid not promoted")
    else:
        order = (df.groupby("source")["interaction"].median().sort_values().index.tolist())
        ypos = {s: i for i, s in enumerate(order)}
        markers = {"l2": "o", "l1": "s", "en": "^"}

        sig_keys: set[tuple[str, str]] = set()
        if variants is not None and "clears_bonferroni" in variants.columns:
            sig = variants[variants["clears_bonferroni"].astype(str).str.lower() == "yes"]
            sig_keys = set(zip(sig["source"].astype(str), sig["method"].astype(str), strict=True))
            # The winner must be a REAL result (baseline/model/reference), not a control: the
            # planted positive control clears Bonferroni by construction (it is built to), and
            # its 0.726 is not bounded by the screen-agreement ceiling the way a real
            # cell-line-derived score is -- reporting it as "the rung-3 result" gave a ladder
            # bar at 1.59x the ceiling, which cannot be a real fraction of achievable.
            sig_real = sig[sig["source"].astype(str).apply(lambda s: kind_of(s) != "control")]
            if len(sig_real):
                row = sig_real.sort_values("interaction", ascending=False).iloc[0]
                winner = {"source": str(row["source"]), "method": str(row["method"]),
                          "interaction": float(row["interaction"])}

        # planted is a synthetic positive control BUILT to have a huge effect size (it must
        # clear its null, that's the point) -- including it when sizing the axis stretches the
        # whole panel to fit one point 5-10x any real row, recreating the wasted-space problem
        # the ceiling annotation already exists to avoid. It gets the same treatment: clipped
        # to the axis edge with an off-scale arrow, not a wider axis.
        real_mask = df["source"].astype(str).str.lower() != "planted"
        data_max = float(df.loc[real_mask, "interaction"].max())
        data_min = float(df.loc[real_mask, "interaction"].min())
        xhi = max(0.05, data_max * 1.35)
        xlo = min(-0.02, data_min * 1.2)

        for meth, grp in df.groupby("method"):
            colors = [PALETTE[kind_of(s)] for s in grp["source"]]
            is_sig = [(str(s), str(meth)) in sig_keys for s in grp["source"]]
            alphas = [1.0 if s else 0.55 for s in is_sig]
            for (_, rrow), col, al, sig in zip(grp.iterrows(), colors, alphas, is_sig, strict=True):
                x = rrow["interaction"]
                off_scale = str(rrow["source"]).lower() == "planted" and x > xhi
                ax.scatter(min(x, xhi * 0.985) if off_scale else x, ypos[rrow["source"]],
                           marker=markers.get(str(meth), "o"),
                           s=140 if sig else 46,
                           color=col, alpha=al,
                           edgecolor=PALETTE["ceiling"] if sig else "white",
                           linewidth=1.4 if sig else 0.5,
                           zorder=5 if sig else 3,
                           label=str(meth))
        if "null_p95" in df.columns:
            nl = df.groupby("source")["null_p95"].max()
            ax.scatter([min(nl[s], xhi * 0.985) for s in order], range(len(order)), marker="|", s=110,
                       color=PALETTE["control"], label="null p95 (floor)", zorder=2)
        ax.set_yticks(range(len(order)))
        labels = ax.set_yticklabels(order, fontsize=8.5)
        for lbl in labels:
            if kind_of(lbl.get_text()) == "control":
                lbl.set_color(PALETTE["control"])
                lbl.set_style("italic")
            elif kind_of(lbl.get_text()) == "model":
                lbl.set_color(PALETTE["model"])
                lbl.set_fontweight("bold")

        ax.set_xlim(xlo, xhi)
        ax.axvline(0, color="#cccccc", lw=0.8, zorder=1)
        if "planted" in ypos:
            pv = float(df.loc[df["source"].astype(str).str.lower() == "planted", "interaction"].max())
            ax.annotate(f"planted (positive control): {pv:.2f}, off-scale", xy=(xhi * 0.985, ypos["planted"]),
                        xytext=(xhi * 0.6, ypos["planted"] - 1.4), fontsize=8, color=PALETTE["control"],
                        arrowprops=dict(arrowstyle="->", color=PALETTE["control"], lw=0.9))
        if ceiling and np.isfinite(ceiling):
            ax.annotate(f"screen-agreement ceiling {ceiling:.3f} →", xy=(xhi, len(order) - 1),
                        xytext=(xhi * 0.98, len(order) - 0.6), fontsize=8.5, ha="right",
                        color=PALETTE["ceiling"])
        if winner:
            ax.annotate(f"only row clearing Bonferroni:\n{winner['source']}/{winner['method']} "
                        f"= {winner['interaction']:.3f}",
                        xy=(winner["interaction"], ypos[winner["source"]]),
                        xytext=(xhi * 0.5, ypos[winner["source"]] + 3.2),
                        fontsize=8.5, color=PALETTE["ceiling"],
                        arrowprops=dict(arrowstyle="->", color=PALETTE["ceiling"], lw=1))
        ax.set_xlabel("interaction rho (cell-line-specific drug response)")
        ax.set_title("Rung 3 — GDSC2 viability: bold marker = clears Bonferroni across 24 tests",
                     fontsize=11, loc="left")
        _legend_once(ax, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "rung3_check2.png", dpi=200)
    plt.close(fig)
    return f"rung3_check2.png (job {job})", winner


def fig_rung4(results: Path, out: Path) -> str:
    """Rung 4: organoid viability, source x method, as a fraction of rung 3's own numbers
    (the protocol's rule -- rung 4 has no ceiling of its own, so every value here is reported
    relative to what the SAME method achieves on cell lines, never as an absolute)."""
    df, job = load(results, "rung4_viability")
    fig, ax = plt.subplots(figsize=(8, 5))
    if df is None or df.empty:
        _unavailable(ax, "Rung 4 — organoid viability (embargoed)",
                      job or "frozen / not yet unfrozen this round")
    else:
        order = (df.groupby("source")["interaction"].median().sort_values().index.tolist())
        ypos = {s: i for i, s in enumerate(order)}
        markers = {"l2": "o", "l1": "s", "en": "^"}
        for meth, grp in df.groupby("method"):
            colors = [PALETTE[kind_of(s)] for s in grp["source"]]
            ax.scatter(grp["interaction"], [ypos[s] for s in grp["source"]],
                       marker=markers.get(str(meth), "o"), s=60, color=colors,
                       edgecolor="white", linewidth=0.6, label=str(meth), zorder=3)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=8.5)
        ax.axvline(0, color="#cccccc", lw=0.8)
        ax.set_xlabel("interaction rho (patient-specific drug response, embargoed cohort)")
        ax.set_title("Rung 4 — organoid viability; no rung of its own ceiling, see rung 3",
                     fontsize=11, loc="left")
        _legend_once(ax, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "rung4_viability.png", dpi=200)
    plt.close(fig)
    return f"rung4_viability.png (job {job})"


def fig_ladder(rows: list[dict]) -> plt.Figure:
    """The ladder itself: every rung, real ratio where one is valid, an explicit labelled
    'blocked' bar where it is not (an open invariant mismatch, not a missing result) -- the
    single figure a reader should be able to look at alone and see the whole shape of the
    collapse, plus exactly which comparisons are not trustworthy yet and why.

    The bar axis (0-1) and the reason text live in two visually separate columns -- text
    placed just past a bar's own end overlaps neighbours of very different lengths (a 1.0
    reference bar, a near-zero blocked bar, a real 0.3 result), so every reason is instead
    left-aligned at one fixed column to the right of the bars, wrapped to fit it.
    """
    import textwrap

    n = len(rows)
    fig, (ax, axt) = plt.subplots(
        1, 2, figsize=(13, 1.05 * n + 1.6), gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.05}
    )
    y = np.arange(n)
    for i, r in enumerate(rows):
        if r["status"] == "blocked":
            ax.barh(i, 0.015, color=PALETTE["blocked"], height=0.55)
            head = "blocked"
        elif r["status"] == "reference":
            ax.barh(i, 1.0, color=PALETTE["ceiling"], height=0.55, alpha=0.12)
            ax.scatter([1.0], [i], marker="|", s=200, color=PALETTE["ceiling"], zorder=3)
            head = "is the ceiling"
        else:
            color = PALETTE["model"] if r["value"] < 0.35 else PALETTE["baseline"]
            ax.barh(i, min(r["value"], 1.12), color=color, height=0.55)
            ax.text(min(r["value"], 1.12) + 0.02, i, f"{r['value']:.2f}", va="center",
                    fontsize=9.5, fontweight="bold", color=color)
            head = f"{r['value']:.2f} {r.get('label', 'of ceiling')}"

        axt.text(0.0, i, head, va="center", fontsize=9, fontweight="bold",
                  color=(PALETTE["ceiling"] if r["status"] != "blocked" else "#7a7a76"))
        wrapped = textwrap.fill(r["reason"], width=54)
        axt.text(0.0, i + 0.30, wrapped, va="top", fontsize=8, color="#52514e",
                  style="italic" if r["status"] == "blocked" else "normal", linespacing=1.35)

    ax.set_yticks(y)
    ax.set_yticklabels([r["rung"] for r in rows], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_ylim(n - 0.5, -0.5)
    ax.axvline(1.0, color=PALETTE["ceiling"], ls="--", lw=1)
    ax.set_xlabel("best result / that rung's ceiling")
    ax.set_title("Transfer ladder — how far prediction survives, per rung", fontsize=12, loc="left")

    axt.set_xlim(0, 1)
    axt.set_ylim(n - 0.5, -0.5)
    axt.axis("off")

    fig.tight_layout()
    return fig


def _rung1_ladder_row(results: Path) -> dict:
    df, _ = load(results, "rung1_check1_fidelity")
    if df is None or df.empty:
        return {"rung": "1 held-out line", "status": "blocked", "reason": "not promoted"}
    real = df[~df["source"].astype(str).str.lower().isin(["measured_delta"])]
    best = float(real["r"].max()) if "r" in real.columns else float("nan")
    return {
        "rung": "1 held-out line",
        "status": "blocked",
        "reason": (f"best r={best:.2f}, but rung 0's ceiling is median-aggregated "
                   "and rung 1 is mean-aggregated (open invariant-3 mismatch)"),
    }


def _rung2_ladder_row(results: Path) -> dict:
    grid, _ = load(results, "rung2_grid")
    if grid is None or grid.empty:
        return {"rung": "2 cross-platform", "status": "blocked", "reason": "not promoted"}
    real = grid[~grid["source"].astype(str).apply(lambda s: kind_of(s) == "control")]
    if "in_platform" not in real["arm"].to_numpy() or real.empty:
        return {"rung": "2 cross-platform", "status": "blocked", "reason": "no in_platform arm"}
    piv = real.pivot_table(index="source", columns="arm", values="mean_rho")
    if "cross_platform" not in piv.columns or "in_platform" not in piv.columns:
        return {"rung": "2 cross-platform", "status": "blocked", "reason": "arms incomplete"}
    retained = (piv["cross_platform"] / piv["in_platform"]).replace([np.inf, -np.inf], np.nan).dropna()
    if retained.empty:
        return {"rung": "2 cross-platform", "status": "blocked", "reason": "no comparable baseline"}
    best_src = retained.idxmax()
    frac = float(max(0.0, retained[best_src]))
    return {
        "rung": "2 cross-platform",
        "status": "value",
        "value": frac,
        "label": "retained (cross/in-platform, NOT vs rung 0)",
        "reason": f"{best_src}: cross_platform/in_platform. Not a fraction of rung 0's "
                  "ceiling -- that comparison is separately blocked (see rung 1).",
    }


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

    r3_fig, r3_winner = fig_rung3(args.results_dir, args.out_dir, ceiling)
    made = [
        fig_rung0(args.results_dir, args.out_dir),
        fig_rung1(args.results_dir, args.out_dir),
        fig_rung2(args.results_dir, args.out_dir),
        r3_fig,
        fig_rung4(args.results_dir, args.out_dir),
    ]

    rows: list[dict] = []
    r0, _ = load(args.results_dir, "rung0_delta_reproducibility")
    if r0 is not None and not r0.empty:
        sb = float(r0.iloc[0].get("spearman_brown_full", np.nan))
        rows.append({"rung": "0 replicate ceiling", "status": "reference",
                     "reason": f"IS the ceiling (Spearman-Brown {sb:.3f})"})
    else:
        rows.append({"rung": "0 replicate ceiling", "status": "blocked", "reason": "not promoted"})

    rows.append(_rung1_ladder_row(args.results_dir))
    rows.append(_rung2_ladder_row(args.results_dir))

    if r3_winner and ceiling:
        rows.append({"rung": "3 GDSC2", "status": "value",
                     "value": r3_winner["interaction"] / ceiling,
                     "reason": f"{r3_winner['source']}/{r3_winner['method']}, "
                               "the ONLY row clearing Bonferroni"})
    elif ceiling:
        rows.append({"rung": "3 GDSC2", "status": "blocked",
                     "reason": "no row clears Bonferroni across the declared variants"})
    else:
        rows.append({"rung": "3 GDSC2", "status": "blocked", "reason": "ceiling not promoted"})

    r4, _ = load(args.results_dir, "rung4_viability")
    if r4 is not None and not r4.empty and r3_winner:
        real4 = r4[~r4["source"].astype(str).apply(lambda s: kind_of(s) == "control")]
        best4 = float(real4["interaction"].max()) if len(real4) else float("nan")
        if np.isfinite(best4) and r3_winner["interaction"]:
            rows.append({"rung": "4 organoids", "status": "value",
                         "value": max(0.0, best4 / r3_winner["interaction"]),
                         "label": f"of rung 3's {r3_winner['source']} (organoids have no ceiling of their own)",
                         "reason": f"best organoid score as a fraction of rung 3's {r3_winner['source']}"})
        else:
            rows.append({"rung": "4 organoids", "status": "blocked", "reason": "no comparable score"})
    else:
        rows.append({"rung": "4 organoids", "status": "blocked",
                     "reason": "frozen / embargoed holdout, not yet unfrozen this round"})

    ladder_fig = fig_ladder(rows)
    ladder_fig.savefig(args.out_dir / "ladder_summary.png", dpi=200)
    plt.close(ladder_fig)
    made.append("ladder_summary.png")
    pd.DataFrame(rows).to_csv(args.out_dir / "ladder_summary.csv", index=False)

    print("figures written to", args.out_dir)
    for m in made:
        print("  ", m)


if __name__ == "__main__":
    main()
