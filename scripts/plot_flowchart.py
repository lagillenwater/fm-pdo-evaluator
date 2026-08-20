"""Draw the transcript -> viability evaluation flowchart as a clean, editable SVG.

The output is hand-composed SVG (every box is a <rect>, every label an editable
<text>, every connector a <line> + arrowhead <polygon>), so it opens in Adobe
Illustrator with text and shapes fully editable -- no flattening, no font outlines.
Text is kept pure-ASCII (see ``_ASCII``): Illustrator's SVG import drops non-ASCII
glyphs, so "delta"/"->"/"x" are spelled out rather than typeset as symbols.

    python scripts/plot_flowchart.py            # -> docs/figures/prediction_flowchart.svg

The diagram is split into the two cohorts the harness actually keeps separate:

  TRAIN band -- the supervised readout (l1, l2 CV-tuned penalized regression) is FIT
  on the cell-line cohort: real perturbation deltas (L1000 / Tahoe) paired with GDSC2
  AUC. The fit readout is then FROZEN. hallmark is a fixed signature and skips
  training entirely.

  TEST band -- the frozen readout is APPLIED to held-out deltas from every delta
  source (additive / learned / k-NN / Stack), turned into a predicted sensitivity,
  and scored against a DIFFERENT viability screen (Soragni organoid AUC, or held-out
  GDSC2 AUC). Train and test viability are different labels on different samples, so
  the supervised readout is never fit and scored on the same data.
"""

from __future__ import annotations

import math
from pathlib import Path

# ---- palette -----------------------------------------------------------------
INK = "#2b2b2b"
ARROW = "#5a5a5a"
AUX = "#9a9a9a"  # dashed auxiliary wiring (labels, controls)
BRIDGE = "#7A57B5"  # the fit -> frozen -> applied readout, carried across the bands
PALETTE = {
    "input": ("#E8F1FB", "#4A78B5"),
    "delta": ("#E6F4EA", "#3C8C54"),
    "stack": ("#FDEBD0", "#D9892B"),  # the FM under test -- highlighted
    "readout": ("#EFE8FB", "#7A57B5"),  # the readout at every life stage (fit/frozen/applied)
    "pred": ("#FFFFFF", "#333333"),
    "metric": ("#E4F5F3", "#2E9C8E"),
    "control": ("#FBEAEA", "#C0504D"),
    "label": ("#F2F2F2", "#777777"),
}
FONT = "Helvetica, Arial, sans-serif"


class Node:
    def __init__(self, cx, cy, w, h, lines, kind, dashed_border=False):
        self.cx, self.cy, self.w, self.h = cx, cy, w, h
        self.lines, self.kind, self.dashed_border = lines, kind, dashed_border

    def left(self):
        return (self.cx - self.w / 2, self.cy)

    def right(self):
        return (self.cx + self.w / 2, self.cy)

    def top(self):
        return (self.cx, self.cy - self.h / 2)

    def bottom(self):
        return (self.cx, self.cy + self.h / 2)

    def svg(self):
        fill, stroke = PALETTE[self.kind]
        x, y = self.cx - self.w / 2, self.cy - self.h / 2
        sw = 2.2 if self.kind == "stack" else 1.5
        dash = ' stroke-dasharray="5 4"' if self.dashed_border else ""
        out = [
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{self.w:.1f}" height="{self.h:.1f}" '
            f'rx="9" ry="9" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>'
        ]
        lh = 15.0
        first = self.cy - (len(self.lines) - 1) * lh / 2 + 4
        spans = []
        for i, (txt, bold) in enumerate(self.lines):
            dy = 0 if i == 0 else lh
            weight = ' font-weight="bold"' if bold else ""
            size = 13.5 if bold else 11.5
            spans.append(
                f'<tspan x="{self.cx:.1f}" dy="{dy:.1f}"{weight} '
                f'font-size="{size}">{_esc(txt)}</tspan>'
            )
        out.append(
            f'<text x="{self.cx:.1f}" y="{first:.1f}" text-anchor="middle" '
            f'font-family="{FONT}" fill="{INK}">{"".join(spans)}</text>'
        )
        return "\n".join(out)


# Illustrator's SVG text import drops non-ASCII glyphs (delta, arrows, etc.), so keep
# the file pure-ASCII: every character then renders in any font Illustrator substitutes.
_ASCII = str.maketrans(
    {
        "Δ": "delta",  # Δ  treated-minus-control difference
        "δ": "delta",  # δ
        "→": "->",  # →
        "−": "-",  # − minus
        "—": "-",  # — em dash
        "×": "x",  # ×
        "·": " | ",  # ·
        "↑": "up",  # ↑
        "↓": "down",  # ↓
    }
)


def _esc(s: str) -> str:
    s = s.translate(_ASCII)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def arrow(p1, p2, color=ARROW, width=1.7, dashed=False, head=9.0):
    x1, y1 = p1
    x2, y2 = p2
    ang = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * math.cos(ang), y2 - head * math.sin(ang)
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    lx = bx + head * 0.5 * math.cos(ang + math.pi / 2)
    ly = by + head * 0.5 * math.sin(ang + math.pi / 2)
    rx = bx + head * 0.5 * math.cos(ang - math.pi / 2)
    ry = by + head * 0.5 * math.sin(ang - math.pi / 2)
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
        f'stroke="{color}" stroke-width="{width}"{dash}/>\n'
        f'<polygon points="{x2:.1f},{y2:.1f} {lx:.1f},{ly:.1f} {rx:.1f},{ry:.1f}" '
        f'fill="{color}"/>'
    )


def label(x, y, text, size=11, color=INK, anchor="middle", italic=False, bold=False):
    st = ' font-style="italic"' if italic else ""
    wt = ' font-weight="bold"' if bold else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{FONT}" '
        f'font-size="{size}" fill="{color}"{st}{wt}>{_esc(text)}</text>'
    )


def build() -> str:
    W, H = 1820, 910
    body: list[str] = []

    # ===== TRAIN band: fit the supervised readout, then freeze it =============
    realdelta = Node(240, 150, 300, 62, [("Real perturbation Δ", True),
                                         ("L1000 / Tahoe treated − DMSO", False),
                                         ("(cell lines)", False)], "input")
    trainvia = Node(240, 234, 300, 54, [("Training viability", True),
                                        ("GDSC2 AUC (cell-line screen)", False)], "label")
    fit = Node(685, 192, 330, 78, [("Fit supervised readout", True),
                                   ("l2: RidgeCV (dense)", False),
                                   ("l1: LassoCV (sparse)", False)], "readout")
    frozen = Node(1085, 192, 236, 58, [("FROZEN readout", True),
                                       ("Δ -> sensitivity", False)], "readout")
    TH_train = (455, 192)

    # ===== TEST band: apply the frozen readout to held-out deltas =============
    qbase = Node(240, 452, 300, 62, [("Held-out query baseline", True),
                                     ("patient tumor / held-out", False),
                                     ("cell-line DMSO (log-CPM)", False)], "input")
    pctx = Node(240, 602, 300, 54, [("Perturbation context", True),
                                    ("drug-treated cells (for Stack)", False)], "input")
    TH_in = (445, 527)

    deltas = [
        Node(635, 400, 292, 62, [("additive  (floor)", True), ("each drug's mean real Δ", False),
                                 ("no patient × drug interaction", False)], "delta"),
        Node(635, 500, 292, 62, [("learned  PCA / NMF", True), ("baseline → Δ residual (ridge)", False),
                                 ("drug-mean + sample correction", False)], "delta"),
        Node(635, 600, 292, 62, [("k-NN", True), ("mean real Δ of nearest-", False),
                                 ("baseline lines (cosine)", False)], "delta"),
        Node(635, 712, 292, 66, [("Stack-Aligned generation", True),
                                 ("in-context generated treated − baseline", False),
                                 ("[ foundation model under test ]", False)], "stack"),
    ]
    TH_ro = (880, 560)
    apply = Node(1085, 560, 250, 84, [("Apply readout", True), ("Δ -> sensitivity", False),
                                      ("frozen l1 / l2", False),
                                      ("+ hallmark (fixed)", False)], "readout")
    pred = Node(1400, 452, 250, 48, [("Predicted sensitivity", True),
                                     ("per (sample, drug)", False)], "pred")
    metrics = Node(1400, 642, 250, 108, [("Metrics", True), ("global / interaction /", False),
                                         ("within-drug Spearman", False),
                                         ("normalized regret@k", False),
                                         ("Δ-fidelity Pearson (check 1)", False)], "metric")
    testvia = Node(1400, 808, 310, 54, [("Test viability  (held out)", True),
                                        ("Soragni organoid AUC / held-out GDSC2 AUC", False)], "label")
    controls = [
        Node(1680, 432, 232, 58, [("negative control", True), ("within-drug perm null", False)],
             "control", dashed_border=True),
        Node(1680, 556, 232, 58, [("negative control", True), ("random gene-set", False)],
             "control", dashed_border=True),
        Node(1680, 680, 232, 58, [("positive control", True), ("planted interaction, recovered", False)],
             "control", dashed_border=True),
        Node(1680, 804, 232, 58, [("validation", True), ("readout gate on real Δ", False)],
             "control", dashed_border=True),
    ]

    # ----- band frame ---------------------------------------------------------
    body += [
        label(W / 2, 30, "fm-pdo-evaluator  —  transcript → viability evaluation",
              size=17, color=INK),
        f'<line x1="30" y1="300" x2="{W - 30}" y2="300" stroke="#cccccc" '
        f'stroke-width="1.2" stroke-dasharray="3 5"/>',
        label(38, 74, "TRAIN  —  fit the supervised readout   (cell-line cohort)",
              size=14, color="#555", anchor="start", bold=True),
        label(38, 340, "TEST  —  held-out evaluation", size=14, color="#555",
              anchor="start", bold=True),
    ]

    # ----- TRAIN wiring -------------------------------------------------------
    body += [
        arrow(realdelta.right(), TH_train),
        arrow(trainvia.right(), TH_train),
        arrow(TH_train, fit.left()),
        arrow(fit.right(), frozen.left()),
    ]

    # ----- the bridge: frozen readout drops into the apply node (train -> test)
    body.append(arrow(frozen.bottom(), apply.top(), color=BRIDGE, width=2.6))

    # ----- TEST wiring --------------------------------------------------------
    for n in (qbase, pctx):
        body.append(arrow(n.right(), TH_in))
    for n in deltas:
        body.append(arrow(TH_in, n.left()))
        body.append(arrow(n.right(), TH_ro))
    body += [
        arrow(TH_ro, apply.left()),
        arrow(apply.right(), pred.left()),
        arrow(pred.bottom(), metrics.top()),
        arrow(testvia.top(), metrics.bottom(), color=AUX, dashed=True),
    ]
    mx = metrics.cx + metrics.w / 2
    for n, my in zip(controls, (metrics.cy - 45, metrics.cy - 15, metrics.cy + 15,
                                metrics.cy + 45)):
        body.append(arrow(n.left(), (mx, my), color=AUX, dashed=True))

    # ----- boxes on top -------------------------------------------------------
    for n in [realdelta, trainvia, fit, frozen, qbase, pctx, *deltas, apply, pred,
              metrics, testvia, *controls]:
        body.append(n.svg())

    # ----- annotations --------------------------------------------------------
    body += [
        label(685, 250, "hallmark: unsupervised signature -- skips training",
              size=10, color=AUX, italic=True),
        label(1140, 380, "frozen readout applied to every test delta",
              size=10, color=BRIDGE, italic=True, anchor="start"),
        label(880, 528, "every source through the same frozen readout",
              size=10, color=INK, italic=True),
        label(635, 356,
              "additive / learned / k-NN reuse the train-cohort real Δ;  Stack generates from the query",
              size=10, color=AUX, italic=True),
        label(1412, 748, "measured y", size=10, color=AUX, anchor="start"),
        label(1680, 852, "null (neg) / recovery (pos) / gate (validation)",
              size=9.5, color=AUX, italic=True),
        label(30, H - 34,
              "Supervised readout is fit on the cell-line cohort, frozen, then applied to held-out "
              "test deltas -- train and test viability are different screens on different samples.",
              size=10.5, color="#666", anchor="start"),
        label(30, H - 18,
              "All Δ on log-CPM fold-change; each gene z-scored per cohort before the readout.  "
              "Purple = the readout (fit -> frozen -> applied).  Amber = FM under test.  Dashed = labels / controls.",
              size=10.5, color="#666", anchor="start"),
    ]

    inner = "\n".join(body)
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">\n'
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>\n'
        f"{inner}\n</svg>\n"
    )


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "docs" / "figures" / "prediction_flowchart.svg"
    out.write_text(build(), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
