"""
Shared publication figure style (GraphPad/Lancet house look) for all Python plots.

Pure matplotlib --- no seaborn dependency.  Import and call ``use_paper_style()``
once at the top of any figure routine, build axes as usual, then call
``despine(ax)`` on each axis before saving.  ``PALETTE`` (Okabe--Ito,
colour-blind safe) and ``MARKERS`` give consistent, distinct series styling.

Design choices, matched to journal figures:
  * white background, no grid;
  * only the left and bottom spines, drawn thin in near-black;
  * ticks pointing outward, on the left/bottom only;
  * serif text with Computer-Modern math, to sit beside LaTeX body type;
  * tight layout, 300 dpi, both PNG and PDF emitted by the callers.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Blue -> teal family matched to the journal bar figures (light blue / mid blue /
# teal), with two high-contrast accents at the end so a 7-series plot stays
# readable.  Learner order is (ols, lasso, ridge, enet, rf, xgb, stacked), so
# XGBoost -> orange and Stacked -> black: the two series that carry the overlap
# story stand out from the blue-teal regularised learners.
PALETTE = ["#BBD7EA", "#7FB0D4", "#3D7CB5", "#1C6E76",
           "#0B4F4A", "#E69F00", "#000000", "#9C5BA6"]

# Named single/dual-series colours, taken from the bar-figure palette.
PRISM_LIGHT = "#BBD7EA"   # light blue (e.g. control / saline arm)
PRISM_BLUE = "#3D7CB5"    # mid blue (primary single-series fill)
PRISM_TEAL = "#1C6E76"    # dark teal (e.g. treated / high-dose arm)

MARKERS = ["o", "s", "^", "D", "v", "<", ">", "P"]

_INK = "#222222"  # near-black for axes/text

PAPER_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.dpi": 120,

    "font.family": "serif",
    "font.size": 11,
    "mathtext.fontset": "cm",

    "axes.edgecolor": _INK,
    "axes.labelcolor": _INK,
    "text.color": _INK,
    "axes.linewidth": 0.9,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,

    "xtick.color": _INK,
    "ytick.color": _INK,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,

    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "patch.linewidth": 0.6,
}


def use_paper_style():
    """Apply the paper rcParams globally (idempotent)."""
    plt.rcParams.update(PAPER_RC)


def despine(ax, left=True, bottom=True):
    """Remove top/right spines; keep ticks only on the kept spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(top=False, right=False)
    if not left:
        ax.tick_params(left=False)
    if not bottom:
        ax.tick_params(bottom=False)


def palette(n):
    """First ``n`` palette colours, cycling if needed."""
    return [PALETTE[i % len(PALETTE)] for i in range(n)]
