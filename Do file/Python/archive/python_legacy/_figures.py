"""
Publication-quality figures for the MICS DDML Analysis.

Includes:
- Propensity score overlap plots
- Coefficient stability over progressive confounders
- RiskSource heterogeneity
"""

from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from _config import FIGURE_DIR, LEARNER_LABELS
from _style import use_paper_style, despine, PALETTE, PRISM_LIGHT, PRISM_BLUE, PRISM_TEAL


def plot_propensity_overlap_panel(
    records: List[Dict[str, Any]],
    title: str,
    filename: str,
    out_dir: Optional[Path] = None,
    trim: float = 0.01,
) -> Optional[Path]:
    """Direct overlap exhibit: the DDML cross-fitted propensity m(X) per treatment.

    ``records`` is a list of dicts, one per treatment, each:
        {"label": str, "m": 1d array of P(D=d|X), "d": 1d array 0/1 treatment}
    Draws one subplot per treatment — treated vs control propensity histograms,
    trimming bounds marked, and the share of observations pushed past the
    trimming threshold annotated.  An any-treatment subsample shows two
    overlapping clouds (good overlap); a single-method subsample where country FE
    predict adoption shows the treated mass piled at 1 and control at 0 (positivity
    failure).  Saved to the ROOT Figures/ directory.
    """
    use_paper_style()
    out_dir = Path(out_dir) if out_dir is not None else FIGURE_DIR
    recs = [r for r in records if r.get("m") is not None and len(r["m"])]
    if not recs:
        return None

    ncols = min(len(recs), len(recs) if len(recs) <= 3 else 3)
    nrows = (len(recs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for idx, r in enumerate(recs):
        ax = axes_flat[idx]
        m = np.asarray(r["m"], dtype=float)
        d = np.asarray(r["d"], dtype=int)
        m_t, m_c = m[d == 1], m[d == 0]
        bins = np.linspace(0, 1, 51)
        ax.hist(m_c, bins=bins, alpha=0.65, density=True, label="Control (no treat.)",
                color=PRISM_LIGHT, edgecolor="white", linewidth=0.3)
        ax.hist(m_t, bins=bins, alpha=0.65, density=True, label="Treated",
                color=PRISM_TEAL, edgecolor="white", linewidth=0.3)
        ax.axvline(trim, color="0.5", ls=":", lw=0.8)
        ax.axvline(1 - trim, color="0.5", ls=":", lw=0.8)
        # share of obs trimming would bite (propensity outside [trim, 1-trim]).
        share = float(np.mean((m < trim) | (m > 1 - trim))) if len(m) else 0.0
        ax.set_title(f"{r['label']} (trimmed {100*share:.1f}%)")
        ax.set_xlabel("Propensity $\\hat m(X)=P(D{=}d\\mid X)$")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.legend()
        despine(ax)

    for idx in range(len(recs), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Title omitted: the LaTeX float \caption supplies it.
    fig.tight_layout()
    png = out_dir / filename
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Propensity overlap figure saved to: {png}")
    return png


def plot_overlap_grid(
    panel_d: List[Dict[str, Any]],
    panel_y: List[Dict[str, Any]],
    filename: str,
    out_dir: Optional[Path] = None,
    trim: float = 0.01,
    xlabel_d: str = "Propensity $\\hat m(X)=P(D{=}d\\mid X)$",
    xlabel_y: str = "Outcome prediction $\\hat g(X)=\\hat E[Y\\mid X,D]$",
) -> Optional[Path]:
    """Combined overlap figure: propensity (top row) over outcome (bottom row).

    ``panel_d`` / ``panel_y`` are the per-treatment dicts
    ``{"label": str, "curves": {learner: 1d array}}`` for the D and Y models.
    Columns are treatments (shared across rows); each panel overlays one density
    line per learner.  No suptitle --- the LaTeX float \\caption supplies it.
    """
    use_paper_style()
    out_dir = Path(out_dir) if out_dir is not None else FIGURE_DIR
    panel_d = [p for p in panel_d if p.get("curves")]
    panel_y = [p for p in panel_y if p.get("curves")]
    if not panel_d:
        return None
    ncols = len(panel_d)

    fig, axes = plt.subplots(2, ncols, figsize=(3.1 * ncols, 5.4), squeeze=False)

    # stable colour per learner across the whole figure
    learners = []
    for p in panel_d:
        for k in p["curves"]:
            if k not in learners:
                learners.append(k)
    colour = {lk: PALETTE[i % len(PALETTE)] for i, lk in enumerate(learners)}
    bins = np.linspace(0, 1, 61)
    centers = 0.5 * (bins[:-1] + bins[1:])

    def _draw(ax, curves, show_trim):
        for lk, arr in curves.items():
            a = np.asarray(arr, dtype=float)
            a = a[np.isfinite(a)]
            if not len(a):
                continue
            dens, _ = np.histogram(a, bins=bins, density=True)
            ax.plot(centers, dens, lw=1.5, label=LEARNER_LABELS.get(lk, lk),
                    color=colour[lk], alpha=0.9)
        if show_trim:
            ax.axvline(trim, color="0.5", ls=":", lw=0.8)
            ax.axvline(1 - trim, color="0.5", ls=":", lw=0.8)
        ax.set_xlim(0, 1)
        despine(ax)

    for j, p in enumerate(panel_d):
        ax = axes[0][j]
        _draw(ax, p["curves"], show_trim=True)
        ax.set_title(p["label"])
        ax.set_xlabel(xlabel_d, fontsize=8)
        if j == 0:
            ax.set_ylabel("Density\n(propensity)")
    # second row aligned to the same treatment order
    by_label_y = {p["label"]: p for p in panel_y}
    for j, p in enumerate(panel_d):
        ax = axes[1][j]
        py = by_label_y.get(p["label"])
        if py is not None:
            _draw(ax, py["curves"], show_trim=True)
        else:
            ax.set_visible(False)
        ax.set_xlabel(xlabel_y, fontsize=8)
        if j == 0:
            ax.set_ylabel("Density\n(outcome)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 7),
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    png = out_dir / filename
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Combined overlap figure saved to: {png}")
    return png


def plot_cate_curves(
    result_df: pd.DataFrame,
    out_dir: Optional[Path] = None,
) -> List[Path]:
    """Plot smooth CATE(x) curves with pointwise + joint (uniform) bands.

    One PNG per (dataset_type, outcome, treatment, moderator) spline group, saved
    to the ROOT ``Figures/`` directory (``FIGURE_DIR``).

    Returns the list of written file paths.
    """
    use_paper_style()
    out_dir = Path(out_dir) if out_dir is not None else FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    spline = result_df[result_df.get("kind", "spline") == "spline"].copy()
    if spline.empty:
        print("plot_cate_curves: no spline CATE rows to plot.")
        return []

    base = PRISM_BLUE
    keys = ["dataset_type", "outcome", "treatment", "moderator"]
    written: List[Path] = []
    for key, grp in spline.groupby(keys, sort=False):
        dataset_type, outcome, treatment, moderator = key
        grp = grp.sort_values("x")
        x = grp["x"].to_numpy()
        cate = grp["cate"].to_numpy()

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        # Joint (uniform) band — wider, lighter
        ax.fill_between(
            x, grp["ci_lower_joint"], grp["ci_upper_joint"],
            color=base, alpha=0.13, lw=0, label="95% joint (uniform) band",
        )
        # Pointwise band — narrower, darker
        ax.fill_between(
            x, grp["ci_lower"], grp["ci_upper"],
            color=base, alpha=0.28, lw=0, label="95% pointwise band",
        )
        ax.plot(x, cate, color=base, lw=2.0, label="CATE")
        ax.axhline(0.0, color="0.45", lw=0.9, ls="--")

        mod_label = grp["moderator_label"].iloc[0] if "moderator_label" in grp else moderator
        trt_label = grp["treatment_label"].iloc[0] if "treatment_label" in grp else treatment
        ax.set_xlabel(str(mod_label))
        ax.set_ylabel("Conditional ATE")
        # Title omitted: the LaTeX float \caption supplies it.
        ax.legend(loc="best")
        despine(ax)
        fig.tight_layout()

        fname = f"cate_{dataset_type}_{outcome}_{treatment}_{moderator}.png"
        # sanitize path-hostile chars in moderator/treatment names
        fname = fname.replace("/", "-").replace(" ", "_")
        path = out_dir / fname
        fig.savefig(path)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        written.append(path)

    print(f"plot_cate_curves: wrote {len(written)} figure(s) to {out_dir}")
    return written
