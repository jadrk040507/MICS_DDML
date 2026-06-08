"""
Publication-quality figures for the MICS DDML Analysis.

Includes:
- Propensity score overlap plots
- Coefficient stability over progressive confounders
- Stacking weight composition bars
- RiskSource heterogeneity
"""

from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config import FIGURE_DIR, LEARNER_LABELS, SUBGROUP_LABELS


def plot_overlap_from_results(
    results: List[Dict[str, Any]],
    save: bool = True,
) -> None:
    """
    Generate propensity score overlap plots.

    NOTE: Since we no longer store model objects in results,
    overlap plots must be generated during estimation by calling
    plot_overlap() in diagnostics.py directly.
    This function is kept as a placeholder for backward compatibility.
    """
    print("NOTE: Model objects are not stored in results. Use diagnostics.plot_overlap() directly.")
    print("      Or re-estimate and call plot_overlap() during the fitting step.")


def plot_coefficient_stability(
    stability_results: List[Dict[str, Any]],
    filename: str = "stability.png",
) -> Path:
    """
    Plot ATE across progressive confounder additions.

    Each panel is a different outcome × treatment combination.
    X-axis: confounder set (step name).
    Y-axis: ATE coefficient.
    Error bars: 95% CI.
    """
    if not stability_results:
        return None

    df = pd.DataFrame(stability_results)
    groups = df.groupby(["outcome", "treatment"])

    n_panels = len(groups)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if n_panels == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for idx, ((outcome, treatment), sub) in enumerate(groups):
        ax = axes_flat[idx]
        if "stability_step" not in sub.columns:
            continue
        steps = sub["stability_step"].values
        coefs = sub["coef"].values
        cis_lo = sub["ci_lower"].values
        cis_hi = sub["ci_upper"].values

        x = np.arange(len(steps))
        ax.errorbar(x, coefs, yerr=[coefs - cis_lo, cis_hi - coefs],
                    fmt="o-", capsize=3, color="steelblue", markersize=4)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(steps, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("ATE")
        ax.set_title(f"{outcome} | {treatment}", fontsize=9)

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Coefficient Stability: Progressive Addition of Confounders", fontsize=11)
    fig.tight_layout()

    filepath = FIGURE_DIR / filename
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    fig.savefig(filepath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_stacking_weights(
    results: List[Dict[str, Any]],
    filename: str = "stacking_weights.png",
) -> Path:
    """
    Plot stacking weights as a stacked bar chart.

    NOTE: Since we no longer store model objects in results,
    stacking weights must be extracted during estimation.
    This function is kept as a placeholder for backward compatibility.
    """
    print("NOTE: Model objects are not stored in results. Extract stacking weights during estimation.")
    print("      Use learners.get_stacking_weights() on the fitted model.")
    return None


def plot_risk_source_heterogeneity(
    subgroup_results: List[Dict[str, Any]],
    filename: str = "risk_source_heterogeneity.png",
) -> Path:
    """
    Plot ATE by RiskSource level with 95% CIs.

    Shows heterogeneity in treatment effects across source contamination.
    """
    if not subgroup_results:
        return None

    stacked = [r for r in subgroup_results if r["learner"] == "stacked"]
    if not stacked:
        return None

    rs_vals = [0, 1, 2]
    rs_labels = ["No Risk", "Moderate Risk", "Very High Risk"]

    outcomes = list(set(r["outcome"] for r in stacked))
    treatments = list(set(r["treatment"] for r in stacked))

    n_panels = len(outcomes) * len(treatments)
    fig, axes = plt.subplots(1, max(n_panels, 1), figsize=(5 * max(n_panels, 1), 4))
    if n_panels == 1:
        axes = [axes]

    panel = 0
    for outcome in outcomes:
        for treatment in treatments:
            ax = axes[panel]
            coefs = []
            ci_lo = []
            ci_hi = []
            labels_present = []

            for rs in rs_vals:
                match = [r for r in stacked
                         if r["outcome"] == outcome and r["treatment"] == treatment
                         and r.get("subgroup_val") == rs]
                if match:
                    coefs.append(match[0]["coef"])
                    ci_lo.append(match[0]["ci_lower"])
                    ci_hi.append(match[0]["ci_upper"])
                    labels_present.append(rs_labels[rs])

            if coefs:
                x = np.arange(len(coefs))
                ax.errorbar(x, coefs, yerr=[np.array(coefs) - np.array(ci_lo),
                                            np.array(ci_hi) - np.array(coefs)],
                            fmt="o", capsize=5, markersize=8, color="steelblue")
                ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(labels_present, fontsize=8)
                ax.set_ylabel("ATE")
                ax.set_title(f"{outcome} | {treatment}", fontsize=9)

            panel += 1

    fig.suptitle("Treatment Effect Heterogeneity by Source Water Risk", fontsize=11)
    fig.tight_layout()

    filepath = FIGURE_DIR / filename
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    fig.savefig(filepath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return filepath