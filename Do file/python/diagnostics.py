"""
Diagnostic tools for DDML estimation.

Propensity score overlap, covariate balance, first-stage fit diagnostics.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Any

from config import FIGURE_DIR, RANDOM_STATE, N_JOBS
from data import create_model_matrix
from learners import create_learners

from sklearn.base import clone
from sklearn.metrics import roc_auc_score, r2_score, log_loss


def compute_propensity_scores(
    dt: pd.DataFrame,
    treatment_var: str,
    confounder_groups: Optional[Dict[str, List[str]]] = None,
    include_source_ecoli: bool = False,
    include_child_controls: bool = False,
    include_robustness: bool = False,
    subgroup_var: Optional[str] = None,
    subgroup_val: Optional[Any] = None,
    dataset_type: str = "HH",
) -> Dict[str, np.ndarray]:
    """
    Compute propensity scores from all learners.

    Returns dict mapping learner name to predicted propensity scores.
    """
    from config import BASE_CONFOUNDERS

    if confounder_groups is None:
        confounder_groups = BASE_CONFOUNDERS

    dt_work = dt.copy()
    if subgroup_var is not None and subgroup_val is not None:
        dt_work = dt_work[dt_work[subgroup_var] == subgroup_val].copy()

    X, dt_work = create_model_matrix(
        dt_work, treatment_var,
        confounder_groups=confounder_groups,
        include_source_ecoli=include_source_ecoli,
        include_child_controls=include_child_controls,
        include_robustness=include_robustness,
    )

    complete_mask = (
        dt_work[treatment_var].notna() &
        ~np.isnan(X).any(axis=1) &
        ~np.isinf(X).any(axis=1)
    )
    dt_clean = dt_work[complete_mask].copy()
    X_clean = X[complete_mask.values]
    d = dt_clean[treatment_var].values.astype(float)

    learners = create_learners()
    scores = {"treatment": d}

    for name, learner in learners.items():
        m = clone(learner["m"])
        try:
            m.fit(X_clean, d)
            ps = m.predict_proba(X_clean)[:, 1] if hasattr(m, "predict_proba") else m.predict(X_clean)
            scores[name] = ps
        except Exception as e:
            print(f"    Warning: Could not fit {name} for propensity scores: {e}")

    return scores, dt_clean


def plot_overlap(
    dt: pd.DataFrame,
    outcome_var: str,
    treatment_var: str,
    confounder_groups: Optional[Dict[str, List[str]]] = None,
    include_source_ecoli: bool = False,
    include_child_controls: bool = False,
    subgroup_var: Optional[str] = None,
    subgroup_val: Optional[Any] = None,
    dataset_type: str = "HH",
    save: bool = True,
) -> None:
    """
    Plot propensity score overlap histograms for each learner.

    Shows distribution of predicted propensity scores by treatment status,
    which assesses the common support assumption.
    """
    if confounder_groups is None:
        from config import BASE_CONFOUNDERS
        confounder_groups = BASE_CONFOUNDERS

    sub_label = f" (RiskSource={subgroup_val})" if subgroup_val is not None else ""
    title_prefix = f"{dataset_type} | {treatment_var}{sub_label}"

    results, _ = compute_propensity_scores(
        dt, treatment_var, confounder_groups,
        include_source_ecoli=include_source_ecoli,
        include_child_controls=include_child_controls,
        subgroup_var=subgroup_var, subgroup_val=subgroup_val,
        dataset_type=dataset_type,
    )

    d = results["treatment"]
    learner_names = [k for k in results.keys() if k != "treatment"]
    n_learners = len(learner_names)

    if n_learners == 0:
        print("    No learners available for overlap plot")
        return

    ncols = min(3, n_learners)
    nrows = (n_learners + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))

    if n_learners == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for idx, name in enumerate(learner_names):
        ax = axes_flat[idx]
        ps = results[name]
        d_treated = ps[d == 1]
        d_untreated = ps[d == 0]

        ax.hist(d_untreated, bins=50, alpha=0.5, density=True, label="Untreated", color="steelblue")
        ax.hist(d_treated, bins=50, alpha=0.5, density=True, label="Treated", color="coral")
        ax.set_title(f"{name}", fontsize=10)
        ax.set_xlabel("Propensity Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    for idx in range(n_learners, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"Propensity Score Overlap: {title_prefix}", fontsize=12)
    fig.tight_layout()

    if save:
        fname = f"overlap_{dataset_type}_{treatment_var}"
        if subgroup_var and subgroup_val is not None:
            fname += f"_rs{subgroup_val}"
        fig.savefig(FIGURE_DIR / f"{fname}.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIGURE_DIR / f"{fname}.pdf", bbox_inches="tight")
        print(f"    Saved: {fname}.png")
    plt.close(fig)


def compute_first_stage_fit(
    dt: pd.DataFrame,
    outcome_var: str,
    treatment_var: str,
    learner_name: str,
    learner: Dict[str, Any],
    confounder_groups: Optional[Dict[str, List[str]]] = None,
    include_source_ecoli: bool = False,
    include_child_controls: bool = False,
    subgroup_var: Optional[str] = None,
    subgroup_val: Optional[Any] = None,
    dataset_type: str = "HH",
) -> Dict[str, float]:
    """
    Compute first-stage fit statistics: AUC for propensity score model,
    R² for outcome model.
    """
    if confounder_groups is None:
        from config import BASE_CONFOUNDERS
        confounder_groups = BASE_CONFOUNDERS

    dt_work = dt.copy()
    if subgroup_var is not None and subgroup_val is not None:
        dt_work = dt_work[dt_work[subgroup_var] == subgroup_val].copy()

    X, dt_work = create_model_matrix(
        dt_work, outcome_var,
        confounder_groups=confounder_groups,
        include_source_ecoli=include_source_ecoli,
        include_child_controls=include_child_controls,
    )

    complete_mask = (
        dt_work[outcome_var].notna() &
        dt_work[treatment_var].notna() &
        ~np.isnan(X).any(axis=1) &
        ~np.isinf(X).any(axis=1)
    )
    dt_clean = dt_work[complete_mask].copy()
    X_clean = X[complete_mask.values]
    y = dt_clean[outcome_var].values.astype(float)
    d = dt_clean[treatment_var].values.astype(float)

    metrics = {"learner": learner_name, "n": len(y)}

    # Propensity score model (m)
    try:
        m = clone(learner["m"])
        m.fit(X_clean, d)
        if hasattr(m, "predict_proba"):
            ps = m.predict_proba(X_clean)[:, 1]
            metrics["auc_m"] = roc_auc_score(d, ps)
        else:
            metrics["auc_m"] = np.nan
    except Exception as e:
        metrics["auc_m"] = np.nan

    # Outcome model (g)
    try:
        g = clone(learner["g"])
        g.fit(X_clean, y)
        y_pred = g.predict(X_clean)
        metrics["r2_g"] = max(r2_score(y, y_pred), 0)
    except Exception:
        metrics["r2_g"] = np.nan

    # Treatment model (predict D given X) - also from m
    try:
        if hasattr(m, "predict_proba"):
            metrics["auc_m"] = roc_auc_score(d, m.predict_proba(X_clean)[:, 1])
    except Exception:
        pass

    return metrics


def covariate_balance(
    dt: pd.DataFrame,
    treatment_var: str,
    confounder_vars: List[str],
    subgroup_var: Optional[str] = None,
    subgroup_val: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Compute standardized mean differences for covariates between
    treated and untreated groups.
    """
    dt_work = dt.copy()
    if subgroup_var and subgroup_val is not None:
        dt_work = dt_work[dt_work[subgroup_var] == subgroup_val].copy()

    d = dt_work[treatment_var].values
    treated = d == 1
    untreated = d == 0

    balance = []
    for var in confounder_vars:
        if var not in dt_work.columns:
            continue
        vals = dt_work[var].values.astype(float)
        mask = dt_work[treatment_var].notna().values & ~np.isnan(vals)
        if mask.sum() < 50:
            continue
        vals_m = vals[mask]
        d_m = d[mask]
        t_vals = vals_m[d_m == 1]
        u_vals = vals_m[d_m == 0]
        mean_t = np.nanmean(t_vals)
        mean_u = np.nanmean(u_vals)
        sd_pooled = np.sqrt((np.nanvar(t_vals) + np.nanvar(u_vals)) / 2)
        smd = (mean_t - mean_u) / sd_pooled if sd_pooled > 0 else 0
        balance.append({
            "variable": var,
            "mean_treated": mean_t,
            "mean_untreated": mean_u,
            "smd": smd,
            "n_treated": len(t_vals),
            "n_untreated": len(u_vals),
        })

    return pd.DataFrame(balance)