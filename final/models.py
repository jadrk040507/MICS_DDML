"""
DDML IRM estimation, checkpointing, sensitivity analysis, and results management.

Core estimation using DoubleML IRM with household-level clustered standard errors.
"""

import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from config import (
    OUTPUT_DIR, CHECKPOINT_DIR, N_FOLDS, N_REP, RANDOM_STATE,
    MIN_OBSERVATIONS, CLUSTER_VAR, USE_CLUSTERING, LEARNER_NAMES,
    logger,
)
from data import create_model_matrix
from learners import create_learners

import doubleml as dml
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold


def _checkpoint_filename(
    outcome_var: str,
    treatment_var: str,
    learner_name: str,
    subgroup_var: Optional[str] = None,
    subgroup_val: Optional[Any] = None,
    confounder_set: str = "base",
    dataset_type: str = "HH",
    prefix: str = "",
) -> str:
    """Generate checkpoint filename with full context."""
    parts = [dataset_type, outcome_var, treatment_var]
    if subgroup_var is not None and subgroup_val is not None:
        parts.extend([subgroup_var, str(subgroup_val)])
    parts.append(learner_name)
    if confounder_set != "base":
        parts.append(confounder_set)
    if prefix:
        parts = [prefix] + parts
    return "_".join(parts) + ".pkl"


def prepare_model_data(
    dt: pd.DataFrame,
    outcome_var: str,
    treatment_var: str,
    confounder_groups: Optional[Dict[str, List[str]]] = None,
    include_source_ecoli: bool = False,
    include_child_controls: bool = False,
    include_robustness: bool = False,
    subgroup_var: Optional[str] = None,
    subgroup_val: Optional[Any] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]]:
    """
    Prepare clean data matrices for DoubleML estimation.

    Builds the model matrix, applies subgroup filtering, drops incomplete cases,
    and returns the cleaned arrays.  This should be called once per
    outcome-treatment-confounder specification and reused across learners.

    Returns
    -------
    (X_clean, y, d, cluster_vars, dt_clean) or None if data is insufficient.
    """
    # Subsample if subgroup specified
    dt_work = dt.copy()
    if subgroup_var is not None and subgroup_val is not None:
        if subgroup_var not in dt_work.columns:
            logger.error(f"    Subgroup variable '{subgroup_var}' not found")
            return None
        dt_work = dt_work[dt_work[subgroup_var] == subgroup_val].copy()
        if len(dt_work) < MIN_OBSERVATIONS:
            logger.error(f"    Too few observations after subgroup filter (N={len(dt_work)})")
            return None

    # Build model matrix
    X_work, dt_work = create_model_matrix(
        dt_work,
        outcome_var,
        confounder_groups=confounder_groups,
        include_source_ecoli=include_source_ecoli,
        include_child_controls=include_child_controls,
        include_robustness=include_robustness,
    )

    # Filter complete cases on Y, D, and X
    complete_mask = (
        dt_work[outcome_var].notna() &
        dt_work[treatment_var].notna() &
        ~np.isnan(X_work).any(axis=1)
    )
    complete_mask &= ~np.isinf(X_work).any(axis=1)

    n_complete = int(complete_mask.sum())
    if n_complete < MIN_OBSERVATIONS:
        logger.error(f"    Too few complete observations (N={n_complete})")
        return None

    dt_clean = dt_work[complete_mask].copy()
    X_clean = X_work[complete_mask.values]
    y = dt_clean[outcome_var].values.astype(float)
    d = dt_clean[treatment_var].values.astype(float)

    # Check treatment variation
    n_treated = int(d.sum())
    n_untreated = int(len(d) - d.sum())
    if n_treated < 20 or n_untreated < 20:
        logger.error(f"    Insufficient treatment variation (treated={n_treated}, untreated={n_untreated})")
        return None

    # Cluster variable
    cluster_vars = None
    if USE_CLUSTERING and CLUSTER_VAR in dt_clean.columns:
        cluster_vars = dt_clean[CLUSTER_VAR].values
        n_clusters = len(np.unique(cluster_vars))
        if n_clusters < 10:
            logger.warning(f"    Only {n_clusters} clusters, clustering may be unreliable")

    return X_clean, y, d, cluster_vars, dt_clean


def estimate_effect(
    dt: pd.DataFrame,
    outcome_var: str,
    treatment_var: str,
    learner_name: str,
    learner: Dict[str, Any],
    confounder_groups: Optional[Dict[str, List[str]]] = None,
    include_source_ecoli: bool = False,
    include_child_controls: bool = False,
    include_robustness: bool = False,
    subgroup_var: Optional[str] = None,
    subgroup_val: Optional[Any] = None,
    confounder_set: str = "base",
    dataset_type: str = "HH",
    prefix: str = "",
    skip_checkpoint: bool = False,
    prepped_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Estimate causal effect using DoubleML IRM.

    Parameters
    ----------
    dt : Prepared DataFrame.
    outcome_var : Name of outcome variable.
    treatment_var : Name of treatment variable.
    learner_name : Name of the ML learner.
    learner : Dict with 'g' (outcome) and 'm' (treatment) models.
    confounder_groups : Which confounder groups to include.
    include_source_ecoli : Include risk_source dummies (for diarrhea).
    include_child_controls : Include child age/sex (U5 only).
    include_robustness : Include water storage + handwashing.
    subgroup_var : Variable for subsampling.
    subgroup_val : Value to subsample on.
    confounder_set : Label for the confounder set (for checkpoint naming).
    dataset_type : "HH" or "U5".
    prefix : Prefix for checkpoint filename.
    skip_checkpoint : If True, don't load/save checkpoints.
    prepped_data : Optional tuple (X_clean, y, d, cluster_vars, dt_clean) from
        prepare_model_data().  Passing this avoids re-building the model matrix
        when looping over multiple learners for the same specification.

    Returns
    -------
    Dict with coef, se, ci, n, model, or None if estimation fails.
    """
    # Checkpoint
    cp_name = _checkpoint_filename(
        outcome_var, treatment_var, learner_name,
        subgroup_var, subgroup_val, confounder_set, dataset_type, prefix,
    )
    cp_file = CHECKPOINT_DIR / cp_name

    if not skip_checkpoint and cp_file.is_file():
        try:
            with open(cp_file, "rb") as f:
                result = pickle.load(f)
            logger.info(f"    Loaded checkpoint: {cp_name}")
            return result
        except Exception as e:
            logger.warning(f"    Could not load checkpoint {cp_name}: {e}")

    # Prepare data (use cached if provided)
    if prepped_data is not None:
        X_clean, y, d, cluster_vars, dt_clean = prepped_data
    else:
        prep = prepare_model_data(
            dt=dt,
            outcome_var=outcome_var,
            treatment_var=treatment_var,
            confounder_groups=confounder_groups,
            include_source_ecoli=include_source_ecoli,
            include_child_controls=include_child_controls,
            include_robustness=include_robustness,
            subgroup_var=subgroup_var,
            subgroup_val=subgroup_val,
        )
        if prep is None:
            return None
        X_clean, y, d, cluster_vars, dt_clean = prep

    n_complete = len(y)
    n_treated = int(d.sum())
    n_untreated = n_complete - n_treated

    # Use stratified folds for rare treatments to avoid empty folds
    treated_fraction = n_treated / n_complete
    use_stratified = treated_fraction < 0.05 or treated_fraction > 0.95

    try:
        dml_data = dml.DoubleMLData.from_arrays(
            x=X_clean, y=y, d=d,
            cluster_vars=cluster_vars,
        )
    except Exception as e:
        logger.error(f"    Creating DoubleMLData: {e}")
        return None

    # Fit DoubleML IRM
    try:
        dml_model = dml.DoubleMLIRM(
            obj_dml_data=dml_data,
            ml_g=clone(learner["g"]),
            ml_m=clone(learner["m"]),
            n_folds=N_FOLDS,
            n_rep=N_REP,
            score="ATE",
            trimming_rule="truncate",
            trimming_threshold=0.01,
            draw_sample_splitting=not use_stratified,
        )
        if use_stratified:
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            smpls = []
            for _ in range(N_REP):
                for train_idx, test_idx in skf.split(X_clean, d):
                    smpls.append((train_idx, test_idx))
            dml_model.set_sample_splitting(smpls)
        dml_model.fit()
    except Exception as e:
        logger.error(f"    Fitting DoubleMLIRM: {e}")
        return None

    # Extract results
    try:
        coef = float(dml_model.coef[0])
        se = float(dml_model.se[0])
        ci = dml_model.confint()
        if hasattr(ci, "iloc"):
            ci_lower = float(ci.iloc[0, 0])
            ci_upper = float(ci.iloc[0, 1])
        else:
            ci_lower = float(ci[0, 0])
            ci_upper = float(ci[0, 1])
    except Exception as e:
        logger.error(f"    Extracting results: {e}")
        return None

    # Sensitivity analysis
    rv_q = None
    rv_qa = None
    try:
        dml_model.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0, level=0.95)
        sens_params = dml_model.sensitivity_params
        if sens_params is not None:
            rv_q = float(sens_params["rv"][0])
            rv_qa = float(sens_params["rva"][0])
    except Exception:
        pass

    result = {
        "outcome": outcome_var,
        "treatment": treatment_var,
        "learner": learner_name,
        "subgroup_var": subgroup_var,
        "subgroup_val": subgroup_val,
        "confounder_set": confounder_set,
        "dataset_type": dataset_type,
        "coef": coef,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": n_complete,
        "n_treated": n_treated,
        "n_untreated": n_untreated,
        "n_clusters": len(np.unique(cluster_vars)) if cluster_vars is not None else None,
        "cluster_var": CLUSTER_VAR if USE_CLUSTERING else None,
        "rv_q": rv_q,
        "rv_qa": rv_qa,
    }

    # Save checkpoint
    if not skip_checkpoint:
        try:
            with open(cp_file, "wb") as f:
                pickle.dump(result, f)
            logger.info(f"    Saved checkpoint: {cp_name}")
        except Exception as e:
            logger.warning(f"    Could not save checkpoint {cp_name}: {e}")

    return result


def run_analysis(
    dt: pd.DataFrame,
    outcomes: List[Dict[str, str]],
    treatments: List[Dict[str, str]],
    learners: Dict[str, Dict],
    confounder_groups: Optional[Dict[str, List[str]]] = None,
    include_source_ecoli: bool = False,
    include_child_controls: bool = False,
    include_robustness: bool = False,
    subgroup_var: Optional[str] = None,
    subgroup_val: Optional[Any] = None,
    confounder_set: str = "base",
    dataset_type: str = "HH",
    prefix: str = "",
) -> List[Dict[str, Any]]:
    """
    Run DDML estimation over all outcome-treatment-learner combinations.

    Model matrices are built once per outcome-treatment pair and reused across
    learners, cutting redundant data preparation by ~5-7x.
    """
    results = []

    for out in outcomes:
        out_var = out["var"]
        out_label = out.get("label", out_var)

        if out_var not in dt.columns:
            logger.warning(f"Outcome '{out_var}' not found, skipping")
            continue

        for treat in treatments:
            trt_var = treat["var"]
            trt_label = treat.get("label", trt_var)

            if trt_var not in dt.columns:
                logger.warning(f"Treatment '{trt_var}' not found, skipping")
                continue

            # Build model matrix ONCE for this outcome-treatment pair
            prepped = prepare_model_data(
                dt=dt,
                outcome_var=out_var,
                treatment_var=trt_var,
                confounder_groups=confounder_groups,
                include_source_ecoli=include_source_ecoli,
                include_child_controls=include_child_controls,
                include_robustness=include_robustness,
                subgroup_var=subgroup_var,
                subgroup_val=subgroup_val,
            )
            if prepped is None:
                continue

            for ln_name, ln in learners.items():
                logger.info(f"{out_label} | {trt_label} | {ln_name}")

                res = estimate_effect(
                    dt=dt,
                    outcome_var=out_var,
                    treatment_var=trt_var,
                    learner_name=ln_name,
                    learner=ln,
                    confounder_groups=confounder_groups,
                    include_source_ecoli=include_source_ecoli,
                    include_child_controls=include_child_controls,
                    include_robustness=include_robustness,
                    subgroup_var=subgroup_var,
                    subgroup_val=subgroup_val,
                    confounder_set=confounder_set,
                    dataset_type=dataset_type,
                    prefix=prefix,
                    prepped_data=prepped,
                )

                if res is not None:
                    results.append(res)
                    sig = "***" if res["ci_lower"] > 0 or res["ci_upper"] < 0 else ""
                    logger.info(f"    Effect: {res['coef']:.4f} ({res['se']:.4f}) {sig}")

    return results


def export_results(results: List[Dict[str, Any]], filename: str = "results.pkl") -> Any:
    """Save results to pickle file."""
    filepath = OUTPUT_DIR / filename
    with open(filepath, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Results saved to: {filepath}")
    return filepath


def export_results_csv(results: List[Dict[str, Any]], filename: str = "results.csv") -> Any:
    """Export results to CSV."""
    filepath = OUTPUT_DIR / filename
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    logger.info(f"CSV saved to: {filepath}")
    return filepath


def print_significant_effects(results: List[Dict[str, Any]]) -> None:
    """Print a summary of statistically significant effects."""
    sig_results = []
    for r in results:
        if r["ci_lower"] > 0 or r["ci_upper"] < 0:
            sig_results.append(r)

    if not sig_results:
        logger.info("No statistically significant effects found.")
        return

    lines = [
        f"Significant effects (95% CI excludes zero): {len(sig_results)}/{len(results)}",
        f"{'Outcome':<22} {'Treatment':<18} {'Learner':<12} {'Coef':>8} {'SE':>8} {'95% CI':>20} {'N':>7}",
    ]

    for r in sorted(sig_results, key=lambda x: (x["outcome"], x["treatment"], x["learner"])):
        sub = f" (RS={r['subgroup_val']})" if r.get("subgroup_val") is not None else ""
        lines.append(
            f"{r['outcome']:<22} {r['treatment']:<18} {r['learner']:<12} "
            f"{r['coef']:>8.4f} {r['se']:>8.4f} "
            f"[{r['ci_lower']:>7.4f}, {r['ci_upper']:>7.4f}] "
            f"{r['n']:>7}{sub}"
        )
    logger.info("\n".join(lines))
