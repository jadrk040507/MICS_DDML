"""
Model estimation functions for the DoubleML MICS Analysis.

This module handles DoubleML model fitting and result export.
Configuration is imported from config.py - edit that file to change settings.
"""

import os
import pickle
from typing import Optional, List, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    OUTPUT_DIR, CHECKPOINT_DIR, N_FOLDS, N_REP, RANDOM_STATE,
    MIN_OBSERVATIONS, CLUSTER_VAR, USE_CLUSTERING
)
from data import create_model_matrix
from learners import create_learners, create_stacked_ensemble

import doubleml as dl
from sklearn.base import clone, BaseEstimator


def _checkpoint_filename(
    outcome_var: str, 
    treatment_var: str, 
    learner_name: str,
    subgroup_var: Optional[str] = None, 
    subgroup_val: Optional[Any] = None
) -> str:
    """
    Generate checkpoint filename.
    
    Args:
        outcome_var: Outcome variable name.
        treatment_var: Treatment variable name.
        learner_name: Name of the ML learner.
        subgroup_var: Subgroup variable name (optional).
        subgroup_val: Subgroup value (optional).
    
    Returns:
        Checkpoint filename string.
    """
    parts = [outcome_var, treatment_var]
    if subgroup_var is not None and subgroup_val is not None:
        parts.extend([subgroup_var, str(subgroup_val)])
    parts.append(learner_name)
    return "_".join(parts) + ".pkl"


def estimate_effect(
    dt: pd.DataFrame,
    outcome_var: str,
    treatment_var: str,
    learner_name: str,
    learner: Dict[str, BaseEstimator],
    include_source_ecoli: bool = False,
    subgroup_var: Optional[str] = None,
    subgroup_val: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """
    Estimate causal effect using DoubleML IRM.
    
    Args:
        dt: Prepared DataFrame.
        outcome_var: Name of outcome variable.
        treatment_var: Name of treatment variable.
        learner_name: Name of ML learner.
        learner: Dictionary with 'g' and 'm' estimators.
        include_source_ecoli: Whether to include source E.coli (not used).
        subgroup_var: Subgroup variable name.
        subgroup_val: Subgroup value.
    
    Returns:
        Dictionary with coef, se, ci, n, model or None if failed.
    """
    # Checkpoint file
    cp_file = CHECKPOINT_DIR / _checkpoint_filename(
        outcome_var, treatment_var, learner_name, subgroup_var, subgroup_val
    )
    
    if cp_file.is_file():
        try:
            with open(cp_file, 'rb') as f:
                result = pickle.load(f)
            print(f"    Loading checkpoint: {cp_file.name}")
            return result
        except Exception as e:
            print(f"    Warning: could not load checkpoint {cp_file}: {e}")
    
    # Subsample if subgroup specified
    if subgroup_var is not None and subgroup_val is not None:
        if subgroup_var not in dt.columns:
            print(f"    ERROR: Subgroup variable '{subgroup_var}' not found")
            return None
        dt_sub = dt[dt[subgroup_var] == subgroup_val].copy()
    else:
        dt_sub = dt.copy()
    
    # Create model matrix (automatically includes risk_source and other confounders)
    try:
        X = create_model_matrix(dt_sub, outcome_var, include_source_ecoli=include_source_ecoli)
    except Exception as e:
        print(f"    ERROR creating model matrix: {e}")
        return None
    
    # Filter complete cases
    complete = np.all(~np.isnan(X), axis=1) & \
               dt_sub[outcome_var].notna() & \
               dt_sub[treatment_var].notna()
    
    n_complete = int(complete.sum())
    if n_complete < MIN_OBSERVATIONS:
        print(f"    ERROR: Too few observations (N = {n_complete})")
        return None
    
    dt_clean = dt_sub[complete].copy()
    X_clean = X[complete]
    
    # Prepare data for DoubleML
    y = dt_clean[outcome_var].values.ravel()
    d = dt_clean[treatment_var].values.ravel()
    x = X_clean
    
    # Prepare cluster variable if using clustering
    cluster_vars = None
    if USE_CLUSTERING and CLUSTER_VAR in dt_clean.columns:
        cluster_vars = dt_clean[CLUSTER_VAR].values
        n_clusters = len(np.unique(cluster_vars))
        print(f"    Using clustered SEs: {n_clusters} clusters ({CLUSTER_VAR})")
    else:
        if USE_CLUSTERING:
            print(f"    Warning: Cluster variable '{CLUSTER_VAR}' not found, using non-clustered SEs")
    
    try:
        dml_data = dl.DoubleMLData.from_arrays(x=x, y=y, d=d, cluster_vars=cluster_vars)
    except Exception as e:
        print(f"    ERROR creating DoubleMLData: {e}")
        return None
    
    # Fit DoubleML IRM
    try:
        dml = dl.DoubleMLIRM(
            obj_dml_data=dml_data,
            ml_g=clone(learner['g']),
            ml_m=clone(learner['m']),
            n_folds=N_FOLDS,
            n_rep=N_REP,
            score='ATE'
        )
        dml.fit(store_predictions=True)
    except Exception as e:
        print(f"    ERROR fitting DoubleMLIRM: {e}")
        return None
    
    # Extract results
    try:
        ci = dml.confint()
        # confint may return a DataFrame or numpy array
        if hasattr(ci, 'iloc'):
            ci_lower = float(ci.iloc[0, 0])
            ci_upper = float(ci.iloc[0, 1])
        else:
            ci_lower = float(ci[0, 0])
            ci_upper = float(ci[0, 1])
    except Exception as e:
        print(f"    ERROR extracting confidence intervals: {e}")
        return None
    
    result: Dict[str, Any] = {
        'outcome': outcome_var,
        'treatment': treatment_var,
        'learner': learner_name,
        'subgroup_var': subgroup_var,
        'subgroup_val': subgroup_val,
        'coef': float(dml.coef[0]),
        'se': float(dml.se[0]),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n': n_complete,
        'n_clusters': len(np.unique(cluster_vars)) if cluster_vars is not None else None,
        'cluster_var': CLUSTER_VAR if USE_CLUSTERING else None,
        'model': dml
    }
    
    # Save checkpoint
    try:
        with open(cp_file, 'wb') as f:
            pickle.dump(result, f)
        print(f"    Saved checkpoint: {cp_file.name}")
    except Exception as e:
        print(f"    Warning: could not save checkpoint: {e}")
    
    return result


def run_analysis(
    dt: pd.DataFrame,
    outcomes: List[Dict[str, str]],
    treatments: List[Dict[str, str]],
    learners: Dict[str, Dict[str, BaseEstimator]],
    include_source_ecoli: bool = False,
    subgroups: bool = False,
    subgroup_var: Optional[str] = None,
    subgroup_val: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Run analysis over outcomes, treatments, and learners.
    
    Args:
        dt: Prepared DataFrame.
        outcomes: List of outcome dictionaries with 'var' and 'label'.
        treatments: List of treatment dictionaries with 'var' and 'label'.
        learners: Dictionary of learners.
        include_source_ecoli: Whether to include source E.coli.
        subgroups: Whether to run subgroup analysis.
        subgroup_var: Subgroup variable name.
        subgroup_val: Subgroup value.
    
    Returns:
        List of result dictionaries.
    """
    results: List[Dict[str, Any]] = []
    
    for out in outcomes:
        out_var = out['var']
        out_label = out.get('label', out_var)
        
        # Validate outcome column exists
        if out_var not in dt.columns:
            print(f"\n  WARNING: Outcome '{out_var}' not found, skipping")
            continue
        
        for treat in treatments:
            trt_var = treat['var']
            trt_label = treat.get('label', trt_var)
            
            # Validate treatment column exists
            if trt_var not in dt.columns:
                print(f"\n  WARNING: Treatment '{trt_var}' not found, skipping")
                continue
            
            for ln in learners:
                print(f"\n  {out_label} | {trt_label} | {ln}")
                res = estimate_effect(
                    dt=dt,
                    outcome_var=out_var,
                    treatment_var=trt_var,
                    learner_name=ln,
                    learner=learners[ln],
                    include_source_ecoli=include_source_ecoli,
                    subgroup_var=subgroup_var,
                    subgroup_val=subgroup_val
                )
                if res is not None:
                    results.append(res)
                    print(f"    Effect: {res['coef']:.4f} ({res['se']:.4f})")
    
    return results


def export_results(results: List[Dict[str, Any]], filename: str = "results.pkl") -> Path:
    """
    Save results to pickle file.
    
    Args:
        results: List of result dictionaries.
        filename: Output filename.
    
    Returns:
        Path to saved file.
    """
    filepath = OUTPUT_DIR / filename
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to: {filepath}")
    return filepath


def export_latex(results: List[Dict[str, Any]], filename: str = "tables.tex") -> Path:
    """
    Export results to LaTeX table.
    
    Args:
        results: List of result dictionaries.
        filename: Output filename.
    
    Returns:
        Path to saved file.
    """
    filepath = OUTPUT_DIR / filename
    
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{DoubleML Estimation Results}")
    lines.append("\\label{tab:results}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\hline")
    lines.append("Outcome & Treatment & Learner & Effect (SE) & N \\\\")
    lines.append("\\hline")
    
    for r in results:
        line = f"{r['outcome']} & {r['treatment']} & {r['learner']} & {r['coef']:.3f} ({r['se']:.3f}) & {r['n']} \\\\"
        lines.append(line)
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    with open(filepath, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"LaTeX table saved to: {filepath}")
    return filepath


def export_results_csv(results: List[Dict[str, Any]], filename: str = "results.csv") -> Path:
    """
    Export results to CSV file (excluding model objects).
    
    Args:
        results: List of result dictionaries.
        filename: Output filename.
    
    Returns:
        Path to saved file.
    """
    filepath = OUTPUT_DIR / filename
    
    # Remove model objects for CSV export
    export_results = []
    for r in results:
        r_export = {k: v for k, v in r.items() if k != 'model'}
        export_results.append(r_export)
    
    df = pd.DataFrame(export_results)
    df.to_csv(filepath, index=False)
    print(f"CSV results saved to: {filepath}")
    return filepath
