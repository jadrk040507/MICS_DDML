"""
LaTeX table generation for DoubleML MICS Analysis.

Generates publication-quality tables in the specified format.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "Output"


def format_coef(coef: float, se: float, p_threshold: float = 0.05) -> str:
    """Format coefficient with significance stars.
    
    Significance levels:
    *** p < 0.01 (1%)
    **  p < 0.05 (5%)
    *   p < 0.10 (10%)
    """
    # Calculate p-value (approximate, using normal distribution)
    t_stat = abs(coef / se) if se > 0 else 0
    
    # Two-tailed p-value approximation
    if t_stat >= 2.576:  # p < 0.01 (1%)
        stars = "***"
    elif t_stat >= 1.96:  # p < 0.05 (5%)
        stars = "**"
    elif t_stat >= 1.645:  # p < 0.10 (10%)
        stars = "*"
    else:
        stars = ""
    
    return f"{coef:.4f}{stars}"


def get_sensitivity_values(result: Dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract RV and RVa from a DoubleML result.
    
    Returns:
        Tuple of (RV, RVa) in percentage terms, or (None, None) if unavailable.
    """
    model = result.get('model')
    if model is None:
        return None, None
    
    try:
        # Run sensitivity analysis if not already done
        if (not hasattr(model, 'sensitivity_params') or 
            model.sensitivity_params is None or
            'rv' not in model.sensitivity_params):
            model.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0)
        
        # Get sensitivity params
        sens_params = model.sensitivity_params
        
        # Extract RV and RVa (they're numpy arrays, convert to float then to %)
        rv_array = sens_params.get('rv')
        rva_array = sens_params.get('rva')
        
        rv = None
        rva = None
        
        if rv_array is not None and len(rv_array) > 0:
            rv = float(rv_array[0]) * 100  # Convert to percentage
        
        if rva_array is not None and len(rva_array) > 0:
            rva = float(rva_array[0]) * 100  # Convert to percentage
        
        return rv, rva
    
    except Exception as e:
        print(f"    Warning: Could not extract sensitivity values: {e}")
        return None, None


def calculate_stacked_weights(dt: pd.DataFrame, outcome_var: str, treatment_var: str,
                               X: np.ndarray, y: np.ndarray, d: np.ndarray,
                               subgroup_var: Optional[str] = None,
                               subgroup_val: Optional[Any] = None) -> Tuple[List[float], List[float]]:
    """
    Re-fit stacked ensemble on full data to extract meta-learner weights.
    
    Returns:
        Tuple of (g_weights, m_weights) - weights for outcome and treatment models.
    """
    from sklearn.linear_model import (LinearRegression, LogisticRegressionCV, 
                                       LassoCV, RidgeCV, ElasticNetCV, Ridge)
    from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, 
                                   StackingRegressor, StackingClassifier)
    import xgboost as xgb
    from config import RANDOM_STATE, N_FOLDS
    
    # Subsample if needed
    if subgroup_var is not None and subgroup_val is not None:
        if subgroup_var not in dt.columns:
            print(f"    Warning: Subgroup variable '{subgroup_var}' not found")
            return [0.167] * 6, [0.167] * 6
        dt_sub = dt[dt[subgroup_var] == subgroup_val].copy()
        
        # Filter X, y, d accordingly
        complete = np.all(~np.isnan(X), axis=1) & pd.Series(dt[outcome_var].notna() & dt[treatment_var].notna())
        dt_complete = dt[complete].copy()
        mask = dt_complete[subgroup_var] == subgroup_val
        y_sub = y[complete][mask]
        d_sub = d[complete][mask]
        X_sub = X[complete][mask]
    else:
        complete = np.all(~np.isnan(X), axis=1) & pd.Series(dt[outcome_var].notna() & dt[treatment_var].notna())
        y_sub = y[complete]
        d_sub = d[complete]
        X_sub = X[complete]
    
    if len(y_sub) < 100:
        print(f"    Warning: Too few observations for weight calculation ({len(y_sub)})")
        return [0.167] * 6, [0.167] * 6
    
    # Create base learners for g (outcome model - regression)
    base_learners_g = [
        ("ols", LinearRegression()),
        ("lasso", LassoCV(alphas=np.logspace(-3, -1, 10), cv=2, random_state=RANDOM_STATE, 
                          max_iter=1000, selection='random')),
        ("ridge", RidgeCV(cv=5)),
        ("enet", ElasticNetCV(l1_ratio=0.5, cv=5, random_state=RANDOM_STATE)),
        ("rf", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        ("xgb", xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8,
                                  objective='reg:squarederror', eval_metric='rmse',
                                  tree_method='hist', seed=RANDOM_STATE, n_jobs=-1))
    ]
    
    # Create base learners for m (treatment model - classification)
    base_learners_m = [
        ("ols", LogisticRegressionCV(cv=5, random_state=RANDOM_STATE, max_iter=1000)),
        ("lasso", LogisticRegressionCV(Cs=np.logspace(-1, 3, 10), cv=2, penalty='l1', 
                                        solver='saga', random_state=RANDOM_STATE, max_iter=500)),
        ("ridge", LogisticRegressionCV(cv=5, penalty='l2', solver='lbfgs', 
                                        random_state=RANDOM_STATE, max_iter=1000)),
        ("enet", LogisticRegressionCV(cv=5, random_state=RANDOM_STATE, max_iter=1000)),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        ("xgb", xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                   subsample=0.8, colsample_bytree=0.8,
                                   objective='binary:logistic', eval_metric='logloss',
                                   tree_method='hist', seed=RANDOM_STATE, n_jobs=-1))
    ]
    
    # Fit stacked ensemble for g (outcome)
    try:
        stacked_g = StackingRegressor(
            estimators=base_learners_g,
            final_estimator=RidgeCV(cv=5),
            cv=N_FOLDS,
            passthrough=False,
            n_jobs=-1
        )
        stacked_g.fit(X_sub, y_sub)
        
        # Extract g weights from fitted final estimator (final_estimator_ with underscore)
        final_g = stacked_g.final_estimator_
        if hasattr(final_g, 'coef_'):
            coefs_g = np.abs(final_g.coef_.ravel())
            total_g = np.sum(coefs_g)
            if total_g > 0:
                g_weights = (coefs_g / total_g).tolist()
            else:
                g_weights = [0.167] * 6
        else:
            g_weights = [0.167] * 6
    except Exception as e:
        print(f"    Warning: Could not fit stacked_g: {e}")
        g_weights = [0.167] * 6
    
    # Fit stacked ensemble for m (treatment)
    try:
        stacked_m = StackingClassifier(
            estimators=base_learners_m,
            final_estimator=LogisticRegressionCV(cv=5, random_state=RANDOM_STATE),
            cv=N_FOLDS,
            stack_method='predict_proba',
            n_jobs=-1
        )
        stacked_m.fit(X_sub, d_sub)
        
        # Extract m weights from fitted final estimator (final_estimator_ with underscore)
        final_m = stacked_m.final_estimator_
        if hasattr(final_m, 'coef_'):
            coefs_m = np.abs(final_m.coef_.ravel())
            total_m = np.sum(coefs_m)
            if total_m > 0:
                m_weights = (coefs_m / total_m).tolist()
            else:
                m_weights = [0.167] * 6
        else:
            m_weights = [0.167] * 6
    except Exception as e:
        print(f"    Warning: Could not fit stacked_m: {e}")
        m_weights = [0.167] * 6
    
    return g_weights, m_weights


def get_stacked_weights_cached(dt: pd.DataFrame, results: List[Dict], outcome: str, treatment: str,
                                X: np.ndarray, y: np.ndarray, d: np.ndarray,
                                subgroup_var: Optional[str] = None,
                                subgroup_val: Optional[Any] = None) -> Tuple[List[float], List[float]]:
    """
    Get stacked weights from cache or calculate them.
    
    Returns:
        Tuple of (g_weights, m_weights).
    """
    # Re-fit to get weights
    print(f"  Calculating stacked weights for {outcome} | {treatment} | subgroup={subgroup_val}")
    return calculate_stacked_weights(dt, outcome, treatment, X, y, d, subgroup_var, subgroup_val)


def create_outcome_table(results: List[Dict], dt: pd.DataFrame, X_dict: Dict,
                          outcome_var: str, outcome_label: str,
                          table_label: str, caption: str) -> str:
    """Create LaTeX table for a specific outcome with sensitivity analysis."""
    
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{table_label}}}")
    lines.append("\\begin{tabular}{lccccccc}")
    lines.append("\\toprule")
    lines.append(" & OLS & Lasso & Ridge & ENet & RF & XGB & Stacked \\\\")
    lines.append("\\midrule")
    
    # Learner order (6 base learners + stacked)
    learner_order = ['ols', 'lasso', 'ridge', 'enet', 'rf', 'xgb', 'stacked']
    
    # Prepare data
    y = dt[outcome_var].values
    
    # Define treatments and their labels
    treatments = [
        ('any_treatment', 'Any Treatment', 'Panel A'),
        ('treat_boil', 'Boil (vs Control)', 'Panel B'),
        ('treat_chlorine', 'Chlorine (vs Control)', 'Panel C'),
        ('treat_filter', 'Filter (vs Control)', 'Panel D'),
        ('treat_other', 'Other (vs Control)', 'Panel E')
    ]
    
    for treat_var, treat_label, panel_label in treatments:
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{{panel_label}: {treat_label}}}}} \\\\")
        lines.append("\\midrule")
        
        # Get results
        treat_results = {}
        for r in results:
            if (r['outcome'] == outcome_var and r['treatment'] == treat_var and
                r.get('subgroup_var') is None):
                treat_results[r['learner']] = r
        
        # Calculate stacked weights
        if treat_var in dt.columns and X_dict.get(treat_var) is not None:
            d_treat = dt[treat_var].values
            g_weights, m_weights = get_stacked_weights_cached(
                dt, results, outcome_var, treat_var,
                X_dict.get(treat_var), y, d_treat
            )
        else:
            g_weights, m_weights = [0.167] * 6, [0.167] * 6
        
        # Extract sensitivity values (RV, RVa) for each learner
        rv_values = {}
        rva_values = {}
        for learner in learner_order:
            if learner in treat_results:
                rv, rva = get_sensitivity_values(treat_results[learner])
                rv_values[learner] = rv
                rva_values[learner] = rva
        
        # Coefficient row
        coef_line = "Coefficient & "
        for learner in learner_order:
            if learner in treat_results:
                r = treat_results[learner]
                coef_str = format_coef(r['coef'], r['se'])
                coef_line += f"{coef_str} & "
            else:
                coef_line += "--- & "
        coef_line = coef_line.rstrip(" & ") + " \\\\"
        lines.append(coef_line)
        
        # SE row
        se_line = " & "
        for learner in learner_order:
            if learner in treat_results:
                r = treat_results[learner]
                se_line += f"({r['se']:.4f}) & "
            else:
                se_line += " & "
        se_line = se_line.rstrip(" & ") + " \\\\"
        lines.append(se_line)
        
        # RV row (Robustness Value)
        rv_line = "RV (\\%) & "
        for learner in learner_order:
            if learner in treat_results and rv_values.get(learner) is not None:
                rv_line += f"{rv_values[learner]:.2f} & "
            else:
                rv_line += " & "
        rv_line = rv_line.rstrip(" & ") + " \\\\"
        lines.append(rv_line)
        
        # RVa row (Adjusted Robustness Value)
        rva_line = "RVa (\\%) & "
        for learner in learner_order:
            if learner in treat_results and rva_values.get(learner) is not None:
                rva_line += f"{rva_values[learner]:.3f} & "
            else:
                rva_line += " & "
        rva_line = rva_line.rstrip(" & ") + " \\\\"
        lines.append(rva_line)
        
        # Weights row (for g - outcome model)
        weight_line = "Weight $(g)$ & "
        for i, learner in enumerate(learner_order):
            if learner == 'stacked':
                weight_line += "— & "
            elif learner in treat_results and i < len(g_weights):
                weight_line += f"{g_weights[i]:.2f} & "
            else:
                weight_line += " & "
        weight_line = weight_line.rstrip(" & ") + " \\\\"
        lines.append(weight_line)
        
        # Weights row (for m - treatment model)
        weight_line_m = "Weight $(m)$ & "
        for i, learner in enumerate(learner_order):
            if learner == 'stacked':
                weight_line_m += "— & "
            elif learner in treat_results and i < len(m_weights):
                weight_line_m += f"{m_weights[i]:.2f} & "
            else:
                weight_line_m += " & "
        weight_line_m = weight_line_m.rstrip(" & ") + " \\\\"
        lines.append(weight_line_m)
        
        lines.append("")
    
    # Observations
    n_obs = len(y)
    lines.append("\\midrule")
    lines.append(f"Observations & \\multicolumn{{7}}{{c}}{{{n_obs:,}}} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    
    # Notes
    lines.append("\\begin{minipage}{\\textwidth}")
    lines.append("\\footnotesize")
    lines.append("\\textit{Notes:} DoubleML IRM estimates with 5-fold cross-validation, 2 repetitions. ")
    lines.append("Standard errors clustered at household level (13,596 clusters) in parentheses. ")
    lines.append("Controls: wealth quintiles, education, urban, sanitation, water source FE, country FE. ")
    lines.append("RV (Robustness Value) = minimum % of residual variance explained by unobserved confounder ")
    lines.append("to make 95% CI include zero. RVa = minimum % to make point estimate equal zero. ")
    lines.append("Stacked weights are normalized absolute coefficients from the ridge/logistic meta-learner. ")
    lines.append("Weight (g) = outcome model, Weight (m) = treatment (propensity score) model. ")
    lines.append("$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$")
    lines.append("\\end{minipage}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def create_subgroup_table(results: List[Dict], dt: pd.DataFrame, X_dict: Dict,
                           outcome_var: str, outcome_label: str,
                           subgroup_var: str = "risk_source") -> str:
    """Create LaTeX table for subgroups (by risk source) with sensitivity analysis."""
    
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Effect of Boiling on {outcome_label} by Water Source Risk Level}}")
    lines.append(f"\\label{{tab:{outcome_var}_subgroups}}")
    lines.append("\\begin{tabular}{lccccccc}")
    lines.append("\\toprule")
    lines.append(" & OLS & Lasso & Ridge & ENet & RF & XGB & Stacked \\\\")
    lines.append("\\midrule")
    
    learner_order = ['ols', 'lasso', 'ridge', 'enet', 'rf', 'xgb', 'stacked']
    
    subgroup_labels = {
        0: "No Risk (Risk Source = 0)",
        1: "Moderate Risk (Risk Source = 1)",
        2: "Very High Risk (Risk Source = 2)"
    }
    
    y = dt[outcome_var].values
    
    for subgroup_val in [0, 1, 2]:
        label = subgroup_labels.get(subgroup_val, f"Risk Source = {subgroup_val}")
        
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{Panel {chr(65+subgroup_val)}: {label}}}}} \\\\")
        lines.append("\\midrule")
        
        # Get results for this subgroup
        subgroup_results = {}
        for r in results:
            if (r['outcome'] == outcome_var and r['treatment'] == 'treat_boil' and
                r.get('subgroup_var') == subgroup_var and
                r.get('subgroup_val') == subgroup_val):
                subgroup_results[r['learner']] = r
        
        # Calculate stacked weights for this subgroup
        d_boil = dt['treat_boil'].values
        g_weights, m_weights = get_stacked_weights_cached(
            dt, results, outcome_var, 'treat_boil',
            X_dict.get('treat_boil'), y, d_boil,
            subgroup_var, subgroup_val
        )
        
        # Extract sensitivity values (RV, RVa) for each learner
        rv_values = {}
        rva_values = {}
        for learner in learner_order:
            if learner in subgroup_results:
                rv, rva = get_sensitivity_values(subgroup_results[learner])
                rv_values[learner] = rv
                rva_values[learner] = rva
        
        # Coefficient row
        coef_line = "Coefficient & "
        for learner in learner_order:
            if learner in subgroup_results:
                r = subgroup_results[learner]
                coef_str = format_coef(r['coef'], r['se'])
                coef_line += f"{coef_str} & "
            else:
                coef_line += "--- & "
        coef_line = coef_line.rstrip(" & ") + " \\\\"
        lines.append(coef_line)
        
        # SE row
        se_line = " & "
        for learner in learner_order:
            if learner in subgroup_results:
                r = subgroup_results[learner]
                se_line += f"({r['se']:.4f}) & "
            else:
                se_line += " & "
        se_line = se_line.rstrip(" & ") + " \\\\"
        lines.append(se_line)
        
        # RV row (Robustness Value)
        rv_line = "RV (\\%) & "
        for learner in learner_order:
            if learner in subgroup_results and rv_values.get(learner) is not None:
                rv_line += f"{rv_values[learner]:.2f} & "
            else:
                rv_line += " & "
        rv_line = rv_line.rstrip(" & ") + " \\\\"
        lines.append(rv_line)
        
        # RVa row (Adjusted Robustness Value)
        rva_line = "RVa (\\%) & "
        for learner in learner_order:
            if learner in subgroup_results and rva_values.get(learner) is not None:
                rva_line += f"{rva_values[learner]:.3f} & "
            else:
                rva_line += " & "
        rva_line = rva_line.rstrip(" & ") + " \\\\"
        lines.append(rva_line)
        
        # Weights row
        weight_line = "Weight $(g)$ & "
        for i, learner in enumerate(learner_order):
            if learner == 'stacked':
                weight_line += "— & "
            elif learner in subgroup_results and i < len(g_weights):
                weight_line += f"{g_weights[i]:.2f} & "
            else:
                weight_line += " & "
        weight_line = weight_line.rstrip(" & ") + " \\\\"
        lines.append(weight_line)
        
        weight_line_m = "Weight $(m)$ & "
        for i, learner in enumerate(learner_order):
            if learner == 'stacked':
                weight_line_m += "— & "
            elif learner in subgroup_results and i < len(m_weights):
                weight_line_m += f"{m_weights[i]:.2f} & "
            else:
                weight_line_m += " & "
        weight_line_m = weight_line_m.rstrip(" & ") + " \\\\"
        lines.append(weight_line_m)
        
        # N for this subgroup
        n_obs = subgroup_results.get('ols', {}).get('n', 0)
        lines.append(f"N & \\multicolumn{{7}}{{c}}{{{n_obs:,}}} \\\\")
        lines.append("")
    
    lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    
    # Notes
    lines.append("\\begin{minipage}{\\textwidth}")
    lines.append("\\footnotesize")
    lines.append("\\textit{Notes:} DoubleML IRM estimates with 5-fold cross-validation, 2 repetitions. ")
    lines.append("Standard errors clustered at household level (13,596 clusters) in parentheses. ")
    lines.append("Controls: wealth quintiles, education, urban, sanitation, water source FE, country FE. ")
    lines.append("Subgroup analysis by water source risk level (0=No Risk, 1=Moderate, 2=Very High). ")
    lines.append("RV (Robustness Value) = minimum % of residual variance explained by unobserved confounder ")
    lines.append("to make 95% CI include zero. RVa = minimum % to make point estimate equal zero. ")
    lines.append("Stacked weights are normalized absolute coefficients from the ridge/logistic meta-learner. ")
    lines.append("Weight (g) = outcome model, Weight (m) = treatment (propensity score) model. ")
    lines.append("$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$")
    lines.append("\\end{minipage}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def main():
    """Main function to generate all tables."""
    print("Loading results and data...")
    
    # Load results
    with open(OUTPUT_DIR / "results_all.pkl", 'rb') as f:
        results = pickle.load(f)
    
    # Load and prepare data
    import config
    import data as data_module
    
    dt = data_module.prepare_data(config.DATA_FILE)
    
    # Create treatment indicators (same as in run.py)
    dt['treat_boil'] = dt['boil']
    dt['treat_chlorine'] = dt['chlorine']
    dt['treat_filter'] = dt['filter']
    dt['treat_other'] = dt['other_treat']
    
    # Create model matrices for each treatment (needed for weight calculation)
    print("Creating model matrices...")
    X_dict = {}
    for treat in ['any_treatment', 'treat_boil', 'treat_chlorine', 'treat_filter', 'treat_other']:
        if treat in dt.columns:
            try:
                X = data_module.create_model_matrix(dt, 'diarrhea', include_source_ecoli=False)
                X_dict[treat] = X
                print(f"  Created X for {treat}: shape {X.shape}")
            except Exception as e:
                print(f"  Warning: Could not create X for {treat}: {e}")
                X_dict[treat] = None
    
    # Define outcomes to generate tables for
    outcomes = [
        {
            'var': 'diarrhea',
            'label': 'Diarrhea (Child had diarrhea in last 2 weeks)',
            'caption': 'Effect of Water Treatment on Diarrhea (Child had diarrhea in last 2 weeks)'
        },
        {
            'var': 'some_risk_home',
            'label': 'Any Detectable E.coli at Home ($WQ26>0$)',
            'caption': 'Effect of Water Treatment on Any Detectable E.coli at Home ($WQ26>0$)'
        },
        {
            'var': 'very_high_risk_home',
            'label': 'Very High E.coli Risk at Home ($WQ26\\geq101$)',
            'caption': 'Effect of Water Treatment on Very High E.coli Risk at Home ($WQ26\\geq101$)'
        }
    ]
    
    # Generate tables for each outcome
    for outcome in outcomes:
        outcome_var = outcome['var']
        outcome_label = outcome['label']
        caption = outcome['caption']
        
        print(f"\n{'='*60}")
        print(f"Generating tables for: {outcome_label}")
        print('='*60)
        
        # Main table (all treatments)
        print(f"\nCreating main table for {outcome_var}...")
        main_table = create_outcome_table(
            results, dt, X_dict,
            outcome_var, outcome_label,
            table_label=f"tab:{outcome_var}_full",
            caption=caption
        )
        main_table_path = OUTPUT_DIR / f"table_{outcome_var}.tex"
        with open(main_table_path, 'w') as f:
            f.write(main_table)
        print(f"Saved: {main_table_path}")
        
        # Subgroup table (boil by risk source)
        print(f"\nCreating subgroup table for {outcome_var}...")
        subgroup_table = create_subgroup_table(
            results, dt, X_dict,
            outcome_var, outcome_label,
            subgroup_var='risk_source'
        )
        subgroup_table_path = OUTPUT_DIR / f"table_{outcome_var}_subgroups.tex"
        with open(subgroup_table_path, 'w') as f:
            f.write(subgroup_table)
        print(f"Saved: {subgroup_table_path}")
    
    print("\n" + "="*60)
    print("ALL TABLES GENERATED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    for outcome in outcomes:
        outcome_var = outcome['var']
        print(f"  - {OUTPUT_DIR}/table_{outcome_var}.tex")
        print(f"  - {OUTPUT_DIR}/table_{outcome_var}_subgroups.tex")


if __name__ == "__main__":
    main()
