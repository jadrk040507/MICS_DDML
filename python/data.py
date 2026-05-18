"""
Data preparation functions for the DoubleML MICS Analysis.

This module handles all data loading and preparation.
Configuration is imported from config.py - edit that file to change variables.
"""

from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

from config import (
    DATA_FILE, FACTORS, BASE_CONFOUNDERS,
    OUTCOMES, MIN_OBSERVATIONS, SUBGROUP_LABELS,
    CLUSTER_VAR, USE_CLUSTERING
)


def prepare_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load and prepare data for DoubleML analysis.
    
    Creates all outcome variables, treatment indicators, and confounders
    as specified in config.py.
    
    Args:
        filepath: Path to the data file. Uses DATA_FILE from config if None.
    
    Returns:
        Prepared pandas DataFrame with all variables ready for analysis.
    """
    if filepath is None:
        filepath = DATA_FILE
    
    print(f"  Preparing data from {filepath}")
    
    # Load data
    dt = pd.read_stata(filepath, convert_categoricals=False)
    
    # Convert non-factor columns to numeric
    for col in dt.columns:
        if col not in FACTORS:
            dt[col] = pd.to_numeric(dt[col], errors='coerce')
    
    # ==========================================================================
    # OUTCOMES
    # ==========================================================================
    
    # Create outcomes based on type
    for outcome in OUTCOMES:
        var = outcome["var"]
        outcome_type = outcome["type"]
        
        if var not in dt.columns:
            # Try to create the outcome if it doesn't exist
            _create_outcome(dt, outcome)
        else:
            # Ensure correct type
            if outcome_type == "binary":
                dt[var] = dt[var].astype(int)
            else:
                dt[var] = pd.to_numeric(dt[var], errors='coerce')
    
    # ==========================================================================
    # TREATMENTS
    # ==========================================================================
    
    # Any water treatment (includes boil, chlorine, filter, other)
    if "water_treatment" in dt.columns:
        dt["any_treatment"] = dt["water_treatment"].astype(int)
    
    # Specific treatments (vs control = no treatment)
    # WQ15_g codes: 0=no treatment, 1=boil, 2=chlorine, 3=filter, 98=other
    treatment_mappings = {
        "no_treatment": "WQ15_g_0",
        "boil": "WQ15_g_1",
        "chlorine": "WQ15_g_2",
        "filter": "WQ15_g_3",
        "other_treat": "WQ15_g_98"
    }
    
    for treat_var, source_col in treatment_mappings.items():
        if source_col in dt.columns:
            dt[treat_var] = dt[source_col].astype(int)
    
    # ==========================================================================
    # CONFOUNDERS
    # ==========================================================================
    
    # Wealth quintile dummies from windex5 (categorical 1..5)
    if "windex5" in dt.columns:
        dt["windex5"] = pd.to_numeric(dt["windex5"], errors='coerce')
        for i in range(1, 6):
            dt[f"wealth_q{i}"] = (dt["windex5"] == i).astype(int)
    
    # Education dummies from helevel
    if "helevel" in dt.columns:
        dt["helevel_num"] = pd.to_numeric(dt["helevel"], errors='coerce')
        dt["edu_0"] = dt["helevel_num"].eq(0).astype(int)
        dt["edu_1"] = dt["helevel_num"].eq(1).astype(int)
        dt["edu_2"] = dt["helevel_num"].eq(2).astype(int)
        dt["edu_3"] = dt["helevel_num"].eq(3).astype(int)
        dt["edu_4"] = dt["helevel_num"].eq(4).astype(int)
        dt["edu_na"] = dt["helevel_num"].isna().astype(int)
    
    # Urban/rural (binary)
    if "urban" in dt.columns:
        dt["urban_bin"] = dt["urban"].astype(int)
    
    # Sanitation
    if "improved_latrine" in dt.columns:
        dt["sanitation"] = dt["improved_latrine"].astype(int)
    
    # Number of children
    if "HHCHILDREN" in dt.columns:
        dt["num_children"] = dt["HHCHILDREN"].fillna(0).astype(int)
    
    # Water source (categorical for dummies)
    if "WS1" in dt.columns:
        dt["water_source"] = dt["WS1"].astype('category')
    
    # Country fixed effects (categorical for dummies)
    if "Country" in dt.columns:
        dt["country"] = dt["Country"].astype('category')
    
    # Source risk level (ALWAYS included as control)
    # 0 = No Risk, 1 = Moderate, 2 = Very High Risk
    if "RiskSource" in dt.columns:
        dt["risk_source"] = dt["RiskSource"].astype(int)
    
    # ==========================================================================
    # ADDITIONAL CONFOUNDERS
    # ==========================================================================
    
    # Child age (0-4 years for under-5 children)
    if "age" in dt.columns:
        dt["age"] = pd.to_numeric(dt["age"], errors='coerce')
    
    # Child sex
    if "HHSEX" in dt.columns:
        # HHSEX is child sex: '1. Male' / '2. Female' or similar
        dt["child_sex"] = dt["HHSEX"].astype('category')
    
    # Region (within country)
    if "Region" in dt.columns:
        dt["region"] = dt["Region"].astype('category')
    
    # Rainy season (0=dry, 1=rainy)
    if "rainy_season" in dt.columns:
        dt["rainy_season"] = dt["rainy_season"].astype(int)
    
    # Household ID for clustering (if exists)
    if CLUSTER_VAR in dt.columns:
        dt[CLUSTER_VAR] = dt[CLUSTER_VAR].astype(str)  # Ensure string for grouping
    else:
        print(f"  Warning: Cluster variable '{CLUSTER_VAR}' not found in data")
    
    return dt


def _create_outcome(dt: pd.DataFrame, outcome: Dict[str, Any]) -> None:
    """
    Create an outcome variable if it doesn't exist.
    
    Args:
        dt: DataFrame to modify (in-place).
        outcome: Outcome specification from OUTCOMES list.
    """
    var = outcome["var"]
    
    # Some Risk Home: 1 if 0 <= WQ26 <= 100, else 0
    if var == "some_risk_home":
        if "WQ26" in dt.columns:
            dt[var] = ((dt["WQ26"] >= 0) & (dt["WQ26"] <= 100)).astype(int)
        else:
            raise ValueError(f"Cannot create '{var}': WQ26 not found in data")
    
    # Very High Risk Home: 1 if WQ26 >= 101, else 0
    elif var == "very_high_risk_home":
        if "WQ26" in dt.columns:
            dt[var] = (dt["WQ26"] >= 101).astype(int)
        else:
            raise ValueError(f"Cannot create '{var}': WQ26 not found in data")
    
    # Diarrhea (should already exist in dataset)
    elif var == "diarrhea":
        # Assuming diarrhea is already in the dataset
        # If it needs to be created, add logic here
        pass
    
    else:
        raise ValueError(f"Unknown outcome '{var}': cannot auto-create")


def create_model_matrix(
    dt: pd.DataFrame,
    outcome_var: str,
    include_source_ecoli: bool = False
) -> np.ndarray:
    """
    Create model matrix X for DoubleML.
    
    Automatically includes:
    - Base confounders from config
    - Water source dummies
    - Country fixed effects
    - Risk source (always included)
    
    Args:
        dt: Prepared DataFrame.
        outcome_var: Name of outcome variable (for outcome-specific confounders).
        include_source_ecoli: Whether to include source E.coli (not used currently).
    
    Returns:
        Feature matrix as numpy array.
    """
    from config import BASE_CONFOUNDERS, DIARRHEA_ADDITIONAL_CONFOUNDERS
    
    # Start with base confounders
    X_base = BASE_CONFOUNDERS.copy()
    
    # Add outcome-specific confounders
    if outcome_var == "diarrhea":
        X_base.extend(DIARRHEA_ADDITIONAL_CONFOUNDERS)
    
    # Filter to columns that exist in data AND are not categorical
    # (categorical vars like 'water_source', 'country', 'region', 'child_sex' will be converted to dummies)
    categorical_cols = ['water_source', 'country', 'region', 'child_sex']
    available_cols = [col for col in X_base if col in dt.columns and col not in categorical_cols]
    
    # Extract base columns (numeric only)
    X_df = dt[available_cols].copy()
    
    # Water source dummies
    if "water_source" in dt.columns:
        ws_dummies = pd.get_dummies(dt["water_source"], prefix="ws", drop_first=False)
        # Ensure all values are numeric
        ws_dummies = ws_dummies.astype(float)
        X_df = pd.concat([X_df, ws_dummies], axis=1)
    
    # Country dummies
    if "country" in dt.columns:
        country_dummies = pd.get_dummies(dt["country"], prefix="ctry", drop_first=False)
        country_dummies.columns = country_dummies.columns.str.replace(' ', '_')
        # Ensure all values are numeric
        country_dummies = country_dummies.astype(float)
        X_df = pd.concat([X_df, country_dummies], axis=1)
    
    # Region dummies (within-country geographic variation)
    if "region" in dt.columns:
        region_dummies = pd.get_dummies(dt["region"], prefix="reg", drop_first=False)
        region_dummies = region_dummies.astype(float)
        X_df = pd.concat([X_df, region_dummies], axis=1)
    
    # Child sex dummies
    if "child_sex" in dt.columns:
        sex_dummies = pd.get_dummies(dt["child_sex"], prefix="sex", drop_first=False)
        sex_dummies = sex_dummies.astype(float)
        X_df = pd.concat([X_df, sex_dummies], axis=1)
    
    # Convert to float array (handle any remaining non-numeric values)
    X_df = X_df.apply(pd.to_numeric, errors='coerce')
    X = X_df.values.astype(float)
    return X


def get_summary(dt: pd.DataFrame) -> Dict[str, Any]:
    """
    Print and return summary statistics.
    
    Args:
        dt: Prepared DataFrame.
    
    Returns:
        Dictionary containing summary statistics.
    """
    print("\n=== DATA SUMMARY ===\n")
    print(f"Observations: {dt.shape[0]}\n")
    
    summary = {
        'n_obs': dt.shape[0],
        'outcomes': {},
        'treatments': {}
    }
    
    # Outcomes
    print("OUTCOMES:")
    for outcome in OUTCOMES:
        var = outcome["var"]
        label = outcome["label"]
        
        if var in dt.columns:
            if outcome["type"] == "binary":
                n_positive = int(dt[var].sum())
                mean_val = dt[var].mean()
                print(f"  {label}: {n_positive:,} obs ({mean_val*100:.1f}% positive)")
            else:
                mean_val = dt[var].mean()
                std_val = dt[var].std()
                print(f"  {label}: mean={mean_val:.2f}, sd={std_val:.2f}")
            
            summary['outcomes'][var] = {
                'label': label,
                'type': outcome["type"],
                'mean': float(mean_val)
            }
        else:
            print(f"  {label}: MISSING")
    print()
    
    # Treatments
    print("TREATMENTS:")
    treatment_vars = ["any_treatment", "boil", "chlorine", "filter", "other_treat"]
    
    for treat_var in treatment_vars:
        if treat_var in dt.columns:
            mean_val = dt[treat_var].mean()
            label = treat_var.replace('_', ' ').title()
            print(f"  {label}: {mean_val*100:.1f}%")
            summary['treatments'][treat_var] = mean_val * 100
        else:
            print(f"  {treat_var}: MISSING")
    print()
    
    # Subgroups
    print("SUBGROUPS (by Risk Source):")
    if "risk_source" in dt.columns:
        for level, label in SUBGROUP_LABELS.items():
            n = (dt["risk_source"] == level).sum()
            pct = n / dt.shape[0] * 100
            print(f"  {label} (level {level}): {n:,} obs ({pct:.1f}%)")
    print()
    
    return summary


def validate_data(dt: pd.DataFrame) -> bool:
    """
    Validate that all required variables exist in the data.
    
    Args:
        dt: Prepared DataFrame.
    
    Returns:
        True if all variables are present, False otherwise.
    """
    errors = []
    
    # Check outcomes
    for outcome in OUTCOMES:
        if outcome["var"] not in dt.columns:
            errors.append(f"Outcome '{outcome['var']}' not found")
    
    # Check treatments
    if "any_treatment" not in dt.columns:
        errors.append("Treatment 'any_treatment' not found")
    
    for treat in ["boil", "chlorine", "filter", "other_treat"]:
        if treat not in dt.columns:
            errors.append(f"Treatment '{treat}' not found")
    
    # Check base confounders
    for conf in BASE_CONFOUNDERS:
        if conf not in dt.columns and conf not in ["water_source", "country"]:
            # water_source and country are converted to dummies
            errors.append(f"Confounder '{conf}' not found")
    
    # Check risk_source (always required)
    if "risk_source" not in dt.columns:
        errors.append("Subgroup variable 'risk_source' not found")
    
    if errors:
        print("\n=== DATA VALIDATION ERRORS ===")
        for err in errors:
            print(f"  - {err}")
        return False
    
    print("\n✓ All required variables found in data")
    return True
