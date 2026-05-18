"""
Configuration for the DoubleML MICS Analysis.

This file contains ALL user-editable settings for the analysis.
Modify these variables to change the analysis without touching other files.
"""

from pathlib import Path
import numpy as np

# =============================================================================
# FILE PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "Data" / "3. Final" / "MASTER_MICS_FINAL_U5.dta"
OUTPUT_DIR = PROJECT_ROOT / "Output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# RANDOM SEED & PARALLELIZATION
# =============================================================================

RANDOM_STATE = 42
N_JOBS = -1  # Number of parallel jobs; -1 uses all cores

# =============================================================================
# CROSS-VALIDATION SETTINGS
# =============================================================================

N_FOLDS = 5
N_REP = 3

# =============================================================================
# OUTCOMES (DEPENDENT VARIABLES)
# =============================================================================
# Edit this list to add/remove outcomes.
# Each outcome needs:
#   - var: column name in the dataset
#   - label: human-readable name for tables/plots
#   - desc: description of the outcome
#   - type: "binary" or "continuous"

OUTCOMES = [
    {
        "var": "some_risk_home",
        "label": "Some Risk Home",
        "desc": "E.coli count at home 0-100 (binary)",
        "type": "binary"
    },
    {
        "var": "very_high_risk_home",
        "label": "Very High Risk Home",
        "desc": "E.coli count at home >=101 (binary)",
        "type": "binary"
    },
    {
        "var": "diarrhea",
        "label": "Diarrhea",
        "desc": "Child had diarrhea in last 2 weeks (binary)",
        "type": "binary"
    }
]

# =============================================================================
# TREATMENTS (INDEPENDENT VARIABLES OF INTEREST)
# =============================================================================

# Analysis 1: Any treatment vs no treatment
ANY_TREATMENT = {
    "var": "any_treatment",
    "label": "Any Treatment"
}

# Analysis 2: Specific treatments vs control (no treatment)
SPECIFIC_TREATMENTS = [
    {"var": "treat_boil", "label": "Boil"},
    {"var": "treat_chlorine", "label": "Chlorine"},
    {"var": "treat_filter", "label": "Filter"},
    {"var": "treat_other", "label": "Other"}
]

# =============================================================================
# CONFOUNDERS (CONTROL VARIABLES)
# =============================================================================

# Base confounders included in ALL models
BASE_CONFOUNDERS = [
    # Wealth quintile dummies (created automatically from windex5)
    "wealth_q1", "wealth_q2", "wealth_q3", "wealth_q4", "wealth_q5",
    
    # Education dummies (created automatically from helevel)
    "edu_0", "edu_1", "edu_2", "edu_3", "edu_4", "edu_na",
    
    # Urban/rural
    "urban_bin",
    
    # Sanitation
    "sanitation",
    
    # Household composition
    "num_children",
    
    # Child characteristics (IMPORTANT - added for better control)
    "age",           # Child age (0-4 years)
    "child_sex",     # Child sex (male/female)
    
    # Water source (dummies created automatically from WS1)
    "water_source",
    
    # Geographic controls
    "country",       # Country fixed effects (dummies created automatically from Country)
    
    # Source risk level (ALWAYS included as control)
    # 0 = No Risk, 1 = Moderate, 2 = Very High Risk
    "risk_source"
]

# Additional confounders for specific outcomes (optional)
DIARRHEA_ADDITIONAL_CONFOUNDERS = [
    # Add any extra confounders specific to diarrhea model
    # Example: "child_age", "breastfeeding", etc.
    # Note: age is already included in BASE_CONFOUNDERS
]

# =============================================================================
# SUBGROUP ANALYSIS
# =============================================================================

# Subgroup variable for heterogeneous effects analysis
SUBGROUP_VAR = "risk_source"

# Subgroup labels for interpretation
SUBGROUP_LABELS = {
    0: "No Risk",
    1: "Moderate Risk", 
    2: "Very High Risk"
}

# Which treatments to analyze by subgroup (usually the most important ones)
SUBGROUP_TREATMENTS = [
    {"var": "treat_boil", "label": "Boil"}
]

# =============================================================================
# FACTOR VARIABLES (CATEGORICAL - NOT CONVERTED TO NUMERIC)
# =============================================================================

# These columns are kept as categorical and not converted to numeric
FACTORS = [
    "helevel", "windex5", "WS1", "wq27_decile", "country_cat",
    "urban", "Any_U5", "Girls_less_than15", "Boys_15or_less", "Toilet",
    "Country", "RiskSource"
]

# =============================================================================
# ML LEARNERS
# =============================================================================

# Available learners for DoubleML
LEARNER_NAMES = ["ols", "lasso", "ridge", "enet", "rf", "xgb", "stacked"]

# =============================================================================
# DATA PREPARATION SETTINGS
# =============================================================================

# Minimum observations required for a model to run
MIN_OBSERVATIONS = 50

# Source E.coli control (currently not used, but available)
INCLUDE_SOURCE_ECOLI = False

# =============================================================================
# CLUSTERING
# =============================================================================

# Household ID variable for clustered standard errors
# This is CRITICAL for correct inference when multiple observations per household
CLUSTER_VAR = "Cluster_var"  # Household ID in MICS data

# Use clustered standard errors?
USE_CLUSTERING = True
