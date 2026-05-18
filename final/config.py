"""
Configuration for the MICS DDML Analysis.

All user-editable settings for the analysis.
Modify these variables to change the analysis without touching other files.

Two datasets:
  - HH: MASTER_MICS_FINAL.dta (household-level, E.coli outcomes)
  - U5: MASTER_MICS_FINAL_U5.dta (child-level, diarrhea outcome)
"""

from pathlib import Path
import numpy as np

# =============================================================================
# FILE PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data" / "3. Final"

HH_DATA_FILE = DATA_DIR / "MASTER_MICS_FINAL.dta"
U5_DATA_FILE = DATA_DIR / "MASTER_MICS_FINAL_U5.dta"

OUTPUT_DIR = PROJECT_ROOT / "Output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"

for d in [OUTPUT_DIR, CHECKPOINT_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# RANDOM SEED & PARALLELIZATION
# =============================================================================

import os

RANDOM_STATE = 42
N_JOBS = max(1, os.cpu_count() // 2)  # cap to avoid oversubscription

# =============================================================================
# CROSS-VALIDATION SETTINGS
# =============================================================================

N_FOLDS = 5
N_REP = 3

# =============================================================================
# OUTCOMES (DEPENDENT VARIABLES)
# =============================================================================

HH_OUTCOMES = [
    {"var": "SomeRiskHome", "label": "Some Risk Home", "desc": "E.coli 1-100 CFU at home (binary)", "type": "binary"},
    {"var": "VeryHighRiskHome", "label": "Very High Risk Home", "desc": "E.coli >=101 CFU at home (binary)", "type": "binary"},
]

U5_OUTCOMES = [
    {"var": "diarrhea", "label": "Diarrhea", "desc": "Child had diarrhea in last 2 weeks (binary)", "type": "binary"},
]

ALL_OUTCOMES = HH_OUTCOMES + U5_OUTCOMES

# =============================================================================
# TREATMENTS (INDEPENDENT VARIABLES OF INTEREST)
# =============================================================================

ANY_TREATMENT = {"var": "water_treatment", "label": "Any Treatment"}

SPECIFIC_TREATMENTS = [
    {"var": "treat_boil", "label": "Boil"},
    {"var": "treat_chlorine", "label": "Chlorine"},
    {"var": "treat_filter", "label": "Filter"},
    {"var": "treat_other", "label": "Other"},
]

ALL_TREATMENTS = [ANY_TREATMENT] + SPECIFIC_TREATMENTS

# Treatment variable code mapping (WQ15_g -> treatment dummy)
WQ15G_TREATMENT_MAP = {
    1: "treat_boil",
    2: "treat_chlorine",
    3: "treat_filter",
    98: "treat_other",
}

# =============================================================================
# CONFOUNDER GROUPS
# =============================================================================

BASE_CONFOUNDERS = {
    "wealth": ["wealth_q1", "wealth_q2", "wealth_q3", "wealth_q4", "wealth_q5"],
    "education": ["edu_none", "edu_primary", "edu_secondary", "edu_missing"],
    "urban": ["urban_bin"],
    "sanitation": ["improved_latrine", "toilet_pit_latrine", "toilet_open_defecation", "toilet_missing"],
    "children": ["num_children"],
    "water_source": ["water_source"],
    "country": ["country"],
}

ROBUSTNESS_CONFOUNDERS = {
    "water_storage": ["water_stored_covered", "water_stored_uncovered", "water_straight_from_source"],
    "handwashing": ["SoapandWater"],
}

# Confounders only for U5 (diarrhea) models
U5_ADDITIONAL_CONFOUNDERS = {
    "child": ["child_age", "child_sex_male"],
    "source_ecoli": ["risk_source"],  # mediator for E.coli, confounder for diarrhea
}

# Source E.coli: included for diarrhea, excluded for E.coli outcomes
# This is handled programmatically in data.py

# =============================================================================
# CONFOUNDER SETS FOR STABILITY ANALYSIS
# =============================================================================

# Progressive addition of confounder groups (for coefficient stability)
STABILITY_GROUPS = [
    ("No controls", []),
    ("+ Wealth", ["wealth"]),
    ("+ Education", ["wealth", "education"]),
    ("+ Urban", ["wealth", "education", "urban"]),
    ("+ Water source", ["wealth", "education", "urban", "water_source"]),
    ("+ Sanitation", ["wealth", "education", "urban", "water_source", "sanitation"]),
    ("+ Children", ["wealth", "education", "urban", "water_source", "sanitation", "children"]),
    ("+ Country FE", ["wealth", "education", "urban", "water_source", "sanitation", "children", "country"]),
    ("Full", ["wealth", "education", "urban", "water_source", "sanitation", "children", "country"]),
]

# For U5 models, source_ecoli and child are always added after the base groups
U5_STABILITY_GROUPS = [
    ("No controls", []),
    ("+ Wealth", ["wealth"]),
    ("+ Education", ["wealth", "education"]),
    ("+ Urban", ["wealth", "education", "urban"]),
    ("+ Water source", ["wealth", "education", "urban", "water_source"]),
    ("+ Sanitation", ["wealth", "education", "urban", "water_source", "sanitation"]),
    ("+ Children", ["wealth", "education", "urban", "water_source", "sanitation", "children"]),
    ("+ Country FE", ["wealth", "education", "urban", "water_source", "sanitation", "children", "country"]),
    ("+ Source E.coli", ["wealth", "education", "urban", "water_source", "sanitation", "children", "country", "source_ecoli"]),
    ("+ Child controls", ["wealth", "education", "urban", "water_source", "sanitation", "children", "country", "source_ecoli", "child"]),
]

# Leave-one-out: drop one confounder group at a time from the full set
LOO_GROUPS_HH = {
    "wealth": ["education", "urban", "sanitation", "children", "water_source", "country"],
    "education": ["wealth", "urban", "sanitation", "children", "water_source", "country"],
    "urban": ["wealth", "education", "sanitation", "children", "water_source", "country"],
    "sanitation": ["wealth", "education", "urban", "children", "water_source", "country"],
    "water_source": ["wealth", "education", "urban", "sanitation", "children", "country"],
    "country": ["wealth", "education", "urban", "sanitation", "children", "water_source"],
}

LOO_GROUPS_U5 = {
    "wealth": ["education", "urban", "sanitation", "children", "water_source", "country", "source_ecoli", "child"],
    "education": ["wealth", "urban", "sanitation", "children", "water_source", "country", "source_ecoli", "child"],
    "urban": ["wealth", "education", "sanitation", "children", "water_source", "country", "source_ecoli", "child"],
    "sanitation": ["wealth", "education", "urban", "children", "water_source", "country", "source_ecoli", "child"],
    "water_source": ["wealth", "education", "urban", "sanitation", "children", "country", "source_ecoli", "child"],
    "country": ["wealth", "education", "urban", "sanitation", "children", "water_source", "source_ecoli", "child"],
    "source_ecoli": ["wealth", "education", "urban", "sanitation", "children", "water_source", "country", "child"],
    "child": ["wealth", "education", "urban", "sanitation", "children", "water_source", "country", "source_ecoli"],
}

# =============================================================================
# SUBGROUP ANALYSIS
# =============================================================================

SUBGROUP_VAR = "RiskSource"
SUBGROUP_LABELS = {0: "No Risk", 1: "Moderate Risk", 2: "Very High Risk"}
SUBGROUP_TREATMENTS = [ANY_TREATMENT]

# Education subgroups for CATE
EDUCATION_SUBGROUP_VAR = "helevel"
EDUCATION_SUBGROUP_LABELS = {0: "No Education", 1: "Primary", 2: "Secondary", 98: "Missing"}
EDUCATION_SUBGROUP_VALUES = [0, 1, 2]

# =============================================================================
# ML LEARNERS
# =============================================================================

LEARNER_NAMES = ["ols", "lasso", "ridge", "enet", "rf", "xgb", "stacked"]

# Subset used when --fast is passed (fastest + most informative learners)
FAST_LEARNER_NAMES = ["ols", "lasso", "rf", "stacked"]

LEARNER_LABELS = {
    "ols": "OLS",
    "lasso": "Lasso",
    "ridge": "Ridge",
    "enet": "Elastic Net",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "stacked": "Stacked",
}

# =============================================================================
# CLUSTERING
# =============================================================================

CLUSTER_VAR = "Cluster_var"
USE_CLUSTERING = True

# =============================================================================
# DATA PREPARATION SETTINGS
# =============================================================================

MIN_OBSERVATIONS = 50
MIN_TREATED_FRACTION = 0.01
MAX_TREATED_FRACTION = 0.99

# WQ15_g coding in the dataset
WQ15G_CODES = {
    0: "no_treatment",
    1: "boil",
    2: "chlorine",
    3: "filter",
    98: "other",
}

# Helevel coding in the dataset
HELEVEL_CODES = {0: "none", 1: "primary", 2: "secondary", 98: "missing"}

# RiskSource coding
RISKSOURCE_CODES = {0: "no_risk", 1: "moderate", 2: "very_high"}

# =============================================================================
# LOGGING
# =============================================================================

import logging


def setup_logging() -> None:
    """Configure root logger for the project."""
    log_file = OUTPUT_DIR / "mics_ddml.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(),
        ],
    )


logger = logging.getLogger("mics_ddml")