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

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Do file/python/config.py -> repo root
DATA_DIR = PROJECT_ROOT / "Data" / "3. Final"

HH_DATA_FILE = DATA_DIR / "MASTER_MICS_FINAL.dta"
U5_DATA_FILE = DATA_DIR / "MASTER_MICS_FINAL_U5.dta"

import os as _os
# Output / figure dirs are env-overridable so a parallel run (e.g. the Rob_1
# overlap-robustness mirror) can write into an isolated tree with the SAME
# filenames as the headline, with no logic drift.  Checkpoints follow OUTPUT_DIR.
_OUT_ENV = _os.environ.get("MICS_OUTPUT_DIR")
_FIG_ENV = _os.environ.get("MICS_FIGURE_DIR")
OUTPUT_DIR = Path(_OUT_ENV) if _OUT_ENV else PROJECT_ROOT / "Output"
CHECKPOINT_DIR = OUTPUT_DIR / "Checkpoints"
FIGURE_DIR = Path(_FIG_ENV) if _FIG_ENV else PROJECT_ROOT / "Figures"

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

ALL_OUTCOMES = [
    {"var": "SomeRiskHome", "label": "Some Risk Home", "desc": "E.coli 1-100 CFU at home (binary)", "type": "binary"},
    {"var": "VeryHighRiskHome", "label": "Very High Risk Home", "desc": "E.coli >=101 CFU at home (binary)", "type": "binary"},
    {"var": "diarrhea", "label": "Diarrhea", "desc": "Child had diarrhea in last 2 weeks (binary)", "type": "binary"},
]

HH_OUTCOMES = list(ALL_OUTCOMES)
U5_OUTCOMES = list(ALL_OUTCOMES)

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
#
# This vector mirrors the Stata specification exactly:
#
#   Controls:  i.windex5 i.helevel i.country_cat i.urban i.WS1_g ///
#              Any_U5 Girls_less_than15 Boys_15or_less i.Toilet i.wq27_decile
#   Controls_U5 (child only):  i.male i.age
#   Cluster:   HHID  (U5 only)
#
# Notes:
#   - All `i.<var>` categoricals are expanded to drop-first dummy sets here.
#   - source_ecoli (RiskSource) and num_children are NOT controls: RiskSource is
#     reserved as a heterogeneity splitter (GATE), and child counts are captured
#     by the Any_U5 / Girls_less_than15 / Boys_15or_less indicators instead.
#   - Child age/sex enter ONLY for U5 (U5_ADDITIONAL_CONFOUNDERS).

BASE_CONFOUNDERS = {
    "wealth": ["wealth_q2", "wealth_q3", "wealth_q4", "wealth_q5"],        # i.windex5, ref: q1
    "education": ["edu_primary", "edu_secondary", "edu_missing"],          # i.helevel, ref: none
    "country": ["country_cat"],         # i.country_cat -> country_cat_* drop-first (FE)
    "urban": ["urban_bin"],             # i.urban
    "water_source": ["water_source"],   # i.WS1_g -> ws1g_* drop-first
    "hh_demog": ["Any_U5", "Girls_less_than15", "Boys_15or_less"],  # binary indicators
    "sanitation": ["toilet_pit_latrine", "toilet_open_defecation", "toilet_missing"],  # i.Toilet, ref: flush
    "wq27": ["wq27_decile"],            # i.wq27_decile -> wq27_d_* drop-first
}

# Child controls — U5 (child-level) dataset ONLY.
U5_ADDITIONAL_CONFOUNDERS = {
    "child": ["child_age", "child_sex_male"],  # i.age (-> child_age_* drop-first) + i.male
}

# =============================================================================
# ML LEARNERS
# =============================================================================

LEARNER_NAMES = ["ols", "lasso", "ridge", "enet", "rf", "xgb", "stacked"]

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
# CLUSTERING  (per dataset)
# =============================================================================
#
# Clustering is applied to the U5 (child-level) analysis ONLY, where multiple
# children nest within a household (mean ~1.4 children / HH).  Clusters are the
# household id (HHID).  The HH (E.coli) dataset has one row per household and no
# HHID, so it is estimated i.i.d. (unclustered).

CLUSTER_VARS = {
    "HH": None,        # E.coli analysis: unclustered
    "U5": "HHID",      # child analysis: cluster on household
}

# Back-compat aliases (HH default)
CLUSTER_VAR = "HHID"


def cluster_var_for(dataset_type: str):
    """Return the cluster variable for a dataset, or None if unclustered."""
    return CLUSTER_VARS.get(dataset_type, None)


# =============================================================================
# HETEROGENEITY: CATE (smooth basis) and GATE (group indicators)
# =============================================================================
#
# Both are projection-based estimands computed on a SINGLE fitted DoubleMLIRM
# (n_rep must be 1 for .cate()/.gate() in doubleml 0.11.x), then projecting the
# orthogonal signal onto a basis with uniform (joint) confidence bands.

# CATE moderators: continuous-ish covariates -> spline basis via .cate().
#   `kind`: "spline" (smooth) or "onehot" (categorical -> degenerate CATE = GATE).
#   Only genuinely continuous / well-supported ordered counts belong here; every
#   discrete-cell moderator lives in GATE_GROUPS instead.
CATE_MODERATORS = [
    # Continuous wealth factor score — full support -> smooth df=5 spline curve.
    {"var": "wscore",        "label": "Wealth index (continuous)",  "kind": "spline", "df": 5, "datasets": ["HH", "U5"]},
    # Raw child count (0-10, 11 support points) -> smooth cubic family-size trend.
    # df>=4 required by patsy bs for degree=3 + include_intercept.
    {"var": "num_children",  "label": "Number of children",         "kind": "spline", "df": 4, "datasets": ["HH", "U5"]},
]

# GATE groups: categorical / discrete-cell splitters -> group-indicator basis via .gate().
GATE_GROUPS = [
    {"var": "RiskSource",       "label": "Source E.coli risk",      "datasets": ["HH", "U5"]},
    {"var": "helevel",          "label": "Education",               "datasets": ["HH", "U5"]},
    {"var": "windex5",          "label": "Wealth quintile",         "datasets": ["HH", "U5"]},
    {"var": "country_cat",      "label": "Country",                 "datasets": ["HH", "U5"]},
    {"var": "Toilet",           "label": "Sanitation",              "datasets": ["HH", "U5"]},
    {"var": "WS1_g",            "label": "Water source",            "datasets": ["HH", "U5"]},
    {"var": "num_children_bin", "label": "Number of children (0-4+)", "datasets": ["HH", "U5"]},
    {"var": "child_age",        "label": "Child age (years)",       "datasets": ["U5"]},
]

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

# Toilet (sanitation) coding
TOILET_CODES = {1: "flush", 2: "pit_latrine", 3: "open_defecation", 98: "missing"}

# WS1_g (grouped water source) coding
WS1G_CODES = {
    11: "piped",
    21: "tube/well/borehole",
    31: "protected_well/spring",
    32: "unprotected_well/spring",
    51: "surface/rain",
    91: "packaged/bottled",
    96: "other",
}

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