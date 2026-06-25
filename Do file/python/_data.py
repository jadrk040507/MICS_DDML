"""
Data loading and preparation for the MICS DDML Analysis.

Handles both HH (household-level, E.coli outcomes) and U5 (child-level, diarrhea).
Constructs all derived variables and model matrices for DoubleML estimation.
"""

import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import pyreadstat
from _config import (
    HH_DATA_FILE, U5_DATA_FILE, WQ15G_TREATMENT_MAP, HELEVEL_CODES,
    RISKSOURCE_CODES, BASE_CONFOUNDERS,
    U5_ADDITIONAL_CONFOUNDERS, CLUSTER_VAR,
    logger,
)


def prepare_hh_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load and prepare the household-level dataset for E.coli outcomes."""
    filepath = filepath or HH_DATA_FILE
    dt, _ = pyreadstat.read_dta(filepath)
    dt = _construct_common_variables(dt)
    dt = _construct_ecoli_outcomes(dt)
    dt = _construct_treatment_variables(dt)
    dt = _construct_hh_confounders(dt)
    dt["dataset"] = "HH"
    return dt


def prepare_u5_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load and prepare the child-level dataset for diarrhea outcomes."""
    filepath = filepath or U5_DATA_FILE
    dt, _ = pyreadstat.read_dta(filepath)
    dt = _construct_common_variables(dt)
    dt = _construct_ecoli_outcomes(dt)
    dt = _construct_treatment_variables(dt)
    dt = _construct_hh_confounders(dt)
    dt = _construct_u5_variables(dt)
    dt["dataset"] = "U5"
    return dt


def _construct_common_variables(dt: pd.DataFrame) -> pd.DataFrame:
    """Construct variables common to both datasets."""
    dt = dt.copy()

    # Wealth quintile dummies from windex5 (1=poorest to 5=richest)
    for q in range(1, 6):
        dt[f"wealth_q{q}"] = (dt["windex5"] == q).astype(int)

    # Education dummies from helevel (0=none, 1=primary, 2=secondary, 98=missing)
    dt["edu_none"] = (dt["helevel"] == 0).astype(int)
    dt["edu_primary"] = (dt["helevel"] == 1).astype(int)
    dt["edu_secondary"] = (dt["helevel"] == 2).astype(int)
    dt["edu_missing"] = (dt["helevel"] == 98).astype(int)

    # Urban binary
    dt["urban_bin"] = dt["urban"].astype(int)

    # Improved latrine
    dt["improved_latrine"] = dt["improved_latrine"].fillna(0).astype(int)

    # Number of children (fill missing with 0, then cap)
    dt["num_children"] = dt["HHCHILDREN"].fillna(0).astype(int)
    dt.loc[dt["num_children"] > 10, "num_children"] = 10

    # Water source dummies from WS1_g (grouped) — i.WS1_g
    ws1g_cats = sorted(dt["WS1_g"].dropna().unique())
    for cat in ws1g_cats:
        dt[f"ws1g_{int(cat)}"] = (dt["WS1_g"] == cat).astype(int)

    # Country fixed effects from country_cat (drop first) — i.country_cat
    country_dummies = pd.get_dummies(
        dt["country_cat"].astype("Int64"), prefix="country_cat", drop_first=True
    ).astype(int)
    dt = pd.concat([dt, country_dummies], axis=1)

    # WQ27 decile dummies (drop first) — i.wq27_decile
    wq27_dummies = pd.get_dummies(
        dt["wq27_decile"].astype("Int64"), prefix="wq27_d", drop_first=True
    ).astype(int)
    dt = pd.concat([dt, wq27_dummies], axis=1)

    # Household demographic indicators (binary 0/1) — enter as-is
    for col in ["Any_U5", "Girls_less_than15", "Boys_15or_less"]:
        if col in dt.columns:
            dt[col] = dt[col].fillna(0).astype(int)

    # Risk source dummies (kept for GATE splitting / diagnostics; NOT a control)
    for rs_val in [0, 1, 2]:
        dt[f"risk_source_{rs_val}"] = (dt["RiskSource"] == rs_val).astype(int)

    # Binned child count (0,1,2,3,4+) for a stable one-hot CATE moderator:
    # the raw count has integer-only support, so a spline basis is unstable.
    if "num_children" in dt.columns:
        dt["num_children_bin"] = dt["num_children"].clip(upper=4).astype(int)

    # Continuous wealth factor score (for CATE moderator)
    if "wscore" in dt.columns:
        dt["wscore"] = dt["wscore"].astype(float)

    # Cluster variable (U5 only): cluster on HHID when present
    if CLUSTER_VAR in dt.columns:
        dt[CLUSTER_VAR] = dt[CLUSTER_VAR].fillna(-1)

    # RiskSource as integer
    dt["risk_source"] = dt["RiskSource"].astype(int)

    # Region
    dt["region"] = dt["Region"].fillna(0).astype(int)

    # Diarrhea outcome: HH-level child diarrhea is only in the HH dataset,
    # while U5 already has a diarrhea variable.
    if "diarrhea" in dt.columns:
        dt["diarrhea"] = dt["diarrhea"].astype("Int64")
    elif "HH_child_diarrhea" in dt.columns:
        dt["diarrhea"] = dt["HH_child_diarrhea"].astype("Int64")

    return dt


def _construct_ecoli_outcomes(dt: pd.DataFrame) -> pd.DataFrame:
    """Construct E.coli outcome variables."""
    dt = dt.copy()
    # SomeRiskHome already exists: 1 if RiskHome >= 1
    if "SomeRiskHome" not in dt.columns:
        dt["SomeRiskHome"] = (dt["RiskHome"] >= 1).astype(int)
    else:
        dt["SomeRiskHome"] = dt["SomeRiskHome"].astype(int)

    # VeryHighRiskHome already exists: 1 if RiskHome >= 2
    if "VeryHighRiskHome" not in dt.columns:
        dt["VeryHighRiskHome"] = (dt["RiskHome"] >= 2).astype(int)
    else:
        dt["VeryHighRiskHome"] = dt["VeryHighRiskHome"].astype(int)

    return dt


def _construct_treatment_variables(dt: pd.DataFrame) -> pd.DataFrame:
    """Construct treatment variables from WQ15_g coding and treatment count."""
    dt = dt.copy()

    # Any treatment (binary)
    dt["water_treatment"] = (dt["WQ15_g"] > 0).astype(int)
    if "water_treatment" in dt.columns:
        dt["water_treatment"] = dt["water_treatment"].fillna(0).astype(int)

    # Specific treatments: create dummies from WQ15_g
    for code, name in WQ15G_TREATMENT_MAP.items():
        dt[name] = (dt["WQ15_g"] == code).astype(int)

    # Count how many treatment methods are used (from WQ15A-WQ15Z columns).
    # These columns contain letters ('A','B',...) or empty/'?' for unused.
    treat_cols = ["WQ15A", "WQ15B", "WQ15C", "WQ15D", "WQ15E", "WQ15F",
                  "WQ15G", "WQ15H", "WQ15X", "WQ15Z"]
    existing_treat_cols = [c for c in treat_cols if c in dt.columns]
    if existing_treat_cols:
        # Count non-empty, non-'?' entries per row
        method_flags = dt[existing_treat_cols].apply(
            lambda x: x.str.strip().str.len() > 0
        )
        dt["treatment_count"] = method_flags.sum(axis=1)
    else:
        # Fallback: only the primary method recorded in WQ15_g
        dt["treatment_count"] = dt["water_treatment"]

    # Single-method indicator for clean specific-treatment estimation
    dt["single_method"] = (dt["treatment_count"] == 1).astype(int)

    return dt


def construct_rob1(
    dt: pd.DataFrame,
    country_col: str = "country_cat",
    share_lo: float = 0.05,
    min_count: int = 10,
) -> pd.DataFrame:
    """Add ``Rob_1`` = 1 for TREATED households in weak-support country cells.

    A (country, treatment-level) cell is weak-support when the method's within-
    country share is below ``share_lo`` or fewer than ``min_count`` households use
    it.  Only treated levels (WQ15_g level != 0) are flagged; the no-treatment
    baseline is never dropped for being dominant.  This isolates the orphan-
    treated, no-overlap observations (notably chlorine, concentrated in a few
    countries).  The Rob_1 overlap-robustness mirror re-estimates on Rob_1 == 0.
    """
    from _apos import _build_treat_cat
    d = dt.copy()
    d["_lvl"] = _build_treat_cat(d)
    valid = d[country_col].notna() & d["_lvl"].notna()
    cnt = d[valid].groupby([country_col, "_lvl"]).size().rename("cnt").reset_index()
    tot = d[valid].groupby(country_col).size().rename("ctot").reset_index()
    cell = cnt.merge(tot, on=country_col)
    cell["share"] = cell["cnt"] / cell["ctot"]
    cell["weak"] = (((cell["share"] < share_lo) | (cell["cnt"] < min_count))
                    & (cell["_lvl"] != 0)).astype(int)
    d = d.merge(cell[[country_col, "_lvl", "weak"]], on=[country_col, "_lvl"], how="left")
    d["Rob_1"] = d["weak"].fillna(0).astype(int)
    return d.drop(columns=["_lvl", "weak"])


def _construct_hh_confounders(dt: pd.DataFrame) -> pd.DataFrame:
    """Construct household-level confounders (toilet dummies)."""
    dt = dt.copy()

    # Toilet dummies from Toilet (1=flush, 2=pit, 3=open, 98=missing)
    if "Toilet" in dt.columns:
        for cat, label in [(1, "flush"), (2, "pit_latrine"), (3, "open_defecation"), (98, "toilet_missing")]:
            dt[f"toilet_{label}"] = (dt["Toilet"] == cat).astype(int)

    return dt


def _construct_u5_variables(dt: pd.DataFrame) -> pd.DataFrame:
    """Construct child-level variables for the U5 dataset."""
    dt = dt.copy()

    # Child age (years, 0-4) — raw value kept for CATE moderator
    dt["child_age"] = dt["age"].fillna(0).astype(int)

    # Child age dummies (drop first) — i.age, used as a control
    age_dummies = pd.get_dummies(dt["child_age"], prefix="child_age", drop_first=True).astype(int)
    dt = pd.concat([dt, age_dummies], axis=1)

    # Child sex: male variable (1=male, 0=female) — i.male
    dt["child_sex_male"] = dt["male"].fillna(0).astype(int)

    # Girls under 15 and Boys under 15 (binary indicators)
    dt["girls_under15"] = dt["Girls_less_than15"].fillna(0).astype(int)
    dt["boys_under15"] = dt["Boys_15or_less"].fillna(0).astype(int)

    return dt


def create_model_matrix(
    dt: pd.DataFrame,
    outcome_var: str,
    confounder_groups: Optional[Dict[str, List[str]]] = None,
    return_names: bool = False,
):
    """
    Build the confounder matrix X for DoubleML estimation.

    Parameters
    ----------
    dt : DataFrame with all constructed variables.
    outcome_var : Name of the outcome variable (used for selecting columns).
    confounder_groups : Dict mapping group names to lists of column names.
        If None, uses BASE_CONFOUNDERS.

    Returns
    -------
    X : numpy array of shape (n, p) with confounders.
    dt : DataFrame with any new columns added.
    """
    if confounder_groups is None:
        confounder_groups = BASE_CONFOUNDERS

    dt = dt.copy()
    columns = []

    def _expand(prefix: str, drop_first: bool) -> list:
        cols = sorted(
            [c for c in dt.columns if c.startswith(prefix)],
            key=lambda c: (len(c), c),
        )
        if drop_first and len(cols) > 1:
            return cols[1:]
        return cols

    for group_name, group_cols in confounder_groups.items():
        for col in group_cols:
            if col == "water_source":
                # i.WS1_g -> ws1g_* dummies, drop first (avoid collinearity)
                columns.extend(_expand("ws1g_", drop_first=True))
            elif col == "country_cat":
                # i.country_cat -> country_cat_* dummies (already drop-first built)
                columns.extend(_expand("country_cat_", drop_first=False))
            elif col == "wq27_decile":
                # i.wq27_decile -> wq27_d_* dummies (already drop-first built)
                columns.extend(_expand("wq27_d_", drop_first=False))
            elif col == "child_age":
                # i.age -> child_age_* dummies (already drop-first built)
                columns.extend(_expand("child_age_", drop_first=False))
            else:
                columns.append(col)

    # Filter to columns that actually exist in the data, preserving order and removing duplicates
    seen = set()
    available_columns = []
    for c in columns:
        if c not in seen and c in dt.columns:
            available_columns.append(c)
            seen.add(c)

    # Build matrix
    X = dt[available_columns].values.astype(float)

    # Replace inf with nan
    X[np.isinf(X)] = np.nan

    if return_names:
        return X, dt, available_columns
    return X, dt


def get_summary(dt: pd.DataFrame, dataset_type: str = "HH") -> dict:
    """Print and return summary statistics for the prepared data."""
    summary = {}
    summary["n_total"] = len(dt)
    summary["dataset"] = dataset_type

    # Outcomes
    if "SomeRiskHome" in dt.columns:
        summary["some_risk_home_mean"] = dt["SomeRiskHome"].mean()
        summary["some_risk_home_n"] = dt["SomeRiskHome"].notna().sum()
    if "VeryHighRiskHome" in dt.columns:
        summary["very_high_risk_home_mean"] = dt["VeryHighRiskHome"].mean()
        summary["very_high_risk_home_n"] = dt["VeryHighRiskHome"].notna().sum()
    if "diarrhea" in dt.columns:
        summary["diarrhea_mean"] = dt["diarrhea"].mean()
        summary["diarrhea_n"] = dt["diarrhea"].notna().sum()

    # Treatment
    summary["any_treatment_mean"] = dt["water_treatment"].mean()
    summary["any_treatment_n"] = dt["water_treatment"].notna().sum()
    if "treat_boil" in dt.columns:
        summary["boil_mean"] = dt["treat_boil"].mean()
    if "treat_chlorine" in dt.columns:
        summary["chlorine_mean"] = dt["treat_chlorine"].mean()
    if "treat_filter" in dt.columns:
        summary["filter_mean"] = dt["treat_filter"].mean()
    if "treat_other" in dt.columns:
        summary["other_mean"] = dt["treat_other"].mean()

    # RiskSource breakdown
    if "risk_source" in dt.columns:
        for rs_val in [0, 1, 2]:
            n = (dt["risk_source"] == rs_val).sum()
            label = RISKSOURCE_CODES.get(rs_val, str(rs_val))
            summary[f"risk_source_{label}_n"] = n

    # Print summary
    lines = [
        "=" * 60,
        f"Dataset: {dataset_type} | N = {summary['n_total']:,}",
        "=" * 60,
    ]
    for key, val in summary.items():
        if key not in ["dataset"]:
            lines.append(f"  {key}: {val}")
    lines.append("=" * 60)
    logger.info("\n".join(lines))

    return summary
