"""
Robustness checks for the MICS DDML Analysis.

Includes:
- Falsification test (RiskSource=0 placebo)
- Coefficient stability (progressive addition of confounder groups)
- Leave-one-out confounders
- Water storage + handwashing controls
"""

from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from config import (
    BASE_CONFOUNDERS, ROBUSTNESS_CONFOUNDERS, U5_ADDITIONAL_CONFOUNDERS,
    STABILITY_GROUPS, U5_STABILITY_GROUPS, LOO_GROUPS_HH, LOO_GROUPS_U5,
    ANY_TREATMENT, ALL_TREATMENTS, HH_OUTCOMES, U5_OUTCOMES,
    LEARNER_NAMES, SUBGROUP_VAR, SUBGROUP_LABELS,
    logger,
)
from data import prepare_hh_data, prepare_u5_data
from learners import create_learners
from models import estimate_effect, run_analysis, export_results


def _is_u5_outcome(outcome_var: str) -> bool:
    """Check if an outcome is U5-specific (diarrhea)."""
    return outcome_var == "diarrhea"


def run_falsification(
    dt: pd.DataFrame,
    outcomes: List[Dict[str, str]],
    treatments: List[Dict[str, str]],
    learners: Dict[str, Dict],
    dataset_type: str = "HH",
    include_source_ecoli: bool = False,
    include_child_controls: bool = False,
) -> List[Dict[str, Any]]:
    """
    Falsification test: estimate treatment effects on RiskSource=0 subsample.

    When source water has no E.coli contamination, treatment should have
    little or no effect on home contamination. A significant positive effect
    here would suggest confounding bias.
    """
    logger.info("=" * 60)
    logger.info("FALSIFICATION TEST: RiskSource=0 (No Risk at Source)")
    logger.info("=" * 60)

    results = run_analysis(
        dt=dt,
        outcomes=outcomes,
        treatments=treatments,
        learners=learners,
        include_source_ecoli=include_source_ecoli,
        include_child_controls=include_child_controls,
        subgroup_var=SUBGROUP_VAR,
        subgroup_val=0,
        confounder_set="falsification",
        dataset_type=dataset_type,
        prefix="falsification",
    )

    return results


def run_coefficient_stability(
    dt: pd.DataFrame,
    outcomes: List[Dict[str, str]],
    treatments: List[Dict[str, str]],
    learner_name: str = "stacked",
    dataset_type: str = "HH",
) -> List[Dict[str, Any]]:
    """
    Coefficient stability: progressively add confounder groups.

    Shows how the ATE changes as more confounders are controlled for.
    Uses only the stacked learner for computational efficiency.
    """
    logger.info("=" * 60)
    logger.info("COEFFICIENT STABILITY: Progressive addition of confounders")
    logger.info("=" * 60)

    is_u5 = any(_is_u5_outcome(o["var"]) for o in outcomes)
    stability_groups = U5_STABILITY_GROUPS if is_u5 else STABILITY_GROUPS
    learners = {learner_name: create_learners()[learner_name]}

    all_results = []

    for treat in treatments:
        for group_label, group_keys in stability_groups:
            # Build confounder dict from group keys
            if len(group_keys) == 0:
                # No controls: just intercept
                confounder_groups = {}
            else:
                confounder_groups = {k: BASE_CONFOUNDERS[k] for k in group_keys if k in BASE_CONFOUNDERS}
                # Add U5-specific groups
                if is_u5:
                    for k in group_keys:
                        if k in U5_ADDITIONAL_CONFOUNDERS:
                            confounder_groups[k] = U5_ADDITIONAL_CONFOUNDERS[k]

            include_source_ecoli = "source_ecoli" in group_keys
            include_child_controls = "child" in group_keys

            for outcome in outcomes:
                include_src = include_source_ecoli if _is_u5_outcome(outcome["var"]) else False
                include_child = include_child_controls if _is_u5_outcome(outcome["var"]) else False

                res = estimate_effect(
                    dt=dt,
                    outcome_var=outcome["var"],
                    treatment_var=treat["var"],
                    learner_name=learner_name,
                    learner=learners[learner_name],
                    confounder_groups=confounder_groups,
                    include_source_ecoli=include_src,
                    include_child_controls=include_child,
                    confounder_set=f"stability_{group_label}",
                    dataset_type=dataset_type,
                )

                if res is not None:
                    res["stability_step"] = group_label
                    all_results.append(res)

    return all_results


def run_leave_one_out(
    dt: pd.DataFrame,
    outcomes: List[Dict[str, str]],
    treatments: List[Dict[str, str]],
    learner_name: str = "stacked",
    dataset_type: str = "HH",
) -> List[Dict[str, Any]]:
    """
    Leave-one-out confounders: drop one confounder group at a time
    from the full specification to check robustness.
    """
    logger.info("=" * 60)
    logger.info("LEAVE-ONE-OUT: Dropping one confounder group at a time")
    logger.info("=" * 60)

    is_u5 = any(_is_u5_outcome(o["var"]) for o in outcomes)
    loo_groups = LOO_GROUPS_U5 if is_u5 else LOO_GROUPS_HH
    learners = {learner_name: create_learners()[learner_name]}

    all_results = []

    # Full specification first
    full_groups_hh = BASE_CONFOUNDERS.copy()
    full_groups_u5 = {**BASE_CONFOUNDERS, **U5_ADDITIONAL_CONFOUNDERS}

    for treat in treatments:
        # Full specification
        for outcome in outcomes:
            is_u5_out = _is_u5_outcome(outcome["var"])
            fg = full_groups_u5 if is_u5_out else full_groups_hh

            res = estimate_effect(
                dt=dt,
                outcome_var=outcome["var"],
                treatment_var=treat["var"],
                learner_name=learner_name,
                learner=learners[learner_name],
                confounder_groups=fg,
                include_source_ecoli=is_u5_out,
                include_child_controls=is_u5_out,
                confounder_set="loo_full",
                dataset_type=dataset_type,
            )
            if res is not None:
                res["loo_dropped"] = "Full"
                all_results.append(res)

        # Leave-one-out
        for dropped_group, remaining_keys in loo_groups.items():
            if is_u5:
                confounder_groups = {}
                for k in remaining_keys:
                    if k in BASE_CONFOUNDERS:
                        confounder_groups[k] = BASE_CONFOUNDERS[k]
                    if k in U5_ADDITIONAL_CONFOUNDERS:
                        confounder_groups[k] = U5_ADDITIONAL_CONFOUNDERS[k]
                include_src = "source_ecoli" in remaining_keys
                include_child = "child" in remaining_keys
            else:
                confounder_groups = {k: BASE_CONFOUNDERS[k] for k in remaining_keys if k in BASE_CONFOUNDERS}
                include_src = False
                include_child = False

            for outcome in outcomes:
                is_u5_out = _is_u5_outcome(outcome["var"])
                res = estimate_effect(
                    dt=dt,
                    outcome_var=outcome["var"],
                    treatment_var=treat["var"],
                    learner_name=learner_name,
                    learner=learners[learner_name],
                    confounder_groups=confounder_groups,
                    include_source_ecoli=include_src if is_u5_out else False,
                    include_child_controls=include_child if is_u5_out else False,
                    confounder_set=f"loo_no_{dropped_group}",
                    dataset_type=dataset_type,
                )
                if res is not None:
                    res["loo_dropped"] = dropped_group
                    all_results.append(res)

    return all_results


def run_water_storage_handwashing(
    dt: pd.DataFrame,
    outcomes: List[Dict[str, str]],
    treatments: List[Dict[str, str]],
    learners: Dict[str, Dict],
    dataset_type: str = "HH",
) -> List[Dict[str, Any]]:
    """
    Add water storage and handwashing controls as a robustness check.

    Note: SoapandWater has ~27% missing values.
    Observations with missing data are dropped.
    """
    logger.info("=" * 60)
    logger.info("WATER STORAGE + HANDWASHING: Adding WQ12 + SoapandWater")
    logger.info("=" * 60)

    results = run_analysis(
        dt=dt,
        outcomes=outcomes,
        treatments=treatments,
        learners=learners,
        include_robustness=True,
        confounder_set="water_hw",
        dataset_type=dataset_type,
        prefix="water_hw",
    )

    return results


def run_all_robustness(
    dt: pd.DataFrame,
    dataset_type: str = "HH",
    outcomes: Optional[List[Dict[str, str]]] = None,
    treatments: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run all robustness checks for a given dataset.

    Returns dict mapping robustness type to results.
    """
    if outcomes is None:
        outcomes = HH_OUTCOMES if dataset_type == "HH" else U5_OUTCOMES
    if treatments is None:
        treatments = [ANY_TREATMENT]

    learners = create_learners()
    is_u5 = dataset_type == "U5"

    all_results = {}

    # 1. Falsification test
    try:
        all_results["falsification"] = run_falsification(
            dt=dt, outcomes=outcomes, treatments=treatments,
            learners=learners, dataset_type=dataset_type,
            include_source_ecoli=is_u5,
            include_child_controls=is_u5,
        )
    except Exception as e:
        logger.warning(f"    Falsification failed: {e}")
        all_results["falsification"] = []

    # 2. Coefficient stability (stacked learner only for efficiency)
    try:
        all_results["stability"] = run_coefficient_stability(
            dt=dt, outcomes=outcomes, treatments=treatments,
            learner_name="stacked", dataset_type=dataset_type,
        )
    except Exception as e:
        logger.warning(f"    Coefficient stability failed: {e}")
        all_results["stability"] = []

    # 3. Leave-one-out (stacked learner only)
    try:
        all_results["loo"] = run_leave_one_out(
            dt=dt, outcomes=outcomes, treatments=treatments,
            learner_name="stacked", dataset_type=dataset_type,
        )
    except Exception as e:
        logger.warning(f"    Leave-one-out failed: {e}")
        all_results["loo"] = []

    # 4. Water storage + handwashing
    try:
        all_results["water_hw"] = run_water_storage_handwashing(
            dt=dt, outcomes=outcomes, treatments=treatments,
            learners={"stacked": learners["stacked"]},
            dataset_type=dataset_type,
        )
    except Exception as e:
        logger.warning(f"    Water storage + handwashing failed: {e}")
        all_results["water_hw"] = []

    return all_results