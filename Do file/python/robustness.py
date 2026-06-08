"""
Robustness checks for the MICS DDML Analysis.

Includes:
- Falsification test (RiskSource=0 placebo)
- Coefficient stability (progressive addition of confounder groups)
- Leave-one-out confounders
- Water storage + handwashing controls
"""

from typing import Dict, List, Optional, Any

from config import (
    BASE_CONFOUNDERS, U5_ADDITIONAL_CONFOUNDERS,
    STABILITY_GROUPS, U5_STABILITY_GROUPS, LOO_GROUPS_HH, LOO_GROUPS_U5,
    ANY_TREATMENT, HH_OUTCOMES, U5_OUTCOMES,
    SUBGROUP_VAR,
    logger,
)
from learners import create_learners
from models import estimate_effect, run_analysis


def _is_u5_outcome(outcome_var: str) -> bool:
    """Check if an outcome is U5-specific (diarrhea)."""
    return outcome_var == "diarrhea"


def run_falsification(
    dt: pd.DataFrame,
    outcomes: List[Dict[str, str]],
    treatments: List[Dict[str, str]],
    learners: Dict[str, Dict],
    confounder_groups: Optional[Dict[str, List[str]]] = None,
    dataset_type: str = "HH",
) -> List[Dict[str, Any]]:
    """
    Falsification test: estimate treatment effects on RiskSource=0 subsample.
    """
    logger.info("=" * 60)
    logger.info("FALSIFICATION TEST: RiskSource=0 (No Risk at Source)")
    logger.info("=" * 60)

    if confounder_groups is None:
        confounder_groups = BASE_CONFOUNDERS.copy()

    results = run_analysis(
        dt=dt,
        outcomes=outcomes,
        treatments=treatments,
        learners=learners,
        confounder_groups=confounder_groups,
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
            if len(group_keys) == 0:
                confounder_groups = {}
            else:
                confounder_groups = {k: BASE_CONFOUNDERS[k] for k in group_keys if k in BASE_CONFOUNDERS}
                if is_u5:
                    for k in group_keys:
                        if k in U5_ADDITIONAL_CONFOUNDERS:
                            confounder_groups[k] = U5_ADDITIONAL_CONFOUNDERS[k]

            for outcome in outcomes:
                res = estimate_effect(
                    dt=dt,
                    outcome_var=outcome["var"],
                    treatment_var=treat["var"],
                    learner_name=learner_name,
                    learner=learners[learner_name],
                    confounder_groups=confounder_groups,
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
    Leave-one-out confounders: drop one confounder group at a time.
    """
    logger.info("=" * 60)
    logger.info("LEAVE-ONE-OUT: Dropping one confounder group at a time")
    logger.info("=" * 60)

    is_u5 = any(_is_u5_outcome(o["var"]) for o in outcomes)
    loo_groups = LOO_GROUPS_U5 if is_u5 else LOO_GROUPS_HH
    learners = {learner_name: create_learners()[learner_name]}

    all_results = []

    for treat in treatments:
        for outcome in outcomes:
            is_u5_out = _is_u5_outcome(outcome["var"])
            fg = {**BASE_CONFOUNDERS, **U5_ADDITIONAL_CONFOUNDERS} if is_u5_out else BASE_CONFOUNDERS.copy()

            res = estimate_effect(
                dt=dt,
                outcome_var=outcome["var"],
                treatment_var=treat["var"],
                learner_name=learner_name,
                learner=learners[learner_name],
                confounder_groups=fg,
                confounder_set="loo_full",
                dataset_type=dataset_type,
            )
            if res is not None:
                res["loo_dropped"] = "Full"
                all_results.append(res)

        for dropped_group, remaining_keys in loo_groups.items():
            if is_u5:
                confounder_groups = {}
                for k in remaining_keys:
                    if k in BASE_CONFOUNDERS:
                        confounder_groups[k] = BASE_CONFOUNDERS[k]
                    if k in U5_ADDITIONAL_CONFOUNDERS:
                        confounder_groups[k] = U5_ADDITIONAL_CONFOUNDERS[k]
            else:
                confounder_groups = {k: BASE_CONFOUNDERS[k] for k in remaining_keys if k in BASE_CONFOUNDERS}

            for outcome in outcomes:
                res = estimate_effect(
                    dt=dt,
                    outcome_var=outcome["var"],
                    treatment_var=treat["var"],
                    learner_name=learner_name,
                    learner=learners[learner_name],
                    confounder_groups=confounder_groups,
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
    confounder_groups: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Add water storage and handwashing controls as a robustness check.

    Note: SoapandWater has ~27% missing values.
    Observations with missing data are dropped.
    """
    logger.info("=" * 60)
    logger.info("WATER STORAGE + HANDWASHING: Adding WQ12 + SoapandWater")
    logger.info("=" * 60)

    # Merge base confounders with robustness variables
    from config import ROBUSTNESS_CONFOUNDERS
    if confounder_groups is None:
        confounder_groups = BASE_CONFOUNDERS.copy()
    cg = {**confounder_groups, **ROBUSTNESS_CONFOUNDERS}

    results = run_analysis(
        dt=dt,
        outcomes=outcomes,
        treatments=treatments,
        learners=learners,
        confounder_groups=cg,
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
    confounder_groups: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run all robustness checks for a given dataset.

    Returns dict mapping robustness type to results.
    """
    if outcomes is None:
        outcomes = HH_OUTCOMES if dataset_type == "HH" else U5_OUTCOMES
    if treatments is None:
        treatments = [ANY_TREATMENT]
    if confounder_groups is None:
        confounder_groups = BASE_CONFOUNDERS.copy()
        if dataset_type == "U5":
            confounder_groups = {**confounder_groups, **U5_ADDITIONAL_CONFOUNDERS}

    learners = create_learners()

    all_results = {}

    # 1. Falsification test
    try:
        all_results["falsification"] = run_falsification(
            dt=dt, outcomes=outcomes, treatments=treatments,
            learners=learners, dataset_type=dataset_type,
            confounder_groups=confounder_groups,
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
            confounder_groups=confounder_groups,
        )
    except Exception as e:
        logger.warning(f"    Water storage + handwashing failed: {e}")
        all_results["water_hw"] = []

    return all_results


import pandas as pd