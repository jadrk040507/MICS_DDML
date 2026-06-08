"""
CATE by education subgroups.

Estimates DDML separately for each household head education level,
using the stacked ensemble.
"""

from config import (
    HH_OUTCOMES, U5_OUTCOMES, ANY_TREATMENT,
    EDUCATION_SUBGROUP_VAR, EDUCATION_SUBGROUP_LABELS,
    EDUCATION_SUBGROUP_VALUES, MIN_OBSERVATIONS,
    logger,
)
from models import estimate_effect
from runners import setup_environment, load_data, select_learners, save_results


def main():
    setup_environment()
    logger.info("=" * 70)
    logger.info("MICS DDML: CATE BY EDUCATION SUBGROUPS")
    logger.info("=" * 70)

    hh_dt, u5_dt = load_data()

    # Use stacked ensemble only (skip OLS for small subgroups)
    learners = select_learners(names=["stacked"])

    all_results = []

    # HH Dataset: E.coli outcomes by education
    for edu_val in EDUCATION_SUBGROUP_VALUES:
        edu_label = EDUCATION_SUBGROUP_LABELS.get(edu_val, str(edu_val))
        logger.info(f"\n--- HH Dataset: Education={edu_val} ({edu_label}) ---")

        dt_sub = hh_dt[hh_dt[EDUCATION_SUBGROUP_VAR] == edu_val].copy()
        n = len(dt_sub)
        logger.info(f"  N observations: {n:,}")

        if n < MIN_OBSERVATIONS:
            logger.info("  Skipping: too few observations")
            continue

        for outcome in HH_OUTCOMES:
            res = estimate_effect(
                dt=dt_sub,
                outcome_var=outcome["var"],
                treatment_var=ANY_TREATMENT["var"],
                learner_name="stacked",
                learner=learners["stacked"],
                dataset_type="HH",
                prefix=f"hh_edu{edu_val}",
            )
            if res is not None:
                res["subgroup_var"] = "helevel"
                res["subgroup_val"] = edu_val
                all_results.append(res)

    # U5 Dataset: Diarrhea by education
    for edu_val in EDUCATION_SUBGROUP_VALUES:
        edu_label = EDUCATION_SUBGROUP_LABELS.get(edu_val, str(edu_val))
        logger.info(f"\n--- U5 Dataset: Education={edu_val} ({edu_label}) ---")

        dt_sub = u5_dt[u5_dt[EDUCATION_SUBGROUP_VAR] == edu_val].copy()
        n = len(dt_sub)
        logger.info(f"  N observations: {n:,}")

        if n < MIN_OBSERVATIONS:
            logger.info("  Skipping: too few observations")
            continue

        for outcome in U5_OUTCOMES:
            res = estimate_effect(
                dt=dt_sub,
                outcome_var=outcome["var"],
                treatment_var=ANY_TREATMENT["var"],
                learner_name="stacked",
                learner=learners["stacked"],
                dataset_type="U5",
                prefix=f"u5_edu{edu_val}",
            )
            if res is not None:
                res["subgroup_var"] = "helevel"
                res["subgroup_val"] = edu_val
                all_results.append(res)

    save_results(all_results, tag="cate_education")


if __name__ == "__main__":
    main()