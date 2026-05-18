"""
CATE by education subgroups.

Estimates DDML separately for each household head education level,
using the stacked ensemble (without OLS to avoid quasi-complete separation).
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    HH_DATA_FILE, U5_DATA_FILE, HH_OUTCOMES, U5_OUTCOMES,
    ANY_TREATMENT, EDUCATION_SUBGROUP_VAR, EDUCATION_SUBGROUP_LABELS,
    EDUCATION_SUBGROUP_VALUES, MIN_OBSERVATIONS,
    OUTPUT_DIR,
)
from data import prepare_hh_data, prepare_u5_data, get_summary
from learners import create_learners
from models import estimate_effect, export_results, export_results_csv, print_significant_effects


def main():
    print("=" * 70)
    print("MICS DDML: CATE BY EDUCATION SUBGROUPS")
    print("=" * 70)

    # Load data
    hh_dt = prepare_hh_data(HH_DATA_FILE)
    u5_dt = prepare_u5_data(U5_DATA_FILE)

    # Use stacked ensemble only (skip OLS for small subgroups)
    learners = {"stacked": create_learners()["stacked"]}

    all_results = []

    # =========================================================================
    # HH Dataset: E.coli outcomes by education
    # =========================================================================
    for edu_val in EDUCATION_SUBGROUP_VALUES:
        edu_label = EDUCATION_SUBGROUP_LABELS.get(edu_val, str(edu_val))
        print(f"\n--- HH Dataset: Education={edu_val} ({edu_label}) ---")

        dt_sub = hh_dt[hh_dt[EDUCATION_SUBGROUP_VAR] == edu_val].copy()
        n = len(dt_sub)
        print(f"  N observations: {n:,}")

        if n < MIN_OBSERVATIONS:
            print(f"  Skipping: too few observations")
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

    # =========================================================================
    # U5 Dataset: Diarrhea by education
    # =========================================================================
    for edu_val in EDUCATION_SUBGROUP_VALUES:
        edu_label = EDUCATION_SUBGROUP_LABELS.get(edu_val, str(edu_val))
        print(f"\n--- U5 Dataset: Education={edu_val} ({edu_label}) ---")

        dt_sub = u5_dt[u5_dt[EDUCATION_SUBGROUP_VAR] == edu_val].copy()
        n = len(dt_sub)
        print(f"  N observations: {n:,}")

        if n < MIN_OBSERVATIONS:
            print(f"  Skipping: too few observations")
            continue

        for outcome in U5_OUTCOMES:
            res = estimate_effect(
                dt=dt_sub,
                outcome_var=outcome["var"],
                treatment_var=ANY_TREATMENT["var"],
                learner_name="stacked",
                learner=learners["stacked"],
                include_source_ecoli=True,
                include_child_controls=True,
                dataset_type="U5",
                prefix=f"u5_edu{edu_val}",
            )
            if res is not None:
                res["subgroup_var"] = "helevel"
                res["subgroup_val"] = edu_val
                all_results.append(res)

    # =========================================================================
    # Export
    # =========================================================================
    print("\n" + "=" * 70)
    print("CATE BY EDUCATION COMPLETE")
    print("=" * 70)
    print_significant_effects(all_results)

    export_results(all_results, "results_cate_education.pkl")
    export_results_csv(all_results, "results_cate_education.csv")


if __name__ == "__main__":
    main()