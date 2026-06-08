"""
Quick debug run of the MICS DDML pipeline.
Uses a sample of data and only 2 learners for fast iteration.
"""

import numpy as np
from config import HH_OUTCOMES, U5_OUTCOMES, ANY_TREATMENT, SPECIFIC_TREATMENTS
from data import prepare_hh_data, prepare_u5_data, get_summary
from models import run_analysis, export_results_csv, print_significant_effects
from runners import setup_environment, load_data, select_learners


DEBUG_LEARNER_NAMES = ["ols", "stacked"]


def main():
    setup_environment()
    print("=" * 70)
    print("MICS DDML: QUICK DEBUG RUN")
    print("=" * 70)

    hh_dt, u5_dt = load_data()

    # Sample 20% for speed
    hh_sample = hh_dt.sample(frac=0.2, random_state=42).copy()
    get_summary(hh_sample, dataset_type="HH")

    u5_sample = u5_dt.sample(frac=0.2, random_state=42).copy()
    get_summary(u5_sample, dataset_type="U5")

    learners = select_learners(names=DEBUG_LEARNER_NAMES)

    print("\n--- HH: Any Treatment -> E.coli ---")
    hh_any = run_analysis(
        dt=hh_sample,
        outcomes=HH_OUTCOMES,
        treatments=[ANY_TREATMENT],
        learners=learners,
        dataset_type="HH",
        prefix="debug_hh_any",
    )

    print("\n--- U5: Any Treatment -> Diarrhea ---")
    u5_any = run_analysis(
        dt=u5_sample,
        outcomes=U5_OUTCOMES,
        treatments=[ANY_TREATMENT],
        learners=learners,
        dataset_type="U5",
        prefix="debug_u5_any",
    )

    print("\n--- HH: Specific Treatments -> E.coli ---")
    hh_spec = run_analysis(
        dt=hh_sample,
        outcomes=HH_OUTCOMES,
        treatments=SPECIFIC_TREATMENTS[:2],
        learners=learners,
        dataset_type="HH",
        prefix="debug_hh_spec",
        restrict_single_method=True,
    )

    all_results = hh_any + u5_any + hh_spec
    print("\n" + "=" * 70)
    print("DEBUG RUN COMPLETE")
    print("=" * 70)
    print_significant_effects(all_results)
    export_results_csv(all_results, "results_debug.csv")


if __name__ == "__main__":
    main()
