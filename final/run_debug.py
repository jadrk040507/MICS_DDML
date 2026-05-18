"""
Quick debug run of the MICS DDML pipeline.
Uses a sample of data and only 2 learners for fast iteration.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import HH_DATA_FILE, U5_DATA_FILE, HH_OUTCOMES, U5_OUTCOMES, ANY_TREATMENT, SPECIFIC_TREATMENTS, OUTPUT_DIR, CHECKPOINT_DIR
from data import prepare_hh_data, prepare_u5_data, get_summary
from learners import create_learners
from models import run_analysis, export_results_csv, print_significant_effects
import numpy as np

# Use only OLS and Stacked for speed
DEBUG_LEARNER_NAMES = ["ols", "stacked"]

def main():
    print("=" * 70)
    print("MICS DDML: QUICK DEBUG RUN")
    print("=" * 70)

    # Load and sample data (20% random sample for speed)
    print("\n--- Loading HH dataset ---")
    hh_dt = prepare_hh_data(HH_DATA_FILE)
    hh_sample = hh_dt.sample(frac=0.2, random_state=42).copy()
    get_summary(hh_sample, dataset_type="HH")

    print("\n--- Loading U5 dataset ---")
    u5_dt = prepare_u5_data(U5_DATA_FILE)
    u5_sample = u5_dt.sample(frac=0.2, random_state=42).copy()
    get_summary(u5_sample, dataset_type="U5")

    # Create only debug learners
    all_learners = create_learners()
    learners = {k: all_learners[k] for k in DEBUG_LEARNER_NAMES}
    print(f"\n--- Using learners: {list(learners.keys())} ---")

    # HH: Any treatment on E.coli
    print("\n--- HH: Any Treatment -> E.coli ---")
    hh_any = run_analysis(
        dt=hh_sample,
        outcomes=HH_OUTCOMES,
        treatments=[ANY_TREATMENT],
        learners=learners,
        dataset_type="HH",
        prefix="debug_hh_any",
    )

    # U5: Any treatment on diarrhea
    print("\n--- U5: Any Treatment -> Diarrhea ---")
    u5_any = run_analysis(
        dt=u5_sample,
        outcomes=U5_OUTCOMES,
        treatments=[ANY_TREATMENT],
        learners=learners,
        include_source_ecoli=True,
        include_child_controls=True,
        dataset_type="U5",
        prefix="debug_u5_any",
    )

    # HH: Specific treatments (first 2 only)
    print("\n--- HH: Specific Treatments -> E.coli ---")
    hh_spec = run_analysis(
        dt=hh_sample,
        outcomes=HH_OUTCOMES,
        treatments=SPECIFIC_TREATMENTS[:2],
        learners=learners,
        dataset_type="HH",
        prefix="debug_hh_spec",
    )

    all_results = hh_any + u5_any + hh_spec
    print("\n" + "=" * 70)
    print("DEBUG RUN COMPLETE")
    print("=" * 70)
    print_significant_effects(all_results)

    export_results_csv(all_results, "results_debug.csv")

if __name__ == "__main__":
    main()
