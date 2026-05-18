"""
Falsification test: Treatment effects when source water has no contamination.

When RiskSource=0 (no E.coli at source), water treatment should have
no effect on home E.coli contamination. A significant effect here
would suggest confounding bias.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    HH_DATA_FILE, U5_DATA_FILE, HH_OUTCOMES, U5_OUTCOMES,
    ANY_TREATMENT, LEARNER_NAMES, SUBGROUP_VAR,
    OUTPUT_DIR,
)
from data import prepare_hh_data, prepare_u5_data, get_summary
from learners import create_learners
from robustness import run_falsification
from models import export_results, export_results_csv, print_significant_effects
from tables import create_falsification_table


def main():
    print("=" * 70)
    print("MICS DDML: FALSIFICATION TEST (RiskSource=0)")
    print("=" * 70)

    # Load data
    hh_dt = prepare_hh_data(HH_DATA_FILE)
    u5_dt = prepare_u5_data(U5_DATA_FILE)

    learners = create_learners()

    # =========================================================================
    # HH Dataset: E.coli outcomes at RiskSource=0
    # =========================================================================
    print("\n--- HH Dataset: E.coli outcomes at RiskSource=0 ---")
    hh_falsification = run_falsification(
        dt=hh_dt,
        outcomes=HH_OUTCOMES,
        treatments=[ANY_TREATMENT],
        learners=learners,
        dataset_type="HH",
        include_source_ecoli=False,
    )

    # =========================================================================
    # U5 Dataset: Diarrhea at RiskSource=0
    # =========================================================================
    print("\n--- U5 Dataset: Diarrhea at RiskSource=0 ---")
    u5_falsification = run_falsification(
        dt=u5_dt,
        outcomes=U5_OUTCOMES,
        treatments=[ANY_TREATMENT],
        learners=learners,
        dataset_type="U5",
        include_source_ecoli=True,
        include_child_controls=True,
    )

    # =========================================================================
    # Export
    # =========================================================================
    all_results = hh_falsification + u5_falsification

    print("\n" + "=" * 70)
    print("FALSIFICATION TEST RESULTS")
    print("=" * 70)
    print("Expected result: near-zero ATE when source water is uncontaminated.")
    print_significant_effects(all_results)

    export_results(all_results, "results_falsification.pkl")
    export_results_csv(all_results, "results_falsification.csv")

    # Generate table
    try:
        create_falsification_table(all_results, filename="table_falsification.tex")
    except Exception as e:
        print(f"Warning: Table generation failed: {e}")


if __name__ == "__main__":
    main()