"""
Robustness checks: coefficient stability, leave-one-out, Mundlak, water storage + handwashing.

Runs all robustness specifications for both HH and U5 datasets.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    HH_DATA_FILE, U5_DATA_FILE, HH_OUTCOMES, U5_OUTCOMES,
    ANY_TREATMENT, OUTPUT_DIR,
    logger,
)
from data import prepare_hh_data, prepare_u5_data, get_summary
from learners import create_learners
from robustness import (
    run_coefficient_stability, run_leave_one_out,
    run_water_storage_handwashing, run_all_robustness,
)
from models import export_results, export_results_csv, print_significant_effects
from tables import create_stability_table, create_loo_table


def main():
    logger.info("=" * 70)
    logger.info("MICS DDML: ROBUSTNESS CHECKS")
    logger.info("=" * 70)

    # Load data
    hh_dt = prepare_hh_data(HH_DATA_FILE)
    u5_dt = prepare_u5_data(U5_DATA_FILE)

    all_robustness = {}

    # =========================================================================
    # 1. HH Dataset robustness
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("HH Dataset — Robustness Checks")
    logger.info("=" * 70)

    hh_results = run_all_robustness(
        dt=hh_dt, dataset_type="HH",
        outcomes=HH_OUTCOMES, treatments=[ANY_TREATMENT],
    )
    for key, val in hh_results.items():
        all_robustness[f"HH_{key}"] = val

    # =========================================================================
    # 2. U5 Dataset robustness
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("U5 Dataset — Robustness Checks")
    logger.info("=" * 70)

    u5_results = run_all_robustness(
        dt=u5_dt, dataset_type="U5",
        outcomes=U5_OUTCOMES, treatments=[ANY_TREATMENT],
    )
    for key, val in u5_results.items():
        all_robustness[f"U5_{key}"] = val

    # =========================================================================
    # Export
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ROBUSTNESS CHECKS COMPLETE")
    logger.info("=" * 70)

    # Flatten all results
    all_results = []
    for key, results in all_robustness.items():
        for r in results:
            r["robustness_type"] = key
        all_results.extend(results)

    export_results(all_results, "results_robustness.pkl")
    export_results_csv(all_results, "results_robustness.csv")

    # Generate tables
    try:
        stability_results = all_robustness.get("HH_stability", []) + all_robustness.get("U5_stability", [])
        if stability_results:
            create_stability_table(stability_results, filename="table_stability.tex")

        loo_results = all_robustness.get("HH_loo", []) + all_robustness.get("U5_loo", [])
        if loo_results:
            create_loo_table(loo_results, filename="table_loo.tex")
    except Exception as e:
        logger.info(f"Warning: Table generation failed: {e}")

    print_significant_effects(all_results)


if __name__ == "__main__":
    main()