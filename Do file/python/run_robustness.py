"""
Robustness checks: coefficient stability, leave-one-out, Mundlak, water storage + handwashing.

Runs all robustness specifications for both HH and U5 datasets.
"""

from config import (
    HH_OUTCOMES, U5_OUTCOMES, ANY_TREATMENT,
    BASE_CONFOUNDERS, U5_ADDITIONAL_CONFOUNDERS,
    OUTPUT_DIR, logger,
)
from robustness import run_all_robustness
from runners import setup_environment, load_data, save_results
from tables import create_stability_table, create_loo_table


def main():
    setup_environment()
    logger.info("=" * 70)
    logger.info("MICS DDML: ROBUSTNESS CHECKS")
    logger.info("=" * 70)

    hh_dt, u5_dt = load_data()

    all_robustness = {}

    # 1. HH Dataset robustness
    logger.info("\n" + "=" * 70)
    logger.info("HH Dataset — Robustness Checks")
    logger.info("=" * 70)

    hh_results = run_all_robustness(
        dt=hh_dt, dataset_type="HH",
        outcomes=HH_OUTCOMES, treatments=[ANY_TREATMENT],
        confounder_groups=BASE_CONFOUNDERS.copy(),
    )
    for key, val in hh_results.items():
        all_robustness[f"HH_{key}"] = val

    # 2. U5 Dataset robustness
    logger.info("\n" + "=" * 70)
    logger.info("U5 Dataset — Robustness Checks")
    logger.info("=" * 70)

    u5_results = run_all_robustness(
        dt=u5_dt, dataset_type="U5",
        outcomes=U5_OUTCOMES, treatments=[ANY_TREATMENT],
        confounder_groups={**BASE_CONFOUNDERS, **U5_ADDITIONAL_CONFOUNDERS},
    )
    for key, val in u5_results.items():
        all_robustness[f"U5_{key}"] = val

    # Flatten all results
    all_results = []
    for key, results in all_robustness.items():
        for r in results:
            r["robustness_type"] = key
        all_results.extend(results)

    save_results(all_results, tag="robustness")

    # Generate tables
    stability_results = all_robustness.get("HH_stability", []) + all_robustness.get("U5_stability", [])
    if stability_results:
        try:
            create_stability_table(stability_results, filename="table_stability.tex")
        except Exception as e:
            logger.info(f"Warning: Table generation failed: {e}")

    loo_results = all_robustness.get("HH_loo", []) + all_robustness.get("U5_loo", [])
    if loo_results:
        try:
            create_loo_table(loo_results, filename="table_loo.tex")
        except Exception as e:
            logger.info(f"Warning: Table generation failed: {e}")


if __name__ == "__main__":
    main()