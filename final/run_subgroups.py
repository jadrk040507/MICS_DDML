"""
Subgroup analysis by RiskSource level.

For both HH (E.coli) and U5 (diarrhea) datasets,
estimates DDML separately for RiskSource=0, 1, 2.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    HH_DATA_FILE, U5_DATA_FILE, HH_OUTCOMES, U5_OUTCOMES,
    ANY_TREATMENT, LEARNER_NAMES, SUBGROUP_VAR, SUBGROUP_LABELS,
    OUTPUT_DIR,
    logger,
)
from data import prepare_hh_data, prepare_u5_data, get_summary
from learners import create_learners
from models import run_analysis, export_results, export_results_csv, print_significant_effects
from tables import create_subgroup_table


def main():
    logger.info("=" * 70)
    logger.info("MICS DDML: SUBGROUP ANALYSIS BY RISK SOURCE")
    logger.info("=" * 70)

    # Load data
    logger.info("\n--- Loading data ---")
    hh_dt = prepare_hh_data(HH_DATA_FILE)
    u5_dt = prepare_u5_data(U5_DATA_FILE)

    learners = create_learners()

    all_results = []

    # =========================================================================
    # HH Dataset: E.coli outcomes by RiskSource
    # =========================================================================
    for rs_val, rs_label in SUBGROUP_LABELS.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"HH Dataset — RiskSource={rs_val} ({rs_label})")
        logger.info(f"{'='*70}")

        n = (hh_dt["RiskSource"] == rs_val).sum()
        logger.info(f"  N observations with RiskSource={rs_val}: {n:,}")

        if n < 100:
            logger.info(f"  Skipping: too few observations")
            continue

        results = run_analysis(
            dt=hh_dt,
            outcomes=HH_OUTCOMES,
            treatments=[ANY_TREATMENT],
            learners=learners,
            subgroup_var=SUBGROUP_VAR,
            subgroup_val=rs_val,
            dataset_type="HH",
            prefix=f"hh_rs{rs_val}",
        )
        all_results.extend(results)

    # =========================================================================
    # U5 Dataset: Diarrhea by RiskSource
    # =========================================================================
    for rs_val, rs_label in SUBGROUP_LABELS.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"U5 Dataset — RiskSource={rs_val} ({rs_label})")
        logger.info(f"{'='*70}")

        n = (u5_dt["RiskSource"] == rs_val).sum()
        logger.info(f"  N observations with RiskSource={rs_val}: {n:,}")

        if n < 100:
            logger.info(f"  Skipping: too few observations")
            continue

        results = run_analysis(
            dt=u5_dt,
            outcomes=U5_OUTCOMES,
            treatments=[ANY_TREATMENT],
            learners=learners,
            include_source_ecoli=True,
            include_child_controls=True,
            subgroup_var=SUBGROUP_VAR,
            subgroup_val=rs_val,
            dataset_type="U5",
            prefix=f"u5_rs{rs_val}",
        )
        all_results.extend(results)

    # =========================================================================
    # Export results
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("SUBGROUP ANALYSIS COMPLETE")
    logger.info("=" * 70)
    print_significant_effects(all_results)

    export_results(all_results, "results_subgroups.pkl")
    export_results_csv(all_results, "results_subgroups.csv")

    # Generate table
    try:
        create_subgroup_table(all_results, filename="table_subgroups.tex")
    except Exception as e:
        logger.info(f"Warning: Table generation failed: {e}")


if __name__ == "__main__":
    main()