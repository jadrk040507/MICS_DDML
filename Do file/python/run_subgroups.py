"""
Subgroup analysis by RiskSource level.

For both HH (E.coli) and U5 (diarrhea) datasets,
estimates DDML separately for RiskSource=0, 1, 2.
"""

from config import (
    HH_OUTCOMES, U5_OUTCOMES,
    ANY_TREATMENT, SUBGROUP_VAR, SUBGROUP_LABELS,
    logger,
)
from models import run_analysis
from runners import setup_environment, load_data, select_learners, save_results


def main():
    setup_environment()
    logger.info("=" * 70)
    logger.info("MICS DDML: SUBGROUP ANALYSIS BY RISK SOURCE")
    logger.info("=" * 70)

    hh_dt, u5_dt = load_data()
    learners = select_learners()

    all_results = []

    # HH Dataset: E.coli outcomes by RiskSource
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

    # U5 Dataset: Diarrhea by RiskSource
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
            subgroup_var=SUBGROUP_VAR,
            subgroup_val=rs_val,
            dataset_type="U5",
            prefix=f"u5_rs{rs_val}",
        )
        all_results.extend(results)

    save_results(all_results, tag="subgroups")

    try:
        from tables import create_subgroup_table
        create_subgroup_table(all_results, filename="table_subgroups.tex")
    except Exception as e:
        logger.info(f"Warning: Table generation failed: {e}")


if __name__ == "__main__":
    main()