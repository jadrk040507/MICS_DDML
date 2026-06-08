"""
Falsification test: Treatment effects when source water has no contamination.

When RiskSource=0 (no E.coli at source), water treatment should have
no effect on home E.coli contamination. A significant effect here
would suggest confounding bias.
"""

from config import (
    HH_OUTCOMES, U5_OUTCOMES,
    ANY_TREATMENT, SUBGROUP_VAR, OUTPUT_DIR,
)
from robustness import run_falsification
from runners import setup_environment, load_data, select_learners, save_results


def main():
    setup_environment()
    import logging
    logging.getLogger("mics_ddml").info("=" * 70)
    logging.getLogger("mics_ddml").info("MICS DDML: FALSIFICATION TEST (RiskSource=0)")
    logging.getLogger("mics_ddml").info("=" * 70)

    hh_dt, u5_dt = load_data()
    learners = select_learners()

    all_results = []

    all_results += run_falsification(
        dt=hh_dt,
        outcomes=HH_OUTCOMES,
        treatments=[ANY_TREATMENT],
        learners=learners,
        dataset_type="HH",
    )

    all_results += run_falsification(
        dt=u5_dt,
        outcomes=U5_OUTCOMES,
        treatments=[ANY_TREATMENT],
        learners=learners,
        dataset_type="U5",
    )

    save_results(all_results, tag="falsification")

    try:
        from tables import create_falsification_table
        create_falsification_table(all_results, filename="table_falsification.tex")
    except Exception as e:
        import logging
        logging.getLogger("mics_ddml").warning(f"    Table generation failed: {e}")


if __name__ == "__main__":
    main()