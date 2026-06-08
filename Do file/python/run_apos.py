"""
Multi-treatment APOS analysis.

Estimates the Average Potential Outcome contrasts ATE(d vs no treatment) for each
specific water-treatment method (boil, chlorine, filter, other) jointly on the
full sample, for every outcome.  Complements the binary subsample IRM in
``run_main.py`` (specific treatments) with the efficient full-sample AIPW
estimator.

    cd "Do file/python"
    python run_apos.py
"""

import logging

from config import HH_OUTCOMES, U5_OUTCOMES, BASE_CONFOUNDERS, U5_ADDITIONAL_CONFOUNDERS, OUTPUT_DIR
from apos import run_apos
from runners import setup_environment, load_data, save_results


def main():
    setup_environment()
    log = logging.getLogger("mics_ddml")
    log.info("=" * 70)
    log.info("MICS DDML: APOS MULTI-TREATMENT ANALYSIS")
    log.info("=" * 70)

    hh_dt, u5_dt = load_data()

    hh_confounders = BASE_CONFOUNDERS.copy()
    u5_confounders = {**BASE_CONFOUNDERS, **U5_ADDITIONAL_CONFOUNDERS}

    all_results = []
    all_results += run_apos(hh_dt, HH_OUTCOMES, dataset_type="HH",
                            confounder_groups=hh_confounders)
    all_results += run_apos(u5_dt, U5_OUTCOMES, dataset_type="U5",
                            confounder_groups=u5_confounders)

    save_results(all_results, tag="apos")

    try:
        from tables import create_apos_table
        create_apos_table(all_results, filename="table_apos.tex")
    except Exception as e:
        logging.getLogger("mics_ddml").warning(f"    Table generation failed: {e}")

    log.info("=" * 70)
    log.info("APOS ANALYSIS COMPLETE")
    log.info(f"Results saved to: {OUTPUT_DIR}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
