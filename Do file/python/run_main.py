"""
Main analysis: DDML IRM estimation for HH (E.coli) and U5 (diarrhea) datasets.

Usage:
    python run_main.py              # Full analysis (7 learners)
    python run_main.py --learners ols lasso rf  # Custom learner subset

Runs:
- Any treatment effect on all outcomes
- Specific treatment methods (boil, chlorine, filter, other) on all outcomes
- Sensitivity analysis (RV/RVa)
- Stacking weight extraction
"""

import argparse

from config import (
    HH_OUTCOMES, U5_OUTCOMES,
    ANY_TREATMENT, SPECIFIC_TREATMENTS,
    LEARNER_NAMES, N_FOLDS, N_REP,
    OUTPUT_DIR, logger,
)
from models import run_analysis
from runners import setup_environment, load_data, select_learners, save_results


def main():
    parser = argparse.ArgumentParser(
        description="Run the MICS DDML main analysis pipeline."
    )
    parser.add_argument(
        "--learners",
        nargs="+",
        choices=LEARNER_NAMES,
        default=None,
        help="Explicit list of learners to run (default: all 7).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help="Override N_FOLDS (default from config.py).",
    )
    parser.add_argument(
        "--n-rep",
        type=int,
        default=None,
        help="Override N_REP (default from config.py).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Number of parallel jobs across learners (default: 1, sequential).",
    )
    args = parser.parse_args()

    setup_environment()

    n_folds = args.n_folds if args.n_folds is not None else N_FOLDS
    n_rep = args.n_rep if args.n_rep is not None else N_REP
    n_jobs = args.parallel if args.parallel is not None else 1

    selected_learners = args.learners if args.learners else LEARNER_NAMES

    logger.info("=" * 70)
    logger.info("MICS DDML: MAIN ANALYSIS")
    logger.info(f"Learners: {selected_learners}")
    if args.n_folds is not None:
        logger.info(f"N_FOLDS (override) = {n_folds}")
    if args.n_rep is not None:
        logger.info(f"N_REP (override) = {n_rep}")
    if n_jobs > 1:
        logger.info(f"Parallel jobs (learners) = {n_jobs}")
    logger.info("=" * 70)

    # =========================================================================
    # 1. Load and prepare data
    # =========================================================================
    hh_dt, u5_dt = load_data()

    # =========================================================================
    # 2. Create learners
    # =========================================================================
    learners = select_learners(names=selected_learners)

    # Confounder groups
    from config import BASE_CONFOUNDERS, U5_ADDITIONAL_CONFOUNDERS
    hh_confounders = BASE_CONFOUNDERS.copy()
    u5_confounders = {**BASE_CONFOUNDERS, **U5_ADDITIONAL_CONFOUNDERS}

    # =========================================================================
    # 3. Analysis 1: Any Treatment on HH (E.coli outcomes)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("ANALYSIS 1: HH Dataset — Any Treatment on E.coli Outcomes")
    logger.info("=" * 70)

    hh_any_results = run_analysis(
        dt=hh_dt,
        outcomes=HH_OUTCOMES,
        treatments=[ANY_TREATMENT],
        learners=learners,
        confounder_groups=hh_confounders,
        dataset_type="HH",
        prefix="hh_any",
        n_folds=n_folds,
        n_rep=n_rep,
        n_jobs=n_jobs,
    )

    # =========================================================================
    # 4. Analysis 2: Any Treatment on U5 (diarrhea outcome)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("ANALYSIS 2: U5 Dataset — Any Treatment on Diarrhea")
    logger.info("=" * 70)

    u5_any_results = run_analysis(
        dt=u5_dt,
        outcomes=U5_OUTCOMES,
        treatments=[ANY_TREATMENT],
        learners=learners,
        confounder_groups=u5_confounders,
        dataset_type="U5",
        prefix="u5_any",
        n_folds=n_folds,
        n_rep=n_rep,
        n_jobs=n_jobs,
    )

    # =========================================================================
    # 5. Analysis 3: Specific Treatments on HH (E.coli outcomes)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("ANALYSIS 3: HH Dataset — Specific Treatments on E.coli Outcomes")
    logger.info("=" * 70)

    hh_specific_results = run_analysis(
        dt=hh_dt,
        outcomes=HH_OUTCOMES,
        treatments=SPECIFIC_TREATMENTS,
        learners=learners,
        confounder_groups=hh_confounders,
        dataset_type="HH",
        prefix="hh_specific",
        restrict_single_method=True,
        n_folds=n_folds,
        n_rep=n_rep,
        n_jobs=n_jobs,
    )

    # =========================================================================
    # 6. Analysis 4: Specific Treatments on U5 (diarrhea)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("ANALYSIS 4: U5 Dataset — Specific Treatments on Diarrhea")
    logger.info("=" * 70)

    u5_specific_results = run_analysis(
        dt=u5_dt,
        outcomes=U5_OUTCOMES,
        treatments=SPECIFIC_TREATMENTS,
        learners=learners,
        confounder_groups=u5_confounders,
        dataset_type="U5",
        prefix="u5_specific",
        restrict_single_method=True,
        n_folds=n_folds,
        n_rep=n_rep,
        n_jobs=n_jobs,
    )

    # =========================================================================
    # 7. Combine and export results
    # =========================================================================
    all_results = (hh_any_results + u5_any_results +
                   hh_specific_results + u5_specific_results)

    save_results(all_results, tag="main")

    # Generate tables
    try:
        from tables import create_main_table, create_sensitivity_table
        create_main_table(hh_any_results + hh_specific_results,
                          filename="table_hh_main.tex", dataset_type="HH")
        create_main_table(u5_any_results + u5_specific_results,
                          filename="table_u5_main.tex", dataset_type="U5")
        # Sensitivity (RV / RVa) for the stacked IRM, any-treatment specs
        create_sensitivity_table(hh_any_results + u5_any_results,
                                 filename="table_sensitivity.tex")
    except Exception as e:
        logger.info(f"Warning: Table generation failed: {e}")

    logger.info("=" * 70)
    logger.info("MAIN ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
