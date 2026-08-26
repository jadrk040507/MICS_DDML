"""
Step 1 of the workflow: DDML IRM estimation for HH (E.coli) and U5 (diarrhea).

Usage:
    python 10_run_irm.py              # Full analysis (7 learners)
    python 10_run_irm.py --learners ols lasso rf  # Custom learner subset

Runs ONLY the any-treatment effect (water_treatment vs. none, full sample) on
every outcome, plus its sensitivity (RV/RVa) and stacking-weight extraction.
Specific-method effects are reported by APOS (20_run_apos.py); the any-treatment
IRM estimated here is loaded by that step and shown as the top row of the
combined E.coli / diarrhea results tables.
"""

import argparse
from _config import (
    ANY_TREATMENT,
    LEARNER_NAMES, N_FOLDS, N_REP,
    OUTPUT_DIR, logger,
)
from _models import run_analysis
from _runners import setup_environment, load_data, select_learners, save_results

# E.coli outcomes belong to the HH dataset; diarrhea to U5.  (SomeRiskHome /
# VeryHighRiskHome also exist as columns in the U5 file, so estimating them
# there would just duplicate the HH specs — restrict each dataset to its own.)
HH_OUTCOMES = [
    {"var": "SomeRiskHome", "label": "Some Risk Home"},
    {"var": "VeryHighRiskHome", "label": "Very High Risk Home"},
]
U5_OUTCOMES = [{"var": "diarrhea", "label": "Diarrhea"}]


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
        default=1,
        help=(
            "Parallel jobs ACROSS learners (default: 1). Keep at 1: the stacked "
            "learner already parallelizes internally over its base learners/folds "
            "(n_jobs=N_JOBS), so an outer pool would oversubscribe the CPU. Raise "
            "only when running many single-threaded learners and the inner "
            "parallelism is idle."
        ),
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
    learners = select_learners(names=selected_learners, binary_outcome=True)

    # Confounder groups
    from _config import BASE_CONFOUNDERS, U5_ADDITIONAL_CONFOUNDERS
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
    # 5. Tag contrast, combine, export
    # =========================================================================
    any_results = hh_any_results + u5_any_results
    for r in any_results:
        r["contrast"] = "vs_none"

    all_results = any_results
    save_results(all_results, tag="main")

    # Generate tables
    try:
        from _tables import create_sensitivity_table
        # Sensitivity (RV / RVa) for the stacked IRM, any-treatment specs.
        # The any-treatment IRM point estimates are tabulated by 20_run_apos.py,
        # which loads results_main.pkl and renders them atop the APOS panel.
        create_sensitivity_table(any_results, filename="table_sensitivity.tex")
    except Exception as e:
        logger.info(f"Warning: Table generation failed: {e}")

    logger.info("=" * 70)
    logger.info("MAIN ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
