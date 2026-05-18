"""
Main analysis: DDML IRM estimation for HH (E.coli) and U5 (diarrhea) datasets.

Usage:
    python run_main.py              # Full analysis (7 learners, slow)
    python run_main.py --fast       # Fast analysis (4 learners, ~2x speedup)
    python run_main.py --learners ols lasso rf  # Custom learner subset

Runs:
- Any treatment effect on all outcomes
- Specific treatment methods (boil, chlorine, filter, other) on all outcomes
- Sensitivity analysis (RV/RVa)
- Stacking weight extraction
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    HH_DATA_FILE, U5_DATA_FILE, HH_OUTCOMES, U5_OUTCOMES,
    ANY_TREATMENT, SPECIFIC_TREATMENTS, ALL_TREATMENTS,
    LEARNER_NAMES, FAST_LEARNER_NAMES, SUBGROUP_VAR, SUBGROUP_LABELS,
    OUTPUT_DIR, CHECKPOINT_DIR,
    logger, setup_logging,
)
from data import prepare_hh_data, prepare_u5_data, get_summary, validate_data
from learners import create_learners
from models import run_analysis, export_results, export_results_csv, print_significant_effects
from tables import create_main_table
from figures import plot_overlap_from_results

import pickle


def main():
    parser = argparse.ArgumentParser(
        description="Run the MICS DDML main analysis pipeline."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use only fast learners (OLS, Lasso, RF, Stacked) for quicker iteration.",
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
    args = parser.parse_args()

    setup_logging()

    # Determine learner subset
    if args.learners:
        selected_learners = args.learners
    elif args.fast:
        selected_learners = FAST_LEARNER_NAMES
    else:
        selected_learners = LEARNER_NAMES

    logger.info("=" * 70)
    logger.info("MICS DDML: MAIN ANALYSIS")
    logger.info(f"Learners: {selected_learners}")
    if args.fast:
        logger.info("FAST MODE: Using reduced learner set for speed")
    logger.info("=" * 70)

    # =========================================================================
    # 1. Load and prepare data
    # =========================================================================
    logger.info("--- Loading HH dataset (E.coli outcomes) ---")
    hh_dt = prepare_hh_data(HH_DATA_FILE)
    hh_summary = get_summary(hh_dt, dataset_type="HH")

    logger.info("--- Loading U5 dataset (diarrhea outcome) ---")
    u5_dt = prepare_u5_data(U5_DATA_FILE)
    u5_summary = get_summary(u5_dt, dataset_type="U5")

    # =========================================================================
    # 2. Create learners
    # =========================================================================
    logger.info("--- Creating ML learners ---")
    all_learners = create_learners()
    learners = {k: all_learners[k] for k in selected_learners if k in all_learners}
    logger.info(f"  Learners: {list(learners.keys())}")

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
        dataset_type="HH",
        prefix="hh_any",
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
        include_source_ecoli=True,
        include_child_controls=True,
        dataset_type="U5",
        prefix="u5_any",
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
        dataset_type="HH",
        prefix="hh_specific",
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
        include_source_ecoli=True,
        include_child_controls=True,
        dataset_type="U5",
        prefix="u5_specific",
    )

    # =========================================================================
    # 7. Combine and export results
    # =========================================================================
    all_results = (hh_any_results + u5_any_results +
                   hh_specific_results + u5_specific_results)

    logger.info("=" * 70)
    logger.info("SIGNIFICANT EFFECTS SUMMARY")
    logger.info("=" * 70)
    print_significant_effects(all_results)

    # Save results
    export_results(all_results, "results_main.pkl")
    export_results_csv(all_results, "results_main.csv")

    # Generate main tables
    logger.info("--- Generating tables ---")
    try:
        create_main_table(hh_any_results + hh_specific_results,
                          filename="table_hh_main.tex", dataset_type="HH")
        create_main_table(u5_any_results + u5_specific_results,
                          filename="table_u5_main.tex", dataset_type="U5")
    except Exception as e:
        logger.info(f"Warning: Table generation failed: {e}")

    logger.info("=" * 70)
    logger.info("MAIN ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
