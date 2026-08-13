"""
Shared runner boilerplate for all DDML analysis scripts.

Provides:
- Environment setup (warnings, sys.path, logging).
- Data loading helpers.
- Learner creation with CLI / program overrides.
- Result export + summary logging.

Usage in 10_run_irm.py, 20_run_apos.py, etc.:
from _runners import setup_environment, load_data, select_learners, save_results

    setup_environment()
    hh_dt, u5_dt = load_data()
    learners = select_learners(names=["ols", "stacked"])
    results = run_analysis(...)
    save_results(results, tag="main")
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any

# In-process filters (main process).
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning)

# Parallel learner fits run in loky subprocesses that do NOT import these
# filters, so sklearn FutureWarnings (e.g. LogisticRegressionCV attribute
# simplification in 1.10) leak from the workers.  PYTHONWARNINGS is read at
# interpreter startup, so children spawned by this process inherit it.
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import (
    HH_DATA_FILE, U5_DATA_FILE,
    LEARNER_NAMES, N_FOLDS, N_REP,
    OUTPUT_DIR, logger, setup_logging,
)
from _data import prepare_hh_data, prepare_u5_data, get_summary
from _learners import create_learners, create_learners_for_binary
from _models import export_results, export_results_csv, print_significant_effects


def setup_environment(log_to_file: bool = True) -> None:
    """Set up warnings, sys.path, and logging (idempotent)."""
    if log_to_file:
        setup_logging()


def load_data(
    hh_file: Optional[Path] = None,
    u5_file: Optional[Path] = None,
    summaries: bool = True,
) -> tuple:
    """Load and prepare both datasets.

    Returns
    -------
    hh_dt, u5_dt : (pd.DataFrame, pd.DataFrame)
    """
    hh_file = hh_file or HH_DATA_FILE
    u5_file = u5_file or U5_DATA_FILE

    logger.info("--- Loading HH dataset (E.coli outcomes) ---")
    hh_dt = prepare_hh_data(hh_file)
    if summaries:
        get_summary(hh_dt, dataset_type="HH")

    logger.info("--- Loading U5 dataset (diarrhea outcome) ---")
    u5_dt = prepare_u5_data(u5_file)
    if summaries:
        get_summary(u5_dt, dataset_type="U5")

    # Rob_1 overlap-robustness mirror: when MICS_ROB1=1, restrict every analysis
    # to households outside weak-support country cells (identical code path, the
    # only change is the sample).  Outputs go to the MICS_OUTPUT_DIR/MICS_FIGURE_DIR
    # set for the run, so headline results are never overwritten.
    import os as _os
    if _os.environ.get("MICS_ROB1") == "1":
        from _data import construct_rob1
        hh_dt = construct_rob1(hh_dt)
        u5_dt = construct_rob1(u5_dt)
        n_hh, n_u5 = len(hh_dt), len(u5_dt)
        hh_dt = hh_dt[hh_dt["Rob_1"] == 0].copy()
        u5_dt = u5_dt[u5_dt["Rob_1"] == 0].copy()
        logger.info(f"[MICS_ROB1] restricted sample: HH {len(hh_dt):,}/{n_hh:,}, "
                    f"U5 {len(u5_dt):,}/{n_u5:,}")

    return hh_dt, u5_dt


def select_learners(
    names: Optional[List[str]] = None,
    binary_outcome: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Create learners, optionally filtered to a subset of names.

    Parameters
    ----------
    names : list of learner names (e.g., ["ols", "stacked"]).
            If None, uses all LEARNER_NAMES.
    binary_outcome : If True, use create_learners_for_binary(),
                     which wraps ml_g in ProbaRegressor for bounded probs.

    Returns
    -------
    dict : {learner_name: {"g": ..., "m": ...}}
    """
    selected = names if names is not None else LEARNER_NAMES
    factory = create_learners_for_binary if binary_outcome else create_learners
    all_learners = factory()
    learners = {k: all_learners[k] for k in selected if k in all_learners}
    logger.info(f"  Learners ({'binary' if binary_outcome else 'regressor'}): {list(learners.keys())}")
    return learners


def save_results(
    results: List[Dict[str, Any]],
    tag: str,
) -> None:
    """Export .pkl, .csv, and print significant effects.

    Parameters
    ----------
    results : list of result dicts
    tag     : base filename tag (e.g., "main", "subgroups")
    """
    export_results(results, f"results_{tag}.pkl")
    export_results_csv(results, f"results_{tag}.csv")

    logger.info("=" * 70)
    logger.info(f"SIGNIFICANT EFFECTS SUMMARY — {tag.upper()}")
    logger.info("=" * 70)
    print_significant_effects(results)

