"""Single command for the MICS ATE and ATT analyses."""

import argparse

import numpy as np

import analysis


FOLD_MODES = ("clustered", "unclustered", "all")
ESTIMANDS = ("ate", "att", "both")
STAGES = ("effects", "sensitivity", "gate", "all")


def run(fold_mode="clustered", estimand="both", stage="all"):
    """Run requested analyses in a stable, reproducible order."""
    if fold_mode not in FOLD_MODES:
        raise ValueError(f"Unknown fold mode: {fold_mode}")
    if estimand not in ESTIMANDS:
        raise ValueError(f"Unknown estimand: {estimand}")
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")

    folds = (
        ("clustered", "unclustered")
        if fold_mode == "all"
        else (fold_mode,)
    )
    estimands = ("ate", "att") if estimand == "both" else (estimand,)

    for selected_fold in folds:
        for selected_estimand in estimands:
            random_state = np.random.get_state()
            try:
                np.random.seed(analysis.SEED)
                analysis.run_analysis(
                    analysis.get_analysis_spec(selected_estimand),
                    fold_mode=selected_fold,
                    stage=stage,
                )
            finally:
                np.random.set_state(random_state)


def build_parser():
    """Create the documented command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run the MICS DoubleML ATE and ATT analyses.",
    )
    parser.add_argument(
        "fold_mode",
        nargs="?",
        default="clustered",
        choices=FOLD_MODES,
        help="cross-fitting scheme (default: clustered)",
    )
    parser.add_argument(
        "--estimand",
        default="both",
        choices=ESTIMANDS,
        help="causal estimand (default: both)",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=STAGES,
        help="analysis stage (default: all)",
    )
    return parser


def main(argv=None):
    """Parse command-line arguments and run the requested workflow."""
    args = build_parser().parse_args(argv)
    run(args.fold_mode, estimand=args.estimand, stage=args.stage)


if __name__ == "__main__":
    main()
