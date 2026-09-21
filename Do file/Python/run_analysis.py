"""Single command-line entry point for the ATE and ATT analyses."""

import argparse
import importlib


def run(analysis):
    """Run one or both estimands using their existing audited pipelines."""

    names = ("ate", "att") if analysis == "both" else (analysis,)
    for name in names:
        module = importlib.import_module(name)
        print(f"\n{'=' * 72}\nRunning {name.upper()}\n{'=' * 72}", flush=True)
        module.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "analysis",
        nargs="?",
        default="both",
        choices=("ate", "att", "both"),
        help=(
            "Causal estimand to run; the default 'both' runs ATE followed "
            "by ATT."
        ),
    )
    args = parser.parse_args()
    run(args.analysis)


if __name__ == "__main__":
    main()
