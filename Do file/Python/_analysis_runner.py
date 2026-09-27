"""Run clustered publication analyses by default; opt in to ordinary folds."""

import argparse
import importlib

import numpy as np


def run(analysis="clustered"):
    """Run estimand and fold combinations in documented order."""
    options = {
        "clustered": (("_ate_impl", "clustered", "effects"),
                      ("_att_impl", "clustered", "effects"),
                      ("_ate_impl", "clustered", "sensitivity"),
                      ("_att_impl", "clustered", "sensitivity"),
                      ("_ate_impl", "clustered", "gate"),
                      ("_att_impl", "clustered", "gate")),
        "unclustered": (("_ate_impl", "unclustered", "effects"),
                        ("_att_impl", "unclustered", "effects"),
                        ("_ate_impl", "unclustered", "sensitivity"),
                        ("_att_impl", "unclustered", "sensitivity"),
                        ("_ate_impl", "unclustered", "gate"),
                        ("_att_impl", "unclustered", "gate")),
    }
    options["all"] = options["clustered"] + options["unclustered"]
    for mode, name, stage in (
        ("ate_c", "_ate_impl", "effects"), ("att_c", "_att_impl", "effects"),
        ("sensitivity_c", "_ate_impl", "sensitivity"),
        ("gate_c", "_ate_impl", "gate"),
        ("ate_u", "_ate_impl", "effects"), ("att_u", "_att_impl", "effects"),
        ("sensitivity_u", "_ate_impl", "sensitivity"),
        ("gate_u", "_ate_impl", "gate"),
    ):
        fold = "clustered" if mode.endswith("_c") else "unclustered"
        if stage in ("sensitivity", "gate"):
            options[mode] = (("_ate_impl", fold, stage), ("_att_impl", fold, stage))
        else:
            options[mode] = ((name, fold, stage),)
    for name, fold_mode, stage in options[analysis]:
        module = importlib.import_module(name)
        label = "ATE" if name == "_ate_impl" else "ATT"
        print(f"\n{'=' * 72}\nRunning {label} {fold_mode} {stage}\n{'=' * 72}", flush=True)
        seed = getattr(module, "SEED", None)
        if seed is None:
            module.main(fold_mode=fold_mode, stage=stage)
        else:
            random_state = np.random.get_state()
            try:
                np.random.seed(seed)
                module.main(fold_mode=fold_mode, stage=stage)
            finally:
                np.random.set_state(random_state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "analysis",
        nargs="?",
        default="clustered",
        choices=("clustered", "unclustered", "all", "ate_c", "att_c",
                 "sensitivity_c", "gate_c", "ate_u", "att_u",
                 "sensitivity_u", "gate_u"),
        help=(
            "Default: clustered ATE then ATT. 'all' additionally runs "
            "the unclustered robustness specifications."
        ),
    )
    args = parser.parse_args()
    run(args.analysis)


if __name__ == "__main__":
    main()
