"""
Propensity / nuisance overlap figures by learner (IRM).

Step 61 of the workflow — the DIRECT positivity/overlap exhibit that the cf_d
column in ``60_run_sensitivity.py`` only proxies.

For each (dataset, outcome) and each specific method (vs none, single-method
subsample) plus any-treatment (full sample), we fit a DoubleMLIRM with EACH
learner and read the cross-fitted nuisance predictions back out:

  * D model — the propensity m(X) = P(D=d | X).  Mass pushed to 0/1 == positivity
    failure.  cf_d ~ 1 (country FE dominate adoption) is only a SYMPTOM; this is
    the disease, shown per learner so the reader sees which learners (e.g.
    flexible/stacked) actually drive m(X) to the extremes while a regularised
    learner (lasso) keeps it interior.
  * Y model — the outcome regression g(X) = E[Y | X, D] (realised: g1 if treated,
    g0 if control).  Same per-learner overlay.

Two figures per (dataset, outcome): ``overlap_D_<ds>_<outcome>.png`` and
``overlap_Y_<ds>_<outcome>.png`` in the ROOT Figures/ directory, faceted by
treatment, one density line per learner (incl. stacked).

The IRM fits are checkpointed (shared with 60) so re-runs are cheap.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from _config import (
    BASE_CONFOUNDERS, U5_ADDITIONAL_CONFOUNDERS, LEARNER_NAMES, logger,
)
from _runners import setup_environment, load_data, select_learners
from _figures import plot_overlap_grid

# Reuse the named-IRM fit + subsample logic from the sensitivity script (numbered
# module -> import by path).
_spec = importlib.util.spec_from_file_location(
    "_sens60", Path(__file__).parent / "60_run_sensitivity.py")
_sens60 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sens60)
_fit_named_irm = _sens60._fit_named_irm
IRM_TREATMENTS = _sens60.IRM_TREATMENTS

DATASET_OUTCOMES = [
    ("HH", "hh", [
        {"var": "SomeRiskHome", "label": "Some Risk Home"},
        {"var": "VeryHighRiskHome", "label": "Very High Risk Home"},
    ]),
    ("U5", "u5", [
        {"var": "diarrhea", "label": "Diarrhea"},
    ]),
]


def _avg_rep(arr):
    """(n_obs, n_rep, 1) cross-fit predictions -> (n_obs,) averaged over reps."""
    a = np.squeeze(np.asarray(arr, dtype=float))
    return a.mean(axis=1) if a.ndim == 2 else a


def _nuisance_predictions(irm, df):
    """Return (m, ghat) per observation from a fitted IRM.

    m   = propensity P(D=d|X) = predictions['ml_m'].
    ghat = realised outcome regression: g1 where treated, g0 where control.
    """
    preds = irm.predictions
    m = _avg_rep(preds["ml_m"])
    g0 = _avg_rep(preds["ml_g0"])
    g1 = _avg_rep(preds["ml_g1"])
    d = df["_d"].to_numpy(dtype=int)
    ghat = np.where(d == 1, g1, g0)
    return m, ghat


def _run_dataset(dt, outcomes, dataset_type, confounders, learners, n_folds, n_rep,
                 skip_checkpoint=False):
    figs = []
    for out in outcomes:
        out_var = out["var"]
        if out_var not in dt.columns:
            logger.warning(f"OVERLAP: outcome '{out_var}' missing in {dataset_type}, skip")
            continue
        logger.info(f"\n=== OVERLAP | {dataset_type} | {out['label']} ===")

        panel_d, panel_y = [], []
        for treatment_var, _tkey, tlabel in IRM_TREATMENTS:
            if treatment_var not in dt.columns:
                continue
            curves_m, curves_g = {}, {}
            for learner_name in learners:
                fitted = _fit_named_irm(
                    dt, out_var, treatment_var, confounders, learner_name,
                    n_folds, n_rep, dataset_type=dataset_type,
                    skip_checkpoint=skip_checkpoint,
                )
                if fitted is None:
                    continue
                irm, df, _names, _gmap, n = fitted
                try:
                    m, ghat = _nuisance_predictions(irm, df)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"    {tlabel}/{learner_name} predictions failed: {exc}")
                    continue
                curves_m[learner_name] = m
                curves_g[learner_name] = ghat
                logger.info(f"    {tlabel:<12} {learner_name:<8} "
                            f"m∈[{m.min():.3f},{m.max():.3f}] "
                            f"trim%={100*np.mean((m<0.01)|(m>0.99)):.1f} (n={n})")
            if curves_m:
                panel_d.append({"label": tlabel, "curves": curves_m})
                panel_y.append({"label": tlabel, "curves": curves_g})

        if panel_d:
            figs.append(plot_overlap_grid(
                panel_d, panel_y,
                filename=f"overlap_{dataset_type}_{out_var}.png",
                trim=0.01,
            ))
    return figs


def main():
    ap = argparse.ArgumentParser(description="Per-learner IRM nuisance overlap figures (D and Y).")
    ap.add_argument("--learners", nargs="+", default=list(LEARNER_NAMES),
                    help="learners to overlay (default all, incl. stacked)")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-rep", type=int, default=1)
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="ignore cached IRM fits and refit from scratch")
    args = ap.parse_args()

    setup_environment()
    learners = select_learners(args.learners)
    logger.info("=" * 70)
    logger.info(f"MICS DDML: OVERLAP FIGURES (learners={list(learners)})")
    logger.info("=" * 70)

    hh_dt, u5_dt = load_data()
    data_by_key = {"hh": hh_dt, "u5": u5_dt}
    hh_conf = dict(BASE_CONFOUNDERS)
    u5_conf = {**BASE_CONFOUNDERS, **U5_ADDITIONAL_CONFOUNDERS}
    conf_by_ds = {"HH": hh_conf, "U5": u5_conf}

    all_figs = []
    for dataset_type, key, outcomes in DATASET_OUTCOMES:
        all_figs += _run_dataset(
            data_by_key[key], outcomes, dataset_type, conf_by_ds[dataset_type],
            list(learners), args.n_folds, args.n_rep, skip_checkpoint=args.no_checkpoint,
        )

    all_figs = [f for f in all_figs if f]
    logger.info("=" * 70)
    logger.info(f"OVERLAP complete — {len(all_figs)} figure(s) in ROOT Figures/")


if __name__ == "__main__":
    main()
