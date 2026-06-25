"""
Propensity / nuisance overlap figures by learner --- APOS (multi-valued).

Companion to 61_run_overlap.py (which does the binary IRM).  The APOS estimator
fits, on the FULL sample, one Average-Potential-Outcome model per treatment level
(none/boil/chlorine/filter/other) with a multinomial propensity.  For each
(dataset, outcome) and each learner we read the cross-fitted nuisances back out of
``apos.modellist[level]``:

  * D model --- the per-level propensity m_d(X) = P(D=d | X) (key ``ml_m``).
  * Y model --- the realised outcome regression g(X)=E[Y|X,D] built from the two
    APO outcome predictions (``ml_g_d_lvl1`` where D=d, ``ml_g_d_lvl0`` otherwise).

One combined figure per (dataset, outcome): propensity (top row) over outcome
(bottom row), columns = treatment levels, one density line per learner.  Saved as
``overlap_apos_<ds>_<outcome>.png`` in the ROOT Figures/ directory.

The APOS fits are checkpointed (shared with the heterogeneity runners) so re-runs
are cheap.  As with the IRM diagnostic, the point is to show that the multinomial
propensity stays interior for the regularised learners and the stacked ensemble.
"""

import argparse

import numpy as np

from _config import (
    BASE_CONFOUNDERS, U5_ADDITIONAL_CONFOUNDERS, LEARNER_NAMES, logger,
)
from _runners import setup_environment, load_data, select_learners
from _apos import APOS_LEVEL_LABELS
from _heterogeneity_apos import fit_base_apos
from _figures import plot_overlap_grid

REFERENCE = 0

DATASET_OUTCOMES = [
    ("HH", "hh", [
        {"var": "SomeRiskHome", "label": "Some Risk Home"},
        {"var": "VeryHighRiskHome", "label": "Very High Risk Home"},
    ]),
    ("U5", "u5", [
        {"var": "diarrhea", "label": "Diarrhea"},
    ]),
]


def _sq(arr):
    """(n,1,1) cross-fit predictions -> (n,)."""
    a = np.asarray(arr, dtype=float)
    return a.reshape(a.shape[0], -1).mean(axis=1)


def _apo_nuisances(apo, d, lv):
    """Per-observation propensity P(D=lv|X) and realised outcome g(X) for level lv."""
    preds = apo.predictions
    m = _sq(preds["ml_m"])
    g0 = _sq(preds["ml_g_d_lvl0"])
    g1 = _sq(preds["ml_g_d_lvl1"])
    ghat = np.where(d == lv, g1, g0)
    return m, ghat


def _run_dataset(dt, outcomes, dataset_type, confounders, learners, skip_checkpoint=False):
    figs = []
    for out in outcomes:
        out_var = out["var"]
        if out_var not in dt.columns:
            logger.warning(f"APOS-OVERLAP: outcome '{out_var}' missing in {dataset_type}, skip")
            continue
        logger.info(f"\n=== APOS-OVERLAP | {dataset_type} | {out['label']} ===")

        # curves per level: {level: {learner: array}}
        curves_d, curves_y, levels_seen = {}, {}, None
        for learner_name in learners:
            fitted = fit_base_apos(
                dt=dt, outcome_var=out_var, dataset_type=dataset_type,
                learner_name=learner_name, confounder_groups=confounders,
                reference=REFERENCE, skip_checkpoint=skip_checkpoint,
            )
            if fitted is None:
                logger.warning(f"  APOS fit failed for {out_var}/{learner_name}")
                continue
            apos, dt_clean, levels = fitted
            levels_seen = levels
            d = dt_clean["treat_cat"].astype(int).to_numpy()
            for lv in levels:
                try:
                    apo = apos.modellist[levels.index(lv)]
                    m, ghat = _apo_nuisances(apo, d, lv)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"  level {lv}/{learner_name} extract failed: {exc}")
                    continue
                curves_d.setdefault(lv, {})[learner_name] = m
                curves_y.setdefault(lv, {})[learner_name] = ghat
            logger.info(f"  {learner_name:<8} done "
                        f"(levels {[APOS_LEVEL_LABELS.get(l, l) for l in levels]})")

        if not levels_seen:
            continue
        panel_d = [{"label": APOS_LEVEL_LABELS.get(lv, str(lv)), "curves": curves_d.get(lv, {})}
                   for lv in levels_seen]
        panel_y = [{"label": APOS_LEVEL_LABELS.get(lv, str(lv)), "curves": curves_y.get(lv, {})}
                   for lv in levels_seen]
        figs.append(plot_overlap_grid(
            panel_d, panel_y,
            filename=f"overlap_apos_{dataset_type}_{out_var}.png",
            trim=0.01,
        ))
    return figs


def main():
    ap = argparse.ArgumentParser(description="Per-learner APOS multinomial overlap figures.")
    ap.add_argument("--learners", nargs="+", default=list(LEARNER_NAMES),
                    help="learners to overlay (default all, incl. stacked)")
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="ignore cached APOS fits and refit from scratch")
    args = ap.parse_args()

    setup_environment()
    learners = list(select_learners(args.learners))
    logger.info("=" * 70)
    logger.info(f"MICS DDML: APOS OVERLAP FIGURES (learners={learners})")
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
            learners, skip_checkpoint=args.no_checkpoint,
        )
    all_figs = [f for f in all_figs if f]
    logger.info("=" * 70)
    logger.info(f"APOS-OVERLAP complete — {len(all_figs)} figure(s) in ROOT Figures/")


if __name__ == "__main__":
    main()
