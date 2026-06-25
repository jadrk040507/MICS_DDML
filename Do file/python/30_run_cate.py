"""
CATE — Conditional Average Treatment Effects (APOS, projection-based).

Step 3 of the heterogeneity workflow (APOS -> CATE -> GATE).

For each (dataset, outcome) a SINGLE DoubleMLAPOS is fit on the FULL sample
(multi-class propensity over WQ15_g levels, stacked learner, n_rep=1).  Then, for
each specific method d (boil / chlorine / filter / other) vs no treatment:

  * CATE(x) — the ATE(d vs 0) contrast signal projected onto a smooth B-spline
    basis of each moderator in ``CATE_MODERATORS`` -> continuous effect curve
    with pointwise + JOINT (uniform) bands.  Written to ``results_cate.csv`` and
    plotted by ``figures.plot_cate_curves`` (ROOT ``Figures/``).
  * CAPO(x) — the native potential-outcome LEVEL E[Y(d) | x] for every level
    (incl. no treatment), written to ``results_capo.csv``.  These are levels,
    NOT effects (see ``_heterogeneity_apos`` docstring).

APOS replaces the old IRM single-method-subsample path, which suffered positivity
failure (country FE predict adoption, cf_d ~ 1) and disagreed with the
full-sample contrasts.  Only genuinely continuous / well-supported moderators
(``wscore`` df=5, ``num_children`` df=4) live here; discrete-cell moderators are
GATE groups (``40_run_gate.py``).
"""

import argparse

import pandas as pd

from _config import (
    BASE_CONFOUNDERS, U5_ADDITIONAL_CONFOUNDERS,
    CATE_MODERATORS, FIGURE_DIR, OUTPUT_DIR, logger,
)
from _heterogeneity_apos import (
    fit_base_apos, estimate_cate_apos_spline, estimate_capo_spline,
)
from _figures import plot_cate_curves
from _runners import setup_environment, load_data

REFERENCE = 0  # no-treatment level

HH_HET_OUTCOMES = [
    {"var": "SomeRiskHome", "label": "Some Risk Home"},
    {"var": "VeryHighRiskHome", "label": "Very High Risk Home"},
]
U5_HET_OUTCOMES = [{"var": "diarrhea", "label": "Diarrhea"}]


def _run_dataset(dt, outcomes, dataset_type, confounders, skip_checkpoint=False):
    cate_rows, capo_rows = [], []
    for out in outcomes:
        out_var = out["var"]
        if out_var not in dt.columns:
            logger.warning(f"CATE: outcome '{out_var}' missing in {dataset_type}, skipping")
            continue

        logger.info(f"\n=== CATE | {dataset_type} | {out['label']} ===")
        fitted = fit_base_apos(
            dt=dt, outcome_var=out_var, dataset_type=dataset_type,
            learner_name="stacked", confounder_groups=confounders, reference=REFERENCE,
            skip_checkpoint=skip_checkpoint,
        )
        if fitted is None:
            logger.warning(f"  base APOS failed for {out_var}")
            continue
        apos, dt_clean, levels = fitted
        non_ref = [lv for lv in levels if lv != REFERENCE]

        for mod in CATE_MODERATORS:
            if dataset_type not in mod["datasets"]:
                continue
            mvar = mod["var"]
            if mvar not in dt_clean.columns:
                logger.warning(f"  moderator '{mvar}' missing, skipping")
                continue
            x = dt_clean[mvar].reset_index(drop=True)

            # Native CAPO level curves (all levels incl. reference) — once per moderator.
            try:
                capo = estimate_capo_spline(apos, levels, x, df_spline=mod.get("df", 5))
                capo.insert(0, "outcome", out_var)
                capo.insert(1, "dataset_type", dataset_type)
                capo.insert(2, "moderator", mvar)
                capo.insert(3, "moderator_label", mod["label"])
                capo_rows.append(capo)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"  CAPO {mvar} failed: {exc}")

            # Contrast CATE(x) per method vs none.
            for lv in non_ref:
                from _apos import APOS_LEVEL_LABELS
                trt_label = APOS_LEVEL_LABELS.get(lv, str(lv))
                logger.info(f"  CATE {trt_label} vs none over {mod['label']} ({mvar})")
                try:
                    df = estimate_cate_apos_spline(
                        apos, levels, lv, x, reference=REFERENCE, df_spline=mod.get("df", 5),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"    CATE {mvar} [{trt_label}] failed: {exc}")
                    continue
                df.insert(0, "outcome", out_var)
                df.insert(1, "dataset_type", dataset_type)
                df.insert(2, "treatment", trt_label)
                df.insert(3, "treatment_label", trt_label)
                df.insert(4, "treatment_level", lv)
                df.insert(5, "moderator", mvar)
                df.insert(6, "moderator_label", mod["label"])
                df.insert(7, "kind", "spline")
                logger.info(
                    f"    CATE range: [{df['cate'].min():+.4f}, {df['cate'].max():+.4f}]"
                )
                cate_rows.append(df)
    return cate_rows, capo_rows


def main():
    ap = argparse.ArgumentParser(description="APOS CATE (conditional effects + CAPO levels).")
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="ignore cached APOS fits and refit from scratch")
    args = ap.parse_args()

    setup_environment()
    logger.info("=" * 70)
    logger.info("MICS DDML: CATE (APOS projection-based conditional effects)")
    logger.info("=" * 70)

    hh_dt, u5_dt = load_data()
    hh_conf = dict(BASE_CONFOUNDERS)
    u5_conf = {**BASE_CONFOUNDERS, **U5_ADDITIONAL_CONFOUNDERS}

    cate_all, capo_all = [], []
    for dt, outs, dtype, conf in [
        (hh_dt, HH_HET_OUTCOMES, "HH", hh_conf),
        (u5_dt, U5_HET_OUTCOMES, "U5", u5_conf),
    ]:
        c, p = _run_dataset(dt, outs, dtype, conf, skip_checkpoint=args.no_checkpoint)
        cate_all += c
        capo_all += p

    if cate_all:
        cate_df = pd.concat(cate_all, ignore_index=True)
        cate_df.to_csv(OUTPUT_DIR / "results_cate.csv", index=False)
        cate_df.to_pickle(OUTPUT_DIR / "results_cate.pkl")
        logger.info("=" * 70)
        logger.info(f"CATE complete — {len(cate_df)} grid rows -> results_cate.csv")
        figs = plot_cate_curves(cate_df, out_dir=FIGURE_DIR)
        logger.info(f"CATE figures -> {len(figs)} PNG(s) in {FIGURE_DIR}")
    else:
        logger.warning("CATE produced no results.")

    if capo_all:
        capo_df = pd.concat(capo_all, ignore_index=True)
        capo_df.to_csv(OUTPUT_DIR / "results_capo.csv", index=False)
        capo_df.to_pickle(OUTPUT_DIR / "results_capo.pkl")
        logger.info(f"CAPO levels -> {len(capo_df)} grid rows -> results_capo.csv")


if __name__ == "__main__":
    main()
