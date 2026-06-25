"""
GATE — Group Average Treatment Effects (APOS, projection-based).

Step 4 of the heterogeneity workflow (APOS -> CATE -> GATE).

For each (dataset, outcome) a SINGLE DoubleMLAPOS is fit on the FULL sample
(multi-class propensity over WQ15_g levels, stacked learner, n_rep=1).  Then, for
each specific method d (boil / chlorine / filter / other) vs no treatment, and
each grouping variable in ``GATE_GROUPS``:

  * GATE — the ATE(d vs 0) contrast signal projected onto group indicators
    (``DoubleMLBLP``, is_gate=True) -> per-group effect with pointwise + JOINT
    (uniform, Semenova-Chernozhukov) bands.  Written to ``results_gate.csv``.
  * GAPO — the native potential-outcome LEVEL E[Y(d) | group] for every level
    (incl. no treatment), written to ``results_gapo.csv``.  Levels, NOT effects.

The full confounder vector (identical to the APOS headline, incl. RiskSource via
the source-E.coli deciles) enters the propensity / outcome models, so splitting
GATE by RiskSource is the econometrically correct conditional contrast
(RiskSource is pre-treatment).  APOS replaces the old IRM single-method-subsample
path (positivity failure, cf_d ~ 1).
"""

import argparse

import pandas as pd

from _config import (
    BASE_CONFOUNDERS, U5_ADDITIONAL_CONFOUNDERS,
    GATE_GROUPS, HELEVEL_CODES, TOILET_CODES, WS1G_CODES,
    OUTPUT_DIR, logger,
)
from _apos import APOS_LEVEL_LABELS
from _heterogeneity_apos import fit_base_apos, estimate_gate_apos, estimate_gapo_levels
from _runners import setup_environment, load_data

REFERENCE = 0  # no-treatment level

HH_HET_OUTCOMES = [
    {"var": "SomeRiskHome", "label": "Some Risk Home"},
    {"var": "VeryHighRiskHome", "label": "Very High Risk Home"},
]
U5_HET_OUTCOMES = [{"var": "diarrhea", "label": "Diarrhea"}]

GROUP_LABEL_MAPS = {
    "RiskSource": {0: "No risk", 1: "Moderate", 2: "Very high"},
    "helevel": HELEVEL_CODES,
    "windex5": {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5"},
    "country_cat": None,
    "Toilet": TOILET_CODES,
    "WS1_g": WS1G_CODES,
    "num_children_bin": {0: "0", 1: "1", 2: "2", 3: "3", 4: "4+"},
    "child_age": None,
}


def _run_dataset(dt, outcomes, dataset_type, confounders, skip_checkpoint=False):
    gate_rows, gapo_rows = [], []
    for out in outcomes:
        out_var = out["var"]
        if out_var not in dt.columns:
            logger.warning(f"GATE: outcome '{out_var}' missing in {dataset_type}, skipping")
            continue

        logger.info(f"\n=== GATE | {dataset_type} | {out['label']} ===")
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

        for grp in GATE_GROUPS:
            if dataset_type not in grp["datasets"]:
                continue
            gvar = grp["var"]
            if gvar not in dt_clean.columns:
                logger.warning(f"  group '{gvar}' missing, skipping")
                continue
            gseries = dt_clean[gvar].reset_index(drop=True)
            glabels = GROUP_LABEL_MAPS.get(gvar)

            # Native GAPO levels (all levels incl. reference) — once per group.
            try:
                gapo = estimate_gapo_levels(apos, levels, gseries, group_labels=glabels)
                gapo.insert(0, "outcome", out_var)
                gapo.insert(1, "dataset_type", dataset_type)
                gapo.insert(2, "group_var", gvar)
                gapo.insert(3, "group_var_label", grp["label"])
                gapo_rows.append(gapo)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"  GAPO {gvar} failed: {exc}")

            # Contrast GATE per method vs none.
            for lv in non_ref:
                trt_label = APOS_LEVEL_LABELS.get(lv, str(lv))
                logger.info(f"  GATE {trt_label} vs none over {grp['label']} ({gvar})")
                try:
                    gate_df = estimate_gate_apos(
                        apos, levels, lv, gseries, group_labels=glabels, reference=REFERENCE,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"    GATE {gvar} [{trt_label}] failed: {exc}")
                    continue
                gate_df.insert(0, "outcome", out_var)
                gate_df.insert(1, "dataset_type", dataset_type)
                gate_df.insert(2, "treatment", trt_label)
                gate_df.insert(3, "treatment_label", trt_label)
                gate_df.insert(4, "treatment_level", lv)
                gate_df.insert(5, "group_var", gvar)
                gate_df.insert(6, "group_var_label", grp["label"])
                for _, r in gate_df.iterrows():
                    logger.info(
                        f"    {str(r['group_label']):<12} GATE={r['coef']:+.4f} "
                        f"[{r['ci_lower']:+.3f},{r['ci_upper']:+.3f}] "
                        f"joint[{r['ci_lower_joint']:+.3f},{r['ci_upper_joint']:+.3f}] (n={r['n']})"
                    )
                gate_rows.append(gate_df)
    return gate_rows, gapo_rows


def main():
    ap = argparse.ArgumentParser(description="APOS GATE (group effects + GAPO levels).")
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="ignore cached APOS fits and refit from scratch")
    args = ap.parse_args()

    setup_environment()
    logger.info("=" * 70)
    logger.info("MICS DDML: GATE (APOS projection-based group effects)")
    logger.info("=" * 70)

    hh_dt, u5_dt = load_data()
    hh_conf = dict(BASE_CONFOUNDERS)
    u5_conf = {**BASE_CONFOUNDERS, **U5_ADDITIONAL_CONFOUNDERS}

    gate_all, gapo_all = [], []
    for dt, outs, dtype, conf in [
        (hh_dt, HH_HET_OUTCOMES, "HH", hh_conf),
        (u5_dt, U5_HET_OUTCOMES, "U5", u5_conf),
    ]:
        g, p = _run_dataset(dt, outs, dtype, conf, skip_checkpoint=args.no_checkpoint)
        gate_all += g
        gapo_all += p

    if gate_all:
        gate_df = pd.concat(gate_all, ignore_index=True)
        gate_df.to_csv(OUTPUT_DIR / "results_gate.csv", index=False)
        gate_df.to_pickle(OUTPUT_DIR / "results_gate.pkl")
        logger.info("=" * 70)
        logger.info(f"GATE complete — {len(gate_df)} group effects -> results_gate.csv")
    else:
        logger.warning("GATE produced no results.")

    if gapo_all:
        gapo_df = pd.concat(gapo_all, ignore_index=True)
        gapo_df.to_csv(OUTPUT_DIR / "results_gapo.csv", index=False)
        gapo_df.to_pickle(OUTPUT_DIR / "results_gapo.pkl")
        logger.info(f"GAPO levels -> {len(gapo_df)} group rows -> results_gapo.csv")


if __name__ == "__main__":
    main()
