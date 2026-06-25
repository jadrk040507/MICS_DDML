"""
Confounder benchmarking / sensitivity analysis (Chernozhukov et al. 2022) — APOS.

Step 60 of the workflow.  APOS is the correct multi-treatment estimand (full
sample, multi-class propensity), so sensitivity now runs on ``DoubleMLAPOS``
rather than the old single-method-subsample IRM (which suffered positivity
failure, cf_d ~ 1, and is not the headline anymore).

For EACH (dataset, outcome) this script:

  1. Fits a NAMED ``DoubleMLAPOS`` (DataFrame backend, real column names so
     covariate groups are addressable for benchmarking_set) on the full sample,
     reference level = no treatment.
  2. Forms the contrast ATE(d vs 0) for each method d (boil/chlorine/filter/
     other) and reports its robustness values RV / RV_alpha (share of residual
     variation an unobserved confounder must explain in BOTH treatment and
     outcome to drive the contrast / its 95% CI to zero), from
     ``causal_contrast(...).sensitivity_analysis``.
  3. Benchmarks each observed confounder GROUP at the CONTRAST level: refits a
     SHORT APOS (dropping that group's columns) on the same sample + learner,
     forms the short d-vs-0 contrast, and applies doubleml's gain_statistics
     formula to the long-vs-short contrast sensitivity_elements -> cf_y (partial
     R^2 in the outcome), cf_d (partial R^2 in adoption), and delta_theta (the
     bias an unobserved confounder "as strong as that group" induces) -- all for
     the ATE(d vs 0), matching the RV / RV_alpha.

The benchmark answers: *is the contrast RV larger than the strongest confounder
we already control?*  Output: a long CSV (results_sensitivity.csv) + a
comparison LaTeX table (table_sensitivity_benchmark.tex) across all specs.

Note: cf_y/cf_d/delta_theta and RV/RV_alpha are now BOTH for the d-vs-0 contrast
(no level/contrast mismatch).  ``DoubleMLFramework`` -- what ``causal_contrast``
returns -- exposes sigma2/nu2 but no ``sensitivity_benchmark`` method, so the
gain formula is replicated in ``_contrast_gain``.  Cost: one short APOS refit per
group.  Default learner ``stacked`` (headline; cf_y/cf_d are design-stable, so
``--learner lasso`` is a faster approximation).
"""

import argparse

import numpy as np
import pandas as pd
import doubleml as dml
from sklearn.base import clone

from _config import (
    BASE_CONFOUNDERS, U5_ADDITIONAL_CONFOUNDERS,
    MIN_OBSERVATIONS, OUTPUT_DIR, CHECKPOINT_DIR, N_FOLDS, logger,
)
from _data import create_model_matrix
from _learners import create_learners_for_binary
from _runners import setup_environment, load_data
from _tables import create_sensitivity_benchmark_table
from _figures import plot_propensity_overlap_panel
from _heterogeneity_apos import load_or_fit
from _apos import _build_treat_cat, APOS_LEVEL_LABELS, APOS_MIN_LEVEL_N

# Sensitivity reference strength (matches the APOS / IRM sensitivity defaults).
SENS_CF = 0.03
SENS_RHO = 1.0
REFERENCE = 0

# APOS level -> the treatment KEY the benchmark table expects (Panel B rows).
LEVEL_TO_TREATKEY = {
    1: "treat_boil", 2: "treat_chlorine", 3: "treat_filter", 4: "treat_other",
}

# (dataset_type, dataframe-key, [outcomes])
DATASET_OUTCOMES = [
    ("HH", "hh", [
        {"var": "SomeRiskHome", "label": "Some Risk Home"},
        {"var": "VeryHighRiskHome", "label": "Very High Risk Home"},
    ]),
    ("U5", "u5", [
        {"var": "diarrhea", "label": "Diarrhea"},
    ]),
]

# Human-readable confounder-group labels for the table.
GROUP_LABELS = {
    "wealth": "Wealth quintile",
    "education": "Education",
    "country": "Country FE",
    "urban": "Urban",
    "water_source": "Water source",
    "hh_demog": "HH demographics",
    "sanitation": "Sanitation",
    "wq27": "Source E.coli (decile)",
    "child": "Child age/sex",
}


def _contrast_gain(long_c, short_c, var_y):
    """Replicate doubleml ``gain_statistics`` at the CONTRAST level.

    Computes cf_y / cf_d / rho / delta_theta for the d-vs-0 contrast from the
    long vs. short contrast sensitivity_elements (sigma2, nu2) and thetas — the
    same formula ``DoubleML.sensitivity_benchmark`` applies per level, but here on
    the contrast framework (which exposes sigma2/nu2 but has no benchmark method).

    Returns four arrays (cf_y, cf_d, rho, delta_theta), one entry per contrast
    coefficient (i.e. per non-reference level, in contrast order).
    """
    s2L = np.squeeze(long_c.sensitivity_elements["sigma2"], axis=0)   # (n_coef, n_rep)
    nuL = np.squeeze(long_c.sensitivity_elements["nu2"], axis=0)
    s2S = np.squeeze(short_c.sensitivity_elements["sigma2"], axis=0)
    nuS = np.squeeze(short_c.sensitivity_elements["nu2"], axis=0)

    R2yL = 1.0 - s2L / var_y
    R2yS = 1.0 - s2S / var_y
    R2riesz = nuS / nuL
    cf_y = np.clip((R2yL - R2yS) / (1.0 - R2yL), 0, 1)
    cf_d = np.clip((1.0 - R2riesz) / R2riesz, 0, 1)

    dth = np.asarray(short_c.all_thetas) - np.asarray(long_c.all_thetas)
    var_g = s2S - s2L
    var_r = nuL - nuS
    denom = np.sqrt(np.multiply(var_g, var_r, out=np.zeros_like(var_g),
                                where=(var_g > 0) & (var_r > 0)))
    rho_vals = np.clip(np.divide(np.abs(dth), denom, out=np.ones_like(dth),
                                 where=denom != 0), 0.0, 1.0)
    rho = rho_vals * np.sign(dth)

    return (np.median(cf_y, axis=1), np.median(cf_d, axis=1),
            np.median(rho, axis=1), np.median(dth, axis=1))


def _named_model_matrix(dt, outcome_var, confounder_groups):
    """Build X with REAL column names + a {group: [col names]} map.

    Mirrors ``create_model_matrix``'s expansion so the columns line up exactly,
    but also returns the per-group column lists needed for benchmarking_set.
    """
    X, dt = create_model_matrix(dt, outcome_var, confounder_groups=confounder_groups)

    def _expand(prefix, drop_first):
        cols = sorted([c for c in dt.columns if c.startswith(prefix)],
                      key=lambda c: (len(c), c))
        return cols[1:] if (drop_first and len(cols) > 1) else cols

    names, group_map, seen = [], {}, set()
    for group_name, group_cols in confounder_groups.items():
        gnames = []
        for col in group_cols:
            if col == "water_source":
                e = _expand("ws1g_", True)
            elif col == "country_cat":
                e = _expand("country_cat_", False)
            elif col == "wq27_decile":
                e = _expand("wq27_d_", False)
            elif col == "child_age":
                e = _expand("child_age_", False)
            else:
                e = [col]
            gnames += [c for c in e if c in dt.columns]
        gnames = [c for c in gnames if not (c in seen or seen.add(c))]
        group_map[group_name] = gnames
        names += gnames
    return dt, names, group_map


def _fit_named_apos(dt, outcome_var, confounder_groups, learner_name, n_folds, n_rep,
                    dataset_type="HH", skip_checkpoint=False):
    """Complete-case + sparse-level drop -> fit a named DoubleMLAPOS.

    The (expensive) long APOS fit is checkpointed per (dataset, outcome, learner);
    the cheap prep (masking, column names) is always recomputed so the returned
    ``df`` stays available for the short benchmark refits.

    Returns (apos, contrast, df, names, group_map, levels, n) or None, where
    ``df`` is the named DataFrame (``_y`` / ``_d`` + confounder columns) so a
    SHORT APOS can be refit per benchmark group without re-masking.
    """
    dt = dt.copy()
    dt["treat_cat"] = _build_treat_cat(dt)
    dt, names, group_map = _named_model_matrix(dt, outcome_var, confounder_groups)

    Xmat = dt[names].to_numpy(dtype=float)
    mask = (
        dt[outcome_var].notna()
        & dt["treat_cat"].notna()
        & ~np.isnan(Xmat).any(axis=1)
        & ~np.isinf(Xmat).any(axis=1)
    )
    n = int(mask.sum())
    if n < MIN_OBSERVATIONS:
        logger.error(f"    too few complete obs (N={n})")
        return None

    sub = dt.loc[mask].reset_index(drop=True)
    d = sub["treat_cat"].astype(int).to_numpy()
    counts = pd.Series(d).value_counts()
    levels = sorted(int(lv) for lv in counts.index
                    if counts[lv] >= APOS_MIN_LEVEL_N or lv == REFERENCE)
    keep = np.isin(d, levels)
    sub = sub.loc[keep].reset_index(drop=True)
    n = int(keep.sum())
    if REFERENCE not in levels or len(levels) < 2:
        logger.error(f"    insufficient treatment levels {levels}")
        return None

    df = sub[names].astype(float).copy()
    df["_y"] = sub[outcome_var].to_numpy(dtype=float)
    df["_d"] = sub["treat_cat"].astype(int).to_numpy()

    def _build_long():
        data = dml.DoubleMLData(df, y_col="_y", d_cols="_d", x_cols=names)
        lset = create_learners_for_binary()[learner_name]
        ap = dml.DoubleMLAPOS(
            data, ml_g=clone(lset["g"]), ml_m=clone(lset["m"]),
            treatment_levels=levels, n_folds=n_folds, n_rep=n_rep,
            trimming_rule="truncate", trimming_threshold=0.01,
        )
        ap.fit()
        return ap

    cp_long = CHECKPOINT_DIR / f"apos_sens_{dataset_type}_{outcome_var}_{learner_name}.pkl"
    apos = load_or_fit(cp_long, _build_long, skip_checkpoint=skip_checkpoint)
    contrast = apos.causal_contrast(reference_levels=REFERENCE)
    return apos, contrast, df, names, group_map, levels, n


def run_spec(dt, outcome, confounders, dataset_type, learner_name, n_folds, n_rep,
             skip_checkpoint=False):
    """One (dataset, outcome): fit APOS + per-level RV + per-group benchmark."""
    out_var = outcome["var"]
    logger.info(f"\n=== SENS(APOS) | {dataset_type} | {outcome['label']} ===")

    fitted = _fit_named_apos(dt, out_var, confounders, learner_name, n_folds, n_rep,
                             dataset_type=dataset_type, skip_checkpoint=skip_checkpoint)
    if fitted is None:
        logger.warning("    spec skipped (insufficient data)")
        return []
    apos, contrast, df, names, group_map, levels, n = fitted
    non_ref = [lv for lv in levels if lv != REFERENCE]
    var_y = float(np.var(df["_y"].to_numpy(dtype=float)))

    # Contrast ATE(d vs 0) coef/se + RV/RVa per non-reference level.
    csum = contrast.summary
    coefs = csum["coef"].to_numpy()
    ses = csum["std err"].to_numpy()
    rv = rva = None
    try:
        contrast.sensitivity_analysis(cf_y=SENS_CF, cf_d=SENS_CF, rho=SENS_RHO)
        sp = contrast.sensitivity_params
        if sp is not None:
            rv = np.ravel(sp["rv"]).astype(float)
            rva = np.ravel(sp["rva"]).astype(float)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"    contrast sensitivity_analysis failed: {exc}")

    # CONTRAST-level benchmark per confounder group: refit a SHORT APOS (dropping
    # the group's columns) on the same sample/learner, form its d-vs-0 contrast,
    # and apply doubleml's gain_statistics formula to the long-vs-short contrast
    # elements.  cf_y/cf_d/delta_theta are then for the ATE(d vs 0), not the
    # level-d potential outcome.  Arrays align to non_ref (contrast order).
    bench = {}  # group_name -> (cf_y[], cf_d[], rho[], dtheta[])
    for group_name, gcols in group_map.items():
        if not gcols:
            continue
        names_short = [c for c in names if c not in set(gcols)]
        if not names_short:
            continue

        def _build_short(names_short=names_short):
            data_s = dml.DoubleMLData(df, y_col="_y", d_cols="_d", x_cols=names_short)
            lset = create_learners_for_binary()[learner_name]
            ap = dml.DoubleMLAPOS(
                data_s, ml_g=clone(lset["g"]), ml_m=clone(lset["m"]),
                treatment_levels=levels, n_folds=n_folds, n_rep=n_rep,
                trimming_rule="truncate", trimming_threshold=0.01,
            )
            ap.fit()
            return ap

        try:
            cp_short = (CHECKPOINT_DIR /
                        f"apos_sens_short_{dataset_type}_{out_var}_{learner_name}_{group_name}.pkl")
            apos_s = load_or_fit(cp_short, _build_short, skip_checkpoint=skip_checkpoint)
            contrast_s = apos_s.causal_contrast(reference_levels=REFERENCE)
            bench[group_name] = _contrast_gain(contrast, contrast_s, var_y)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"    contrast benchmark {group_name} failed: {exc}")

    rows = []
    for i, lv in enumerate(non_ref):
        tkey = LEVEL_TO_TREATKEY.get(lv, f"level_{lv}")
        tlabel = APOS_LEVEL_LABELS.get(lv, str(lv))
        ate = float(coefs[i]); se = float(ses[i])
        rv_q = float(rv[i]) if rv is not None and i < len(rv) else None
        rv_qa = float(rva[i]) if rva is not None and i < len(rva) else None
        logger.info(f"    {tlabel:<10} ATE={ate:+.4f} (se {se:.4f})  "
                    f"RV={None if rv_q is None else round(rv_q,4)} "
                    f"RVa={None if rv_qa is None else round(rv_qa,4)}")
        for group_name, gcols in group_map.items():
            if not gcols or group_name not in bench:
                continue
            cf_y_a, cf_d_a, rho_a, dth_a = bench[group_name]
            cf_y = float(cf_y_a[i])
            cf_d = float(cf_d_a[i])
            rho = float(rho_a[i])
            dtheta = float(dth_a[i])
            rows.append({
                "method": "APOS",
                "dataset_type": dataset_type,
                "outcome": out_var,
                "outcome_label": outcome["label"],
                "treatment": tkey,
                "treatment_label": tlabel,
                "treatment_level": lv,
                "learner": learner_name,
                "n": n,
                "ate": ate,
                "se": se,
                "rv_q": rv_q,
                "rv_qa": rv_qa,
                "group": group_name,
                "group_label": GROUP_LABELS.get(group_name, group_name),
                "k_vars": len(gcols),
                "cf_y": cf_y,
                "cf_d": cf_d,
                "rho": rho,
                "delta_theta": dtheta,
            })
            logger.info(f"      {GROUP_LABELS.get(group_name, group_name):<22} "
                        f"cf_y={cf_y:.4f}  cf_d={cf_d:.4f}  dtheta={dtheta:+.4f}")
    return rows


# =============================================================================
# IRM sensitivity path — exposes the positivity/overlap failure empirically.
#
# The headline IRM specific-method estimand is method-vs-none on the SINGLE-METHOD
# subsample (treated = method-only HH, control = no-treatment HH; mirrors
# prepare_model_data(vs_none=True)).  Country FE predict adoption almost perfectly
# inside that subsample, so the propensity is near-degenerate and the native
# sensitivity_benchmark returns cf_d ~ 1 for the country group -- the empirical
# smoking gun for the overlap failure.  IRM exposes a native sensitivity_benchmark
# (unlike the APOS contrast framework), so no gain-formula replication is needed.
#
# We also run the any-treatment IRM on the FULL sample (no subsample) as the
# well-behaved contrast point (cf_d small) for comparison in the same table.
# =============================================================================

# (treatment_var, table row key, label).  water_treatment = full-sample any-vs-none
# (the GOOD overlap case); the four specifics are the single-method-subsample
# (vs_none) contrasts where overlap breaks.
IRM_TREATMENTS = [
    ("water_treatment", "water_treatment", "Any treatment"),
    ("treat_boil", "treat_boil", "Boil"),
    ("treat_chlorine", "treat_chlorine", "Chlorine"),
    ("treat_filter", "treat_filter", "Filter"),
    ("treat_other", "treat_other", "Other"),
]


def _named_subsample_irm(dt, outcome_var, treatment_var, confounder_groups):
    """Build the IRM estimation sample with REAL column names + a group map.

    For a specific method (``treat_*``) this reproduces the ``vs_none`` subsample
    (method-only treated ∪ no-treatment control); for ``water_treatment`` it is
    the full sample.  Returns (df, names, group_map, n) or None.
    """
    dt = dt.copy()
    is_specific = treatment_var.startswith("treat_")
    if is_specific:
        if not {"single_method", "water_treatment"}.issubset(dt.columns):
            logger.error(f"    IRM vs_none needs single_method/water_treatment for {treatment_var}")
            return None
        treated = (dt[treatment_var] == 1) & (dt["single_method"] == 1)
        control = (dt["water_treatment"] == 0)
        dt = dt[treated | control].copy()

    dt, names, group_map = _named_model_matrix(dt, outcome_var, confounder_groups)
    Xmat = dt[names].to_numpy(dtype=float)
    mask = (
        dt[outcome_var].notna()
        & dt[treatment_var].notna()
        & ~np.isnan(Xmat).any(axis=1)
        & ~np.isinf(Xmat).any(axis=1)
    )
    n = int(mask.sum())
    if n < MIN_OBSERVATIONS:
        logger.error(f"    IRM too few complete obs (N={n})")
        return None
    sub = dt.loc[mask].reset_index(drop=True)
    d = sub[treatment_var].astype(int).to_numpy()
    if int(d.sum()) < 20 or int(len(d) - d.sum()) < 20:
        logger.error(f"    IRM insufficient variation (treated={int(d.sum())})")
        return None

    df = sub[names].astype(float).copy()
    df["_y"] = sub[outcome_var].to_numpy(dtype=float)
    df["_d"] = d.astype(int)
    return df, names, group_map, n


def _fit_named_irm(dt, outcome_var, treatment_var, confounder_groups, learner_name,
                   n_folds, n_rep, dataset_type="HH", skip_checkpoint=False):
    """Complete-case + vs_none subsample -> fit a named DoubleMLIRM (checkpointed)."""
    prepared = _named_subsample_irm(dt, outcome_var, treatment_var, confounder_groups)
    if prepared is None:
        return None
    df, names, group_map, n = prepared

    def _build():
        data = dml.DoubleMLData(df, y_col="_y", d_cols="_d", x_cols=names)
        lset = create_learners_for_binary()[learner_name]
        irm = dml.DoubleMLIRM(
            data, ml_g=clone(lset["g"]), ml_m=clone(lset["m"]),
            n_folds=n_folds, n_rep=n_rep,
            trimming_rule="truncate", trimming_threshold=0.01,
        )
        irm.fit()
        return irm

    cp = (CHECKPOINT_DIR /
          f"irm_sens_{dataset_type}_{outcome_var}_{treatment_var}_{learner_name}.pkl")
    irm = load_or_fit(cp, _build, skip_checkpoint=skip_checkpoint)
    return irm, df, names, group_map, n


def _extract_propensity(irm, df):
    """Per-observation DDML cross-fitted propensity m(X) (averaged over reps)."""
    try:
        m = np.asarray(irm.predictions["ml_m"], dtype=float)  # (n_obs, n_rep, 1)
        m = np.squeeze(m)
        if m.ndim == 2:
            m = m.mean(axis=1)
        return m, df["_d"].to_numpy(dtype=int)
    except Exception:  # noqa: BLE001
        return None, None


def run_spec_irm(dt, outcome, confounders, dataset_type, learner_name, n_folds, n_rep,
                 skip_checkpoint=False, prop_sink=None):
    """One (dataset, outcome): IRM RV + native per-group sensitivity_benchmark for
    any-treatment and every specific method (vs none, single-method subsample).

    If ``prop_sink`` (a list) is given, the per-observation cross-fitted propensity
    m(X) is appended per treatment for the direct overlap figure.
    """
    out_var = outcome["var"]
    logger.info(f"\n=== SENS(IRM) | {dataset_type} | {outcome['label']} ===")
    rows = []
    for treatment_var, tkey, tlabel in IRM_TREATMENTS:
        if treatment_var not in dt.columns:
            continue
        fitted = _fit_named_irm(dt, out_var, treatment_var, confounders, learner_name,
                                n_folds, n_rep, dataset_type=dataset_type,
                                skip_checkpoint=skip_checkpoint)
        if fitted is None:
            logger.warning(f"    IRM {tlabel} skipped (insufficient data)")
            continue
        irm, df, names, group_map, n = fitted
        ate = float(np.ravel(irm.coef)[0])
        se = float(np.ravel(irm.se)[0])

        if prop_sink is not None:
            m_vals, d_vals = _extract_propensity(irm, df)
            if m_vals is not None:
                prop_sink.append({"label": tlabel, "m": m_vals, "d": d_vals})

        rv = rva = None
        try:
            irm.sensitivity_analysis(cf_y=SENS_CF, cf_d=SENS_CF, rho=SENS_RHO)
            sp = irm.sensitivity_params
            if sp is not None:
                rv = float(np.ravel(sp["rv"])[0])
                rva = float(np.ravel(sp["rva"])[0])
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"    IRM {tlabel} sensitivity_analysis failed: {exc}")

        logger.info(f"    {tlabel:<12} ATE={ate:+.4f} (se {se:.4f})  "
                    f"RV={None if rv is None else round(rv,4)} "
                    f"RVa={None if rva is None else round(rva,4)}")

        for group_name, gcols in group_map.items():
            if not gcols:
                continue
            try:
                bdf = irm.sensitivity_benchmark(benchmarking_set=list(gcols))
                r0 = bdf.iloc[0]
                cf_y = float(r0["cf_y"]); cf_d = float(r0["cf_d"])
                dtheta = float(r0["delta_theta"])
                rho = float(r0["rho"]) if "rho" in r0 else float("nan")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"    IRM benchmark {group_name} [{tlabel}] failed: {exc}")
                continue
            rows.append({
                "method": "IRM",
                "dataset_type": dataset_type,
                "outcome": out_var,
                "outcome_label": outcome["label"],
                "treatment": tkey,
                "treatment_label": tlabel,
                "treatment_level": None,
                "learner": learner_name,
                "n": n,
                "ate": ate,
                "se": se,
                "rv_q": rv,
                "rv_qa": rva,
                "group": group_name,
                "group_label": GROUP_LABELS.get(group_name, group_name),
                "k_vars": len(gcols),
                "cf_y": cf_y,
                "cf_d": cf_d,
                "rho": rho,
                "delta_theta": dtheta,
            })
            logger.info(f"      {GROUP_LABELS.get(group_name, group_name):<22} "
                        f"cf_y={cf_y:.4f}  cf_d={cf_d:.4f}  dtheta={dtheta:+.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="APOS + IRM confounder benchmarking / sensitivity.")
    ap.add_argument("--learner", default="stacked",
                    help="benchmark learner (default stacked = headline; 'lasso' is a fast approx)")
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--n-rep", type=int, default=1,
                    help="cross-fitting reps (1 is fine for the benchmark; default 1)")
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="ignore cached APOS fits and refit from scratch")
    ap.add_argument("--methods", nargs="+", default=["apos", "irm"],
                    choices=["apos", "irm"],
                    help="which estimands to benchmark (default both)")
    args = ap.parse_args()

    setup_environment()
    logger.info("=" * 70)
    logger.info(f"MICS DDML: SENSITIVITY / CONFOUNDER BENCHMARKING "
                f"(learner={args.learner}, methods={'+'.join(args.methods)})")
    logger.info("=" * 70)

    hh_dt, u5_dt = load_data()
    data_by_key = {"hh": hh_dt, "u5": u5_dt}
    hh_conf = dict(BASE_CONFOUNDERS)
    u5_conf = {**BASE_CONFOUNDERS, **U5_ADDITIONAL_CONFOUNDERS}
    conf_by_ds = {"HH": hh_conf, "U5": u5_conf}

    all_rows = []
    for dataset_type, key, outcomes in DATASET_OUTCOMES:
        dt = data_by_key[key]
        conf = conf_by_ds[dataset_type]
        for outcome in outcomes:
            if outcome["var"] not in dt.columns:
                logger.warning(f"SENS: outcome '{outcome['var']}' missing in {dataset_type}, skipping")
                continue
            if "apos" in args.methods:
                all_rows += run_spec(
                    dt, outcome, conf, dataset_type, args.learner, args.n_folds, args.n_rep,
                    skip_checkpoint=args.no_checkpoint,
                )
            if "irm" in args.methods:
                prop_sink = []
                all_rows += run_spec_irm(
                    dt, outcome, conf, dataset_type, args.learner, args.n_folds, args.n_rep,
                    skip_checkpoint=args.no_checkpoint, prop_sink=prop_sink,
                )
                if prop_sink:
                    plot_propensity_overlap_panel(
                        prop_sink,
                        title=(f"DDML propensity overlap — {dataset_type} | "
                               f"{outcome['label']} (IRM, {args.learner})"),
                        filename=f"overlap_{dataset_type}_{outcome['var']}_{args.learner}.png",
                    )

    if not all_rows:
        logger.warning("SENSITIVITY produced no results.")
        return

    df = pd.DataFrame(all_rows)
    csv_path = OUTPUT_DIR / "results_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    df.to_pickle(OUTPUT_DIR / "results_sensitivity.pkl")
    logger.info("=" * 70)
    logger.info(f"SENSITIVITY complete — {len(df)} spec x group rows -> {csv_path}")

    # Single headline benchmark table mirroring the results tables: Panel A is the
    # any-treatment IRM (full sample), Panel B the APOS method-vs-none contrasts.
    # The specific-method IRM rows stay in the CSV (overlap exhibit / propensity
    # figures) but are no longer tabulated.
    table_rows = df[(df["method"] == "APOS")
                    | ((df["method"] == "IRM") & (df["treatment"] == "water_treatment"))]
    if not table_rows.empty:
        tex = create_sensitivity_benchmark_table(
            table_rows, learner=args.learner,
            filename="table_sensitivity_benchmark.tex",
            caption=(r"Confounder Benchmarking --- Observed Controls vs.\ the Robustness "
                     r"Value (Panel A: any-treatment IRM; Panel B: APOS method contrasts)"),
            label="tab:sensitivity-benchmark",
            # 3 outcomes x 3 sub-cols (10 cols) reads better upright -> portrait.
            landscape=False)
        logger.info(f"Sensitivity benchmark table -> {tex}")


if __name__ == "__main__":
    main()
