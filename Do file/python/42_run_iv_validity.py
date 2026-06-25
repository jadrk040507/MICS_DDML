"""
Empirical instrument-validity tests for the Wald/IV reframing.

Treats household water treatment Z as an instrument for point-of-use E. coli D,
with child diarrhea Y as the outcome, on the U5 sample, conditioning on the
headline confounder vector X.  Reports the testable instrument conditions:

  1. RELEVANCE (testable)        -- first stage Z -> D; partial F.  Also the four
                                    methods (boil/chlorine/filter/other) as four
                                    separate instruments, joint first-stage F.
  2. REDUCED FORM               -- Z -> Y.
  3. JUST-IDENTIFIED WALD       -- LATE = (reduced form) / (first stage), any-treatment.
  4. OVERIDENTIFICATION (Sargan) -- the only empirical handle on EXCLUSION: with the
                                    four methods as 4 instruments for the single D,
                                    test whether they identify the SAME D->Y effect.
                                    Rejection => methods have method-specific direct
                                    effects on diarrhea (exclusion violated) OR the
                                    complier LATEs differ across methods.
  5. MONOTONICITY (partial check) -- first-stage sign of Z on D within RiskSource and
                                    wealth groups; all same sign => no obvious defiers.

NOTE: exogeneity of Z (conditional independence) is NOT testable and equals the
paper's CIA; exclusion is NOT point-testable with a single instrument -- the Sargan
test only checks it given >=2 instruments and assumes effect homogeneity.

Usage:  cd "Do file/Python"; python 42_run_iv_validity.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from _config import BASE_CONFOUNDERS, U5_ADDITIONAL_CONFOUNDERS, OUTPUT_DIR, logger
from _data import prepare_u5_data, create_model_matrix
from _runners import setup_environment

D_VARS = [("SomeRiskHome", "any point-of-use E. coli"),
          ("VeryHighRiskHome", "very-high point-of-use E. coli")]
Y_VAR = "diarrhea"
Z_VAR = "water_treatment"
METHODS = ["treat_boil", "treat_chlorine", "treat_filter", "treat_other"]


def _cluster_ols(y, Xmat, groups):
    """OLS with cluster-robust (HHID) covariance."""
    return sm.OLS(y, Xmat).fit(cov_type="cluster", cov_kwds={"groups": groups})


def _build(dt_clean, Xcols):
    const = np.ones((len(dt_clean), 1))
    Xexog = np.column_stack([const, dt_clean[Xcols].to_numpy(dtype=float)])
    return Xexog


def _two_sls(Y, D, Xexog, Inst):
    """Manual 2SLS: endog D, exog Xexog (incl const), instruments Inst (excl-only).

    Returns beta on D, full coef vector [D, Xexog], residuals (actual D), and the
    instrument matrix [Xexog, Inst] used for the Sargan regression.
    """
    Zfull = np.column_stack([Xexog, Inst])              # full instrument set
    # Stage 1: project D on Zfull
    pi, *_ = np.linalg.lstsq(Zfull, D, rcond=None)
    Dhat = Zfull @ pi
    # Stage 2: regress Y on [Dhat, Xexog]
    RHS_hat = np.column_stack([Dhat, Xexog])
    coef, *_ = np.linalg.lstsq(RHS_hat, Y, rcond=None)
    # residuals use ACTUAL D
    RHS_act = np.column_stack([D, Xexog])
    resid = Y - RHS_act @ coef
    return coef[0], coef, resid, Zfull


def _sargan(resid, Zfull, n_endog, n_instr_excl):
    """Sargan overid stat = n * R^2 from regressing 2SLS residuals on all instruments."""
    n = len(resid)
    aux = sm.OLS(resid, Zfull).fit()
    r2 = aux.rsquared
    stat = n * r2
    dof = n_instr_excl - n_endog
    pval = 1 - stats.chi2.cdf(stat, dof) if dof > 0 else np.nan
    return stat, dof, pval


def _run_for_D(dt, d_var, d_label, Xcols, out_lines):
    sub = dt.dropna(subset=[Y_VAR, d_var, Z_VAR] + METHODS + Xcols + ["HHID"]).copy()
    n = len(sub)
    Y = sub[Y_VAR].to_numpy(dtype=float)
    D = sub[d_var].to_numpy(dtype=float)
    Z = sub[Z_VAR].to_numpy(dtype=float)
    M = sub[METHODS].to_numpy(dtype=float)
    Xexog = _build(sub, Xcols)
    hhid = sub["HHID"].to_numpy()

    out_lines.append("")
    out_lines.append("=" * 78)
    out_lines.append(f"D = {d_var}  ({d_label})   |   Y = {Y_VAR}   |   Z = {Z_VAR}   |   n = {n:,}")
    out_lines.append("=" * 78)

    # --- 1. Relevance: any-treatment first stage  D ~ Z + X ---
    fs = _cluster_ols(D, np.column_stack([Z[:, None], Xexog]), hhid)
    fs_b, fs_se = fs.params[0], fs.bse[0]
    fs_t = fs_b / fs_se
    fs_F = fs_t ** 2
    out_lines.append("\n[1] RELEVANCE -- first stage (any treatment), cluster-robust on HHID")
    out_lines.append(f"    pi(Z->D) = {fs_b:+.4f}  (se {fs_se:.4f}, t {fs_t:+.2f})   partial F = {fs_F:.1f}"
                     f"   {'STRONG' if fs_F > 10 else 'WEAK'}")

    # --- 1b. Methods as 4 instruments: joint first-stage F ---
    fsm = _cluster_ols(D, np.column_stack([M, Xexog]), hhid)
    R = np.zeros((len(METHODS), fsm.params.shape[0]))
    for i in range(len(METHODS)):
        R[i, i] = 1.0
    wald = fsm.wald_test(R, scalar=True)
    Fm = float(wald.statistic)
    pm = float(wald.pvalue)
    out_lines.append("    methods as 4 instruments -- first-stage coefs (cluster-robust):")
    for i, mname in enumerate(METHODS):
        out_lines.append(f"        {mname:<16} {fsm.params[i]:+.4f}  (se {fsm.bse[i]:.4f}, t {fsm.params[i]/fsm.bse[i]:+.2f})")
    out_lines.append(f"        joint Wald F(all=0) = {Fm:.1f}  (p {pm:.3g})   {'STRONG' if Fm > 10 else 'WEAK'}")

    # --- 2. Reduced form  Y ~ Z + X ---
    rf = _cluster_ols(Y, np.column_stack([Z[:, None], Xexog]), hhid)
    rf_b, rf_se = rf.params[0], rf.bse[0]
    out_lines.append("\n[2] REDUCED FORM  Y ~ Z + X  (cluster-robust)")
    out_lines.append(f"    ITT(Z->Y) = {rf_b:+.4f}  (se {rf_se:.4f}, t {rf_b/rf_se:+.2f}, p {2*(1-stats.norm.cdf(abs(rf_b/rf_se))):.3g})")

    # --- 3. Just-identified Wald (any treatment) = RF / FS ---
    wald_ratio = rf_b / fs_b
    out_lines.append("\n[3] JUST-IDENTIFIED WALD (any treatment) = ITT(Z->Y) / pi(Z->D)")
    out_lines.append(f"    LATE(D->Y) ~ {wald_ratio:+.4f}   [reduced form {rf_b:+.4f} / first stage {fs_b:+.4f}]")

    # --- 4. Overidentification (Sargan) with 4 method instruments ---
    beta_oid, _, resid, Zfull = _two_sls(Y, D, Xexog, M)
    stat, dof, pval = _sargan(resid, Zfull, n_endog=1, n_instr_excl=len(METHODS))
    out_lines.append("\n[4] OVERIDENTIFICATION / EXCLUSION TEST (Sargan), methods as 4 instruments")
    out_lines.append(f"    2SLS LATE(D->Y) (overid) = {beta_oid:+.4f}")
    out_lines.append(f"    Sargan J = {stat:.2f}  ~ chi2({dof})   p = {pval:.4f}   "
                     f"{'REJECT exclusion/homogeneity' if pval < 0.05 else 'fail to reject'}")
    out_lines.append("    (rejection => methods disagree on D->Y: method-specific direct effects "
                     "on diarrhea\n     [exclusion violated] OR complier LATEs differ across methods.)")

    # --- 5. Monotonicity: first-stage sign of Z->D within groups ---
    out_lines.append("\n[5] MONOTONICITY (partial) -- first-stage pi(Z->D) within subgroups")
    out_lines.append("    (all same sign => no obvious defiers; treatment should weakly reduce E. coli)")
    for gvar, gmap in [("RiskSource", {0: "No risk", 1: "Moderate", 2: "Very high"}),
                       ("windex5", {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5"})]:
        if gvar not in sub.columns:
            continue
        out_lines.append(f"    by {gvar}:")
        for gv, glab in gmap.items():
            m = (sub[gvar] == gv).to_numpy()
            if m.sum() < 100 or np.unique(Z[m]).size < 2:
                out_lines.append(f"        {glab:<10} n={int(m.sum()):>6}  (insufficient variation)")
                continue
            Xg = np.column_stack([Z[m, None], Xexog[m]])
            # drop all-constant columns within subgroup to avoid singularity
            keep = np.array([np.ptp(Xg[:, j]) > 0 or j == 0 for j in range(Xg.shape[1])])
            try:
                fg = _cluster_ols(D[m], Xg[:, keep], hhid[m])
                b = fg.params[0]; se = fg.bse[0]
                out_lines.append(f"        {glab:<10} n={int(m.sum()):>6}  pi={b:+.4f} (se {se:.4f}, t {b/se:+.2f})")
            except Exception as exc:  # noqa: BLE001
                out_lines.append(f"        {glab:<10} n={int(m.sum()):>6}  (fit failed: {exc})")

    return {
        "D": d_var, "n": n,
        "first_stage_any": fs_b, "first_stage_F": fs_F,
        "methods_joint_F": Fm,
        "reduced_form": rf_b, "reduced_form_se": rf_se,
        "wald_any": wald_ratio, "wald_2sls_overid": beta_oid,
        "sargan_stat": stat, "sargan_dof": dof, "sargan_p": pval,
    }


def main():
    setup_environment()
    logger.info("=" * 70)
    logger.info("MICS DDML: empirical instrument-validity tests (treatment as IV for E. coli)")
    logger.info("=" * 70)

    dt = prepare_u5_data()
    conf = {**BASE_CONFOUNDERS, **U5_ADDITIONAL_CONFOUNDERS}
    # Build the confounder column names once (use any outcome; columns are outcome-agnostic).
    _, dt2, Xcols = create_model_matrix(dt, Y_VAR, conf, return_names=True)

    out_lines = []
    summary = []
    for d_var, d_label in D_VARS:
        if d_var not in dt2.columns:
            logger.warning(f"D variable {d_var} missing; skipping")
            continue
        summary.append(_run_for_D(dt2, d_var, d_label, Xcols, out_lines))

    report = "\n".join(out_lines)
    print(report)
    (OUTPUT_DIR / "iv_validity_report.txt").write_text(report, encoding="utf-8")
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "iv_validity_summary.csv", index=False)
    logger.info(f"\nWrote {OUTPUT_DIR/'iv_validity_report.txt'} and iv_validity_summary.csv")


if __name__ == "__main__":
    main()
