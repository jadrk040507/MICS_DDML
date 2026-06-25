"""
Characteristics associated with water-treatment adoption.

Descriptive model of which household/environmental covariates predict \emph{any}
water treatment (HH sample), across five learners:

  * Logit       --- full logistic regression: signed log-odds coefficients with
                    p-values (the interpretable "associated with" headline).
  * LASSO       --- L1 logistic: which predictors survive selection.
  * Elastic net --- L1/L2 logistic: middle ground.
  * Random forest, XGBoost --- nonlinear (unsigned) variable importance.

All share the SAME covariate matrix as the causal models (BASE_CONFOUNDERS),
including country fixed effects.  Country dummies are kept in every model but
omitted from the printed table (nuisance).  Outputs: a tidy per-feature CSV
(results_treatment_predictors.csv) and a combined LaTeX table
(table_treatment_predictors.tex) for the "Characteristics associated with Water
Treatment" subsection.
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import (
    HH_DATA_FILE, BASE_CONFOUNDERS, RANDOM_STATE, N_JOBS, OUTPUT_DIR,
    WS1G_CODES,
)
from _data import prepare_hh_data, create_model_matrix

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import xgboost as xgb


# ---- feature name -> (group, readable label); None = drop from table ----
def label_feature(name):
    if name.startswith("country_cat"):
        return None  # country FE: kept in model, omitted from table
    if name.startswith("wealth_q"):
        return ("Wealth quintile", f"Q{name[-1]} (vs Q1)")
    if name == "edu_primary":
        return ("Education (head)", "Primary")
    if name == "edu_secondary":
        return ("Education (head)", "Secondary or higher")
    if name == "edu_missing":
        return ("Education (head)", "Missing")
    if name == "urban_bin":
        return ("Residence", "Urban")
    if name.startswith("ws1g_"):
        code = int(name.split("_")[1])
        return ("Water source", WS1G_CODES.get(code, name).replace("_", " ").capitalize())
    if name == "Any_U5":
        return ("Household composition", "Any under-5 child")
    if name == "Girls_less_than15":
        return ("Household composition", "Girl aged $<$15")
    if name == "Boys_15or_less":
        return ("Household composition", "Boy aged $\\leq$15")
    if name.startswith("toilet_"):
        return ("Sanitation", name.replace("toilet_", "").replace("_", " ").capitalize())
    if name.startswith("wq27_d_"):
        return ("Source E.\\,coli", f"Decile {name.split('_')[-1]}")
    return ("Other", name)


GROUP_ORDER = ["Wealth quintile", "Education (head)", "Residence", "Water source",
               "Household composition", "Sanitation", "Source E.\\,coli", "Other"]


def _stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def main():
    print("=" * 70)
    print("MICS DDML: CHARACTERISTICS ASSOCIATED WITH WATER TREATMENT")
    print("=" * 70)

    dt = prepare_hh_data(HH_DATA_FILE)
    X, dt, names = create_model_matrix(dt, "SomeRiskHome", return_names=True)
    D = dt["water_treatment"].to_numpy(dtype=float)
    mask = dt["water_treatment"].notna().to_numpy() & ~np.isnan(X).any(axis=1)
    X, D = X[mask], D[mask]
    print(f"  N = {len(D):,}   treatment rate = {D.mean():.3f}   p = {X.shape[1]} covariates")

    res = {n: {"name": n} for n in names}
    auc = {}
    # shuffled folds: the data are country-ordered, so unshuffled CV breaks the
    # tree models' out-of-fold predictions (test folds = unseen countries).
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    def _cvauc(est):
        return float(cross_val_score(est, X, D, cv=cv, scoring="roc_auc", n_jobs=N_JOBS).mean())

    # --- Logit (full, signed log-odds + p-values) ---
    print("  fitting logit ...")
    Xc = sm.add_constant(X, has_constant="add")
    logit = sm.Logit(D, Xc).fit(disp=0, maxiter=200, method="lbfgs")
    coef = logit.params[1:]; pval = logit.pvalues[1:]
    for i, n in enumerate(names):
        res[n]["logit_coef"] = float(coef[i]); res[n]["logit_p"] = float(pval[i])
    auc["Logit"] = _cvauc(LogisticRegression(penalty=None, max_iter=500, solver="lbfgs"))

    # --- LASSO logit ---
    print("  fitting lasso ...")
    la = LogisticRegressionCV(cv=5, penalty="l1", solver="saga", max_iter=4000,
                              n_jobs=N_JOBS, random_state=RANDOM_STATE, scoring="roc_auc")
    la.fit(X, D)
    for i, n in enumerate(names):
        res[n]["lasso_coef"] = float(la.coef_.ravel()[i])
    auc["LASSO"] = _cvauc(LogisticRegression(penalty="l1", solver="saga", C=float(la.C_[0]), max_iter=4000))

    # --- Elastic net logit ---
    print("  fitting elastic net ...")
    en = LogisticRegressionCV(cv=5, penalty="elasticnet", solver="saga", l1_ratios=[0.5],
                              max_iter=4000, n_jobs=N_JOBS, random_state=RANDOM_STATE,
                              scoring="roc_auc")
    en.fit(X, D)
    for i, n in enumerate(names):
        res[n]["enet_coef"] = float(en.coef_.ravel()[i])
    auc["ENet"] = _cvauc(LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=float(en.C_[0]), max_iter=4000))

    # --- Random forest importance ---
    print("  fitting random forest ...")
    rf = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=20,
                                n_jobs=N_JOBS, random_state=RANDOM_STATE)
    rf.fit(X, D)
    for i, n in enumerate(names):
        res[n]["rf_imp"] = float(rf.feature_importances_[i])
    auc["RF"] = _cvauc(rf)

    # --- XGBoost importance ---
    print("  fitting xgboost ...")
    xg = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                           random_state=RANDOM_STATE, n_jobs=N_JOBS)
    xg.fit(X, D)
    for i, n in enumerate(names):
        res[n]["xgb_imp"] = float(xg.feature_importances_[i])
    auc["XGB"] = _cvauc(xg)

    print("  AUC: " + "  ".join(f"{k}={v:.3f}" for k, v in auc.items()))

    # --- tidy CSV (all features, incl. country) ---
    df = pd.DataFrame(res.values())
    out_dir = OUTPUT_DIR / "treatment_predictors"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results_treatment_predictors.csv", index=False)

    # --- LaTeX table: substantive features, RANKED by tree importance ---
    feats = []
    for n in names:
        lab = label_feature(n)
        if lab is None:  # country FE: kept in model, omitted from table
            continue
        r = res[n]
        imp = 0.5 * (r.get("rf_imp", 0.0) + r.get("xgb_imp", 0.0))  # mean tree importance
        feats.append((imp, lab[0], lab[1], r))
    feats.sort(key=lambda t: t[0], reverse=True)
    # rank-order persisted to the CSV too
    rank = {id(t[3]): i + 1 for i, t in enumerate(feats)}
    for n in names:
        res[n]["importance_rank"] = rank.get(id(res[n]), "")
    pd.DataFrame(res.values()).to_csv(out_dir / "results_treatment_predictors.csv", index=False)

    L = []
    L += [r"% Requires: \usepackage{booktabs}",
          r"\begin{table}[htbp]\centering",
          r"\caption{Characteristics Associated with Water-Treatment Adoption (ranked by importance)}",
          r"\label{tab:treatment_predictors}",
          r"\small\setlength{\tabcolsep}{5pt}",
          r"\begin{tabular}{lccccc}",
          r"\toprule",
          r"Covariate & Logit & Lasso & E.\,net & RF & XGB \\",
          r" & (log-odds) & \multicolumn{2}{c}{(penalised coef.)} & \multicolumn{2}{c}{(importance $\times100$)} \\",
          r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
          r"\midrule"]
    for imp, group, label, r in feats:
        lc = f"{r.get('logit_coef', float('nan')):+.2f}{_stars(r.get('logit_p', 1))}"
        las = f"{r.get('lasso_coef', 0):+.2f}"
        ent = f"{r.get('enet_coef', 0):+.2f}"
        rfi = f"{100*r.get('rf_imp', 0):.1f}"
        xgi = f"{100*r.get('xgb_imp', 0):.1f}"
        L.append(rf"{label} \textit{{({group})}} & {lc} & {las} & {ent} & {rfi} & {xgi} \\")
    L.append(r"\midrule")
    L.append(rf"AUC & {auc['Logit']:.3f} & {auc['LASSO']:.3f} & {auc['ENet']:.3f} & "
             rf"{auc['RF']:.3f} & {auc['XGB']:.3f} \\")
    notes = (r"\textit{Notes:} Outcome is an indicator for any household water treatment "
             rf"($N={len(D):,}$, adoption rate ${D.mean():.3f}$). All models include country "
             r"fixed effects (omitted from the table). Logit reports signed log-odds "
             r"coefficients ($^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$); Lasso and elastic "
             r"net report penalised logistic coefficients (zero $=$ not selected); RF/XGB "
             r"report Gini/gain importance ($\times100$). Logit AUC "
             rf"${auc['Logit']:.3f}$.")
    L += [r"\bottomrule", r"\end{tabular}", r"\par\vspace{3pt}",
          rf"\begin{{minipage}}{{\linewidth}}\footnotesize {notes}\end{{minipage}}",
          r"\end{table}"]
    tex = out_dir / "table_treatment_predictors.tex"
    tex.write_text("\n".join(L), encoding="utf-8")
    print(f"  table -> {tex}")
    print("=" * 70)


if __name__ == "__main__":
    main()
