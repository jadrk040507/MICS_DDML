"""Predictors of household water-treatment adoption.

This is the presentation-friendly counterpart of the old treatment-predictors
script.  The model matrix is deliberately kept in sync with ``mics.py``:

    i.windex5 i.country_cat i.urban i.WS1_g i.Toilet i.wq27_decile
    Any_U5 Girls_less_than15 Boys_15or_less

For the U5 sample it additionally includes ``i.age`` and ``male``.  The script
fits the three requested descriptive learners separately for each treatment
versus no-treatment contrast and writes a tidy CSV plus grouped LaTeX tables.
"""

from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import pyreadstat
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_score
import xgboost as xgb

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Data" / "3. Final"
OUTPUT_DIR = ROOT / "Output"
SEED = 42
N_JOBS = max(1, min(8, (os.cpu_count() or 2) // 2))
CV_FOLDS = 3
TREE_ESTIMATORS = 200


WATER_SOURCE_LABELS = {
    11: "Piped supply",
    21: "Tube well or borehole",
    31: "Protected well or spring",
    32: "Unprotected well or spring",
    51: "Surface water or rainwater",
    91: "Packaged water",
    96: "Other water source",
}
TOILET_LABELS = {
    1: "Pit latrine",
    2: "Open defecation",
    3: "Missing sanitation category",
    98: "Other sanitation category",
}
COUNTRY_LABELS = {
    1: "Argentina", 2: "Bangladesh", 3: "Benin",
    4: "Central African Republic", 5: "Chad", 6: "Costa Rica",
    7: "Cuba", 8: "DR Congo", 9: "Dominican Republic",
    10: "Eswatini", 11: "Fiji", 12: "Gambia", 13: "Ghana",
    14: "Guinea Bissau", 15: "Guyana", 16: "Honduras",
    17: "Kiribati", 18: "Lao PDR", 19: "Lesotho", 20: "Madagascar",
    21: "Malawi", 22: "Mongolia", 23: "Nepal", 24: "Sierra Leone",
    25: "Suriname", 26: "Togo", 27: "Tonga", 28: "Trinidad and Tobago",
    29: "Turks and Caicos Islands", 30: "Tuvalu", 31: "Viet Nam",
    32: "Zimbabwe",
}


GROUP_ORDER = [
    "Household wealth", "Residence", "Drinking-water source", "Sanitation",
    "Household composition", "Source-water E. coli", "Country fixed effects",
    "Child characteristics",
]

TREATMENTS = {
    "any": "Any treatment",
    1: "Boiling",
    2: "Chlorination/tablets",
    3: "Straining/settling",
}


def _feature_metadata(name, child):
    """Return (group, readable label, interpretation) for an encoded column."""
    if name.startswith("country_"):
        code = int(float(name.split("_")[-1]))
        return ("Country fixed effects", COUNTRY_LABELS.get(code, f"Country {code}"),
                "Relative to the omitted country fixed-effect category")
    if name.startswith("wealth_"):
        q = int(float(name.split("_")[-1]))
        return ("Household wealth", f"Wealth-index quintile {q}",
                "Relative to the poorest wealth quintile")
    if name == "urban":
        return ("Residence", "Urban residence", "Relative to rural residence")
    if name.startswith("water_source_"):
        code = int(float(name.split("_")[-1]))
        return ("Drinking-water source", WATER_SOURCE_LABELS.get(code, f"Water source {code}"),
                "Relative to the omitted water-source category")
    if name.startswith("toilet_"):
        code = int(float(name.split("_")[-1]))
        return ("Sanitation", TOILET_LABELS.get(code, f"Sanitation category {code}"),
                "Relative to flush toilet")
    if name.startswith("source_ecoli_"):
        decile = int(float(name.split("_")[-1]))
        return ("Source-water E. coli", f"Source-water E. coli decile {decile}",
                "CFU per 100 ml; relative to the omitted decile")
    simple = {
        "Any_U5": ("Household composition", "Any child under age five",
                    "Indicator for at least one under-five child"),
        "Girls_less_than15": ("Household composition", "Girl under age fifteen",
                               "Indicator for a girl younger than fifteen"),
        "Boys_15or_less": ("Household composition", "Boy age fifteen or younger",
                            "Indicator for a boy aged fifteen or younger"),
        "male": ("Child characteristics", "Male child", "Relative to female child"),
    }
    if name in simple:
        return simple[name]
    if child and name.startswith("child_age_"):
        age = int(float(name.split("_")[-1]))
        return ("Child characteristics", f"Child age: {age} years",
                "Relative to the omitted child-age category")
    return ("Other", name, "Encoded model covariate")


def _make_frame(data, child, treatment_code):
    controls = [
        "windex5", "urban", "WS1_g", "wq27_decile", "Any_U5",
        "Girls_less_than15", "Boys_15or_less", "Toilet",
    ]
    treatment_column = "water_treatment" if treatment_code == "any" else "WQ15_g"
    required = [treatment_column, "country_cat", "Cluster_var", *controls]
    if child:
        required.extend(["age", "male"])
    sample = data.loc[data[required].notna().all(axis=1)].copy()
    if treatment_code == "any":
        sample = sample[sample["water_treatment"].isin([0, 1])].copy()
    else:
        sample = sample[sample["WQ15_g"].isin([0, treatment_code])].copy()

    categorical = [
        ("windex5", "wealth"), ("country_cat", "country"),
        ("WS1_g", "water_source"), ("Toilet", "toilet"),
        ("wq27_decile", "source_ecoli"),
    ]
    blocks = [
        pd.get_dummies(sample[column].astype("string"), prefix=prefix,
                       drop_first=True, dtype=float)
        for column, prefix in categorical
    ]
    blocks.append(sample[["urban", "Any_U5", "Girls_less_than15",
                          "Boys_15or_less"]].astype(float))
    if child:
        blocks.append(pd.get_dummies(sample["age"].astype("string"),
                                     prefix="child_age", drop_first=True,
                                     dtype=float))
        blocks.append(sample[["male"]].astype(float))
    X = pd.concat(blocks, axis=1)
    X = X.reindex(sorted(X.columns), axis=1)
    D = (pd.to_numeric(sample[treatment_column], errors="raise") == (
        1 if treatment_code == "any" else treatment_code
    )).to_numpy(float)
    groups = sample["Cluster_var"].to_numpy()
    return X.to_numpy(float), D, list(X.columns), groups


def _fit_sample(data_path, sample_label, child, treatment_code, treatment_label):
    data, _ = pyreadstat.read_dta(data_path) if isinstance(data_path, (str, Path)) else (data_path, None)
    X, D, names, groups = _make_frame(data, child, treatment_code)
    print(f"  {sample_label} — {treatment_label}: N={len(D):,}; adoption rate={D.mean():.3f}; p={X.shape[1]}")
    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    results = {name: {"feature": name} for name in names}

    def auc(estimator):
        return float(cross_val_score(estimator, X, D, groups=groups, cv=cv,
                                     scoring="roc_auc", n_jobs=1).mean())

    aucs = {}
    lasso_search = GridSearchCV(
        LogisticRegression(penalty="l1", solver="saga", max_iter=2000,
                           random_state=SEED),
        param_grid={"C": np.logspace(-3, 2, 10)}, cv=cv, scoring="roc_auc",
        n_jobs=N_JOBS,
    )
    lasso_search.fit(X, D, groups=groups)
    lasso = lasso_search.best_estimator_
    for i, name in enumerate(names):
        results[name]["lasso_coef"] = float(lasso.coef_.ravel()[i])
    aucs["LASSO"] = auc(LogisticRegression(penalty="l1", solver="saga",
                                            C=float(lasso.C), max_iter=2000))

    rf = RandomForestClassifier(n_estimators=TREE_ESTIMATORS, min_samples_leaf=20,
                                n_jobs=N_JOBS, random_state=SEED)
    rf.fit(X, D)
    for i, name in enumerate(names):
        results[name]["rf_imp"] = float(rf.feature_importances_[i])
    aucs["RF"] = auc(rf)

    xgb_model = xgb.XGBClassifier(n_estimators=TREE_ESTIMATORS, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  eval_metric="logloss", random_state=SEED,
                                  n_jobs=N_JOBS)
    xgb_model.fit(X, D)
    for i, name in enumerate(names):
        results[name]["xgb_imp"] = float(xgb_model.feature_importances_[i])
    aucs["XGB"] = auc(xgb_model)

    rows = []
    for name, row in results.items():
        group, label, interpretation = _feature_metadata(name, child)
        row.update(sample=sample_label, treatment=treatment_label,
                   treatment_code=treatment_code, group=group, label=label,
                   interpretation=interpretation,
                   tree_importance=(row.get("rf_imp", 0) + row.get("xgb_imp", 0)) / 2)
        rows.append(row)
    frame = pd.DataFrame(rows)
    # Rank each learner independently.  LASSO uses absolute coefficients for
    # importance, while RF/XGB use their direction-neutral feature scores.
    # The sign of ``lasso_coef`` remains available for interpretation.
    for column in ["lasso_coef", "rf_imp", "xgb_imp"]:
        frame[f"{column}_rank"] = frame[column].abs().rank(pct=True)
        section_max = frame.groupby("group")[f"{column}_rank"].transform("max")
        frame[f"{column}_section_winner"] = frame[f"{column}_rank"].eq(section_max)
        frame[f"{column}_overall_winner"] = frame[f"{column}_rank"].eq(
            frame[f"{column}_rank"].max()
        )
    frame["combined_score"] = frame[["lasso_coef_rank", "rf_imp_rank",
                                      "xgb_imp_rank"]].mean(axis=1)
    frame["importance_rank"] = frame["combined_score"].rank(
        method="first", ascending=False
    ).astype(int)
    return frame, aucs, len(D), float(D.mean())


def _latex_escape(value):
    return str(value).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def _write_table(results_by_treatment, sample_label, appendix=False):
    treatment_items = list(results_by_treatment.items())
    treatment_frames = {label: value[0] for label, value in treatment_items}

    lines = [
        r"% Requires: \usepackage{booktabs, pdflscape, graphicx, adjustbox}",
        r"\begin{landscape}", r"\begin{table}[p]\centering",
        rf"\caption{{Predictors of treatment adoption relative to no treatment: {_latex_escape(sample_label)}}}",
        rf"\label{{tab:characteristics-by-treatment-{'u5' if appendix else 'hh'}}}",
        r"\tiny\setlength{\tabcolsep}{1.8pt}\renewcommand{\arraystretch}{0.88}",
        r"\begin{adjustbox}{max width=\textwidth,center}",
        "\\begin{tabular}{ll*{" + str(3 * len(treatment_items)) + "}{c}}", r"\toprule",
        "Group & Predictor & " + " & ".join(
            rf"\multicolumn{{3}}{{c}}{{{_latex_escape(label)}}}"
            for label, _ in treatment_items
        ) + r" \\",
        " & & " + " & ".join(["LASSO & RF & XGB"] * len(treatment_items)) + r" \\",
        r" & & " + " & ".join(
            [r"(coef.) & (imp.) & (imp.)"] * len(treatment_items)
        ) + r" \\",
        "".join(
            rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}"
            for i in range(len(treatment_items))
        ),
        r"\midrule",
    ]

    # Fixed presentation order: prespecified control blocks, then alphabetic
    # predictor labels within each block. Importance only controls emphasis.
    for group in GROUP_ORDER:
        labels = sorted({
            row.label
            for frame in treatment_frames.values()
            for _, row in frame[frame.group.eq(group)].iterrows()
        })
        if not labels:
            continue
        lines.append(rf"\multicolumn{{{2 + 3 * len(treatment_items)}}}{{l}}{{\textit{{{_latex_escape(group)}}}}} \\")
        for label in labels:
            row_by_treatment = {
                treatment: frame[frame.label.eq(label)].iloc[0]
                for treatment, frame in treatment_frames.items()
                if not frame[frame.label.eq(label)].empty
            }
            display_label = _latex_escape(label)
            values = []
            for treatment, frame in treatment_frames.items():
                row = row_by_treatment.get(treatment)
                if row is None:
                    values.extend(["", "", ""])
                    continue
                cells = []
                for value, column in [
                    (f"{row.lasso_coef:+.2f}", "lasso_coef"),
                    (f"{100 * row.rf_imp:.1f}", "rf_imp"),
                    (f"{100 * row.xgb_imp:.1f}", "xgb_imp"),
                ]:
                    if row[f"{column}_overall_winner"]:
                        value = rf"\textbf{{\underline{{{value}}}}}"
                    elif row[f"{column}_section_winner"]:
                        value = rf"\textbf{{{value}}}"
                    cells.append(value)
                values.extend(cells)
            lines.append(" & ".join(["", display_label, *values]) + r" \\")

    lines.append(r"\midrule")
    lines.append("AUC & & " + " & ".join(
        f"{aucs[k]:.3f}" for _, (_, aucs, _, _) in treatment_items
        for k in ["LASSO", "RF", "XGB"]
    ) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\par\vspace{3pt}",
        rf"\begin{{minipage}}{{\linewidth}}\footnotesize \textit{{Notes:}} Each treatment is compared with no treatment. The table includes all prespecified control blocks. Categories not shown are the reference categories: poorest wealth quintile, rural residence, flush toilet, the first observed water-source and source-water E. coli categories, and the omitted country fixed-effect category. Country fixed effects are included in estimation and displayed for transparency. LASSO reports penalised logistic coefficients; a positive coefficient indicates higher treatment-adoption odds and a negative coefficient indicates lower odds, relative to the reference category. Zero indicates that the predictor was not selected. RF/XGB report direction-neutral feature importance multiplied by 100. For each treatment and learner separately, bold identifies the most predictively important variable within its block; bold and underlining identify the most predictively important variable overall. Thus, the highlighted predictor can differ across LASSO, RF, and XGB.\end{{minipage}}",
        r"\end{table}", r"\end{landscape}",
    ]
    path = OUTPUT_DIR / f"table_characteristics_by_treatment_{'U5_appendix' if appendix else 'HH'}.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


print("MICS DDML: CHARACTERISTICS ASSOCIATED WITH WATER TREATMENT")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

hh_results = {}
u5_results = {}
all_results = []
hh_data, _ = pyreadstat.read_dta(DATA_DIR / "MASTER_MICS_FINAL.dta")
u5_data, _ = pyreadstat.read_dta(DATA_DIR / "MASTER_MICS_FINAL_U5.dta")
for treatment_code, treatment_label in TREATMENTS.items():
    hh_results[treatment_label] = _fit_sample(
        hh_data, "Households", False,
        treatment_code, treatment_label
    )
    u5_results[treatment_label] = _fit_sample(
        u5_data, "Under-five children", True,
        treatment_code, treatment_label
    )
    all_results.extend([hh_results[treatment_label][0], u5_results[treatment_label][0]])

all_results = pd.concat(all_results, ignore_index=True)
all_results.to_csv(OUTPUT_DIR / "results_characteristics_by_treatment.csv", index=False)
hh_path = _write_table(hh_results, "Households", appendix=False)
u5_path = _write_table(u5_results, "Under-five children", appendix=True)
print(f"  results -> {OUTPUT_DIR / 'results_characteristics_by_treatment.csv'}")
print(f"  tables  -> {hh_path}; {u5_path}")
