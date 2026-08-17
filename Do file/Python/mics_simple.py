"""Simple, readable MICS DoubleML analysis.

This script is intentionally written like a Stata do-file:

    1. Set the analysis choices.
    2. Load and prepare the data.
    3. Estimate the binary treatment effect.
    4. Estimate treatment-method effects versus no treatment.
    5. Save one small results table.

The technical details of cross-fitting and stacking are kept inside two
small functions so the main analysis remains easy to follow.
"""

from pathlib import Path

import doubleml as dml
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    RidgeCV,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


# ============================================================
# 1. Analysis choices
# ============================================================

SEED = 42
FOLDS = 5
REPETITIONS = 3

# Use three treatment methods, as in the main Python analysis.
TREATMENT_METHODS = (1, 2, 3)

# The script is stored in MICS_DDML/Do file/Python.
PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "Data" / "3. Final"
# Keep outputs next to this script so the file works even when the project
# Output folder is read-only or is mounted from another operating system.
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


# ============================================================
# 2. Variables used in the analysis
# ============================================================

HOUSEHOLD_FILE = DATA_DIR / "MASTER_MICS_FINAL.dta"
CHILD_FILE = DATA_DIR / "MASTER_MICS_FINAL_U5.dta"

HOUSEHOLD_OUTCOMES = ("SomeRiskHome", "VeryHighRiskHome")
CHILD_OUTCOME = "diarrhea"

# These are the controls used by mics.py.
COMMON_CONTROLS = [
    "windex5",
    "urban",
    "WS1_g",
    "wq27_decile",
    "Any_U5",
    "Girls_less_than15",
    "Boys_15or_less",
    "Toilet",
]


# ============================================================
# 3. Machine-learning learners
# ============================================================

def make_learners():
    """Create one readable stacked learner for outcomes and treatment."""

    # These are the five outcome learners used by mics.py:
    # OLS, LASSO, elastic net, random forest, and XGBoost.
    outcome_learners = [
        ("ols", LinearRegression()),
        ("lasso", make_pipeline(StandardScaler(), LassoCV(cv=3, max_iter=3000))),
        ("elastic_net", make_pipeline(
            StandardScaler(),
            ElasticNetCV(
                l1_ratio=(0.25, 0.5, 0.75),
                cv=3,
                max_iter=3000,
                random_state=SEED,
            ),
        )),
        ("forest", RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=-1,
        )),
        ("xgboost", XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=SEED,
            n_jobs=-1,
            objective="reg:squarederror",
            eval_metric="rmse",
            verbosity=0,
        )),
    ]

    # These are the five treatment learners used by mics.py:
    # logit, LASSO, elastic net, random forest, and XGBoost.
    treatment_learners = [
        ("logit", LogisticRegression(max_iter=1000)),
        ("lasso", make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(
                Cs=10,
                cv=3,
                penalty="l1",
                solver="liblinear",
                max_iter=1000,
            ),
        )),
        ("elastic_net", make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(
                Cs=10,
                cv=3,
                penalty="elasticnet",
                solver="saga",
                l1_ratios=(0.5,),
                max_iter=2000,
                scoring="neg_log_loss",
            ),
        )),
        ("forest", RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=-1,
        )),
        ("xgboost", XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=SEED,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        )),
    ]

    outcome_model = StackingRegressor(
        estimators=outcome_learners,
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 7)),
        cv=3,
        n_jobs=-1,
    )
    treatment_model = StackingClassifier(
        estimators=treatment_learners,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
        n_jobs=-1,
    )
    return outcome_model, treatment_model


# ============================================================
# 4. Data preparation
# ============================================================

def prepare_data(data, outcome, treatment, child=False, allowed_treatments=None):
    """Return a clean DoubleML frame and a numeric design matrix."""

    controls = list(COMMON_CONTROLS)
    if child:
        controls += ["age", "male"]

    required = [outcome, treatment, "Cluster_var", *controls]
    frame = data[required].copy()
    frame = frame.dropna()

    if allowed_treatments is not None:
        frame = frame[frame[treatment].isin(allowed_treatments)].copy()

    # Convert categorical controls to the same dummy-variable representation
    # used by mics.py. drop_first=True avoids perfect collinearity.
    categorical = [
        "windex5",
        "WS1_g",
        "wq27_decile",
        "Toilet",
    ]
    if "country_cat" in data.columns:
        frame["country_cat"] = data.loc[frame.index, "country_cat"]
        frame = frame.dropna(subset=["country_cat"])
        categorical.append("country_cat")

    x_variables = [v for v in controls if v not in categorical] + categorical
    x = pd.get_dummies(frame[x_variables], drop_first=True, dtype=float)
    x = x.astype(float)
    x.index = frame.index

    # DoubleML needs the outcome, treatment, controls, and cluster variable
    # in one data frame. The cluster variable is metadata, not a predictor.
    model_frame = pd.concat(
        [frame[[outcome, treatment, "Cluster_var"]], x], axis=1
    ).reset_index(drop=True)
    return model_frame, list(x.columns)


def estimate_irm(data, outcome, treatment, label, child=False, allowed_treatments=None):
    """Estimate one binary-treatment IRM model and return its summary row."""

    frame, x_columns = prepare_data(
        data,
        outcome=outcome,
        treatment=treatment,
        child=child,
        allowed_treatments=allowed_treatments,
    )

    dml_data = dml.DoubleMLData(
        frame,
        y_col=outcome,
        d_cols=treatment,
        x_cols=x_columns,
        cluster_cols="Cluster_var",
    )
    ml_g, ml_m = make_learners()
    model = dml.DoubleMLIRM(
        dml_data,
        ml_g,
        ml_m,
        n_folds=FOLDS,
        n_rep=REPETITIONS,
        draw_sample_splitting=True,
    )
    model.fit(n_jobs_cv=1, store_predictions=False)

    result = model.summary.reset_index()
    result.insert(0, "model", label)
    result.insert(1, "outcome", outcome)
    result.insert(2, "treatment", treatment)
    result.insert(3, "n", len(frame))
    result.insert(4, "clusters", frame["Cluster_var"].nunique())
    return result


# ============================================================
# 5. Main analysis
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

print("Loading data...")
household = pd.read_stata(HOUSEHOLD_FILE)
children = pd.read_stata(CHILD_FILE)
results = []

# Household outcomes: binary water-treatment indicator.
for outcome in HOUSEHOLD_OUTCOMES:
    print(f"Estimating household IRM: {outcome}")
    results.append(estimate_irm(
        household,
        outcome=outcome,
        treatment="water_treatment",
        label=f"HH IRM: {outcome}",
        allowed_treatments=(0, 1),
    ))

# Under-five outcome: binary water-treatment indicator.
print(f"Estimating child IRM: {CHILD_OUTCOME}")
results.append(estimate_irm(
    children,
    outcome=CHILD_OUTCOME,
    treatment="water_treatment",
    label="U5 IRM: diarrhea",
    child=True,
    allowed_treatments=(0, 1),
))

# Treatment methods versus no treatment.
for level in TREATMENT_METHODS:
    for outcome in HOUSEHOLD_OUTCOMES:
        print(f"Estimating HH treatment method {level}: {outcome}")
        data = household.copy()
        data["binary_method"] = np.where(
            data["WQ15_g"].isin((0, level)),
            (data["WQ15_g"] == level).astype(float),
            np.nan,
        )
        results.append(estimate_irm(
            data,
            outcome=outcome,
            treatment="binary_method",
            label=f"HH method {level}: {outcome}",
            allowed_treatments=(0, 1),
        ))

    print(f"Estimating U5 treatment method {level}: {CHILD_OUTCOME}")
    data = children.copy()
    data["binary_method"] = np.where(
        data["WQ15_g"].isin((0, level)),
        (data["WQ15_g"] == level).astype(float),
        np.nan,
    )
    results.append(estimate_irm(
        data,
        outcome=CHILD_OUTCOME,
        treatment="binary_method",
        label=f"U5 method {level}: diarrhea",
        child=True,
        allowed_treatments=(0, 1),
    ))

results = pd.concat(results, ignore_index=True)
results.to_csv(OUTPUT_DIR / "main_results.csv", index=False)
results.to_latex(OUTPUT_DIR / "main_results.tex", index=False)
print(f"Saved results to {OUTPUT_DIR / 'main_results.csv'}")
