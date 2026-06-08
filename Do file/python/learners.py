"""
ML learner definitions for the MICS DDML Analysis.

Provides 7 learners (OLS, Lasso, Ridge, ENet, RF, XGBoost, Stacked)
for both binary outcome and binary treatment nuisance functions.

Compatible with sklearn >= 1.8 (uses l1_ratios instead of deprecated penalty parameter).
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import numpy as np
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, LogisticRegressionCV,
    LassoCV, RidgeCV, ElasticNetCV,
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    StackingRegressor, StackingClassifier,
)
from sklearn.base import BaseEstimator, RegressorMixin, clone
from xgboost import XGBRegressor, XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE, N_JOBS

_CS_FAST = np.logspace(-3, 3, 7)

# Inside StackingRegressor/StackingClassifier, scikit-learn already
# parallelizes across folds and base learners.  Setting base-learner
# n_jobs=1 avoids fork-bombing the machine when the outer ensemble
# also uses n_jobs>1.
N_JOBS_BASE = 1


class ProbaRegressor(RegressorMixin, BaseEstimator):
    """Wrap a classifier so .predict() returns predict_proba[:, 1].

    This makes classifiers compatible with ml_g in DoubleML, ensuring that
    predicted nuisance values for binary outcomes stay bounded in [0, 1].

    Note: RegressorMixin must precede BaseEstimator in the MRO so that the
    ``regressor`` estimator-type tag is set under scikit-learn >= 1.6; otherwise
    StackingRegressor rejects it with "should be a regressor".
    """

    def __init__(self, classifier=None):
        self.classifier = classifier

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        return self.classifier.predict_proba(X)[:, 1]


def _lasso_regressor(n_jobs=N_JOBS):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(cv=3, n_jobs=n_jobs, random_state=RANDOM_STATE, max_iter=5000)),
    ])


def _ridge_regressor():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(cv=3)),
    ])


def _rf_regressor(n_jobs=N_JOBS):
    return RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=n_jobs,
    )


def _lasso_classifier(n_jobs=N_JOBS):
    return LogisticRegressionCV(
        cv=3, Cs=_CS_FAST, solver="liblinear", max_iter=500,
        tol=1e-2, n_jobs=n_jobs, random_state=RANDOM_STATE, scoring="roc_auc",
        l1_ratios=(1,),
    )


def _ridge_classifier(n_jobs=N_JOBS):
    return LogisticRegressionCV(
        cv=3, Cs=_CS_FAST, solver="lbfgs", max_iter=1000,
        tol=1e-2, n_jobs=n_jobs, random_state=RANDOM_STATE, scoring="roc_auc",
        l1_ratios=(0,),
    )


def _rf_classifier(n_jobs=N_JOBS):
    return RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=n_jobs,
    )


def create_learners() -> dict:
    """
    Create all ML learners for DoubleML IRM estimation.

    Returns
    -------
    dict mapping learner name to {'g': outcome_model, 'm': treatment_model}.
    """
    learners = {}

    learners["ols"] = {
        "g": ProbaRegressor(
            LogisticRegressionCV(
                cv=3, Cs=_CS_FAST, solver="lbfgs", max_iter=1000,
                tol=1e-2, n_jobs=N_JOBS, random_state=RANDOM_STATE, scoring="roc_auc",
                l1_ratios=(0,),
            )
        ),
        "m": LogisticRegressionCV(
            cv=3, Cs=_CS_FAST, solver="lbfgs", max_iter=1000,
            tol=1e-2, n_jobs=N_JOBS, random_state=RANDOM_STATE, scoring="roc_auc",
            l1_ratios=(0,),
        ),
    }

    learners["lasso"] = {
        "g": _lasso_regressor(n_jobs=N_JOBS),
        "m": _lasso_classifier(n_jobs=N_JOBS),
    }

    learners["ridge"] = {
        "g": _ridge_regressor(),
        "m": _ridge_classifier(n_jobs=N_JOBS),
    }

    learners["enet"] = {
        "g": Pipeline([
            ("scaler", StandardScaler()),
            ("enet", ElasticNetCV(cv=3, l1_ratio=0.5,
                                    n_jobs=N_JOBS, random_state=RANDOM_STATE, max_iter=3000)),
        ]),
        "m": LogisticRegressionCV(
            cv=3, Cs=_CS_FAST, solver="saga", max_iter=2000,
            tol=1e-2, n_jobs=N_JOBS, random_state=RANDOM_STATE, scoring="roc_auc",
            l1_ratios=[0.2, 0.5, 0.8],
        ),
    }

    learners["rf"] = {
        "g": _rf_regressor(n_jobs=N_JOBS),
        "m": _rf_classifier(n_jobs=N_JOBS),
    }

    learners["xgb"] = {
        "g": XGBRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=RANDOM_STATE, n_jobs=N_JOBS,
            eval_metric="rmse", verbosity=0,
        ),
        "m": XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=RANDOM_STATE, n_jobs=N_JOBS,
            eval_metric="logloss", verbosity=0, use_label_encoder=False,
        ),
    }

    learners["stacked"] = _create_stacked_ensemble()

    return learners


def create_learners_for_binary() -> dict:
    """
    Create learners where ml_g uses classifiers predicting probabilities.

    For binary outcomes, classifiers return predictions bounded in [0,1],
    which improves inverse-propensity weight stability compared to LPM.

    Returns
    -------
    Same dict structure as create_learners, but 'g' entries are wrapped
    in ProbaRegressor when the underlying model is a classifier.
    """
    learners = {}

    learners["ols"] = {
        "g": ProbaRegressor(
            LogisticRegressionCV(
                cv=3, Cs=_CS_FAST, solver="lbfgs", max_iter=1000,
                tol=1e-2, n_jobs=N_JOBS, random_state=RANDOM_STATE, scoring="roc_auc",
                l1_ratios=(0,),
            )
        ),
        "m": LogisticRegressionCV(
            cv=3, Cs=_CS_FAST, solver="lbfgs", max_iter=1000,
            tol=1e-2, n_jobs=N_JOBS, random_state=RANDOM_STATE, scoring="roc_auc",
            l1_ratios=(0,),
        ),
    }

    learners["lasso"] = {
        "g": ProbaRegressor(
            LogisticRegressionCV(
                cv=3, Cs=_CS_FAST, solver="liblinear", max_iter=500,
                tol=1e-2, n_jobs=N_JOBS, random_state=RANDOM_STATE, scoring="roc_auc",
                l1_ratios=(1,),
                penalty="l1",
            )
        ),
        "m": _lasso_classifier(n_jobs=N_JOBS),
    }

    learners["ridge"] = {
        "g": ProbaRegressor(
            LogisticRegressionCV(
                cv=3, Cs=_CS_FAST, solver="lbfgs", max_iter=1000,
                tol=1e-2, n_jobs=N_JOBS, random_state=RANDOM_STATE, scoring="roc_auc",
                l1_ratios=(0,),
            )
        ),
        "m": _ridge_classifier(n_jobs=N_JOBS),
    }

    learners["enet"] = {
        "g": ProbaRegressor(
            LogisticRegressionCV(
                cv=3, Cs=_CS_FAST, solver="saga", max_iter=2000,
                tol=1e-2, n_jobs=N_JOBS, random_state=RANDOM_STATE, scoring="roc_auc",
                l1_ratios=[0.2, 0.5, 0.8],
                penalty="elasticnet",
            )
        ),
        "m": LogisticRegressionCV(
            cv=3, Cs=_CS_FAST, solver="saga", max_iter=2000,
            tol=1e-2, n_jobs=N_JOBS, random_state=RANDOM_STATE, scoring="roc_auc",
            l1_ratios=[0.2, 0.5, 0.8],
        ),
    }

    learners["rf"] = {
        "g": ProbaRegressor(
            RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_leaf=5,
                random_state=RANDOM_STATE, n_jobs=N_JOBS,
            )
        ),
        "m": _rf_classifier(n_jobs=N_JOBS),
    }

    learners["xgb"] = {
        "g": ProbaRegressor(
            XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=RANDOM_STATE, n_jobs=N_JOBS,
                eval_metric="logloss", verbosity=0, use_label_encoder=False,
            )
        ),
        "m": XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=RANDOM_STATE, n_jobs=N_JOBS,
            eval_metric="logloss", verbosity=0, use_label_encoder=False,
        ),
    }

    learners["stacked"] = _create_stacked_ensemble_binary()

    return learners


def _base_learners_regressor() -> list:
    # Use N_JOBS_BASE=1 because StackingRegressor already parallelizes
    # across folds and base learners via its own n_jobs parameter.
    return [
        ("lasso", _lasso_regressor(n_jobs=N_JOBS_BASE)),
        ("ridge", _ridge_regressor()),
        ("rf", _rf_regressor(n_jobs=N_JOBS_BASE)),
    ]


def _base_learners_classifier() -> list:
    return [
        ("lasso", _lasso_classifier(n_jobs=N_JOBS_BASE)),
        ("ridge", _ridge_classifier(n_jobs=N_JOBS_BASE)),
        ("rf", _rf_classifier(n_jobs=N_JOBS_BASE)),
    ]


def _create_stacked_ensemble() -> dict:
    # Meta-learner: RidgeCV with a reduced alpha grid and fewer CV folds.
    # The meta-learner only needs to combine 3 base learners; extreme
    # fine-grained CV adds little accuracy but 3-4x runtime.  We cut the
    # grid to 5 alphas and CV=3, keeping stability without the cost.
    _RIDGE_ALPHAS = np.logspace(-3, 3, 5)

    final_g = RidgeCV(alphas=_RIDGE_ALPHAS, cv=3)
    final_m = LogisticRegressionCV(
        cv=3, Cs=_CS_FAST[:5], solver="lbfgs", max_iter=1000,
        tol=1e-2, n_jobs=N_JOBS, random_state=RANDOM_STATE, scoring="roc_auc",
        l1_ratios=(0,),
    )
    stacked_g = StackingRegressor(
        estimators=_base_learners_regressor(),
        final_estimator=final_g,
        cv=3, n_jobs=N_JOBS,
    )
    stacked_m = StackingClassifier(
        estimators=_base_learners_classifier(),
        final_estimator=final_m,
        cv=3, n_jobs=N_JOBS,
    )
    return {"g": stacked_g, "m": stacked_m}


def _create_stacked_ensemble_binary() -> dict:
    """Stacked ensemble for binary outcomes: classifiers wrapped in ProbaRegressor.

    Base learners for 'g' are classifiers wrapped to return probabilities.
    Meta-learner for 'g' is RidgeCV (on probs) to keep bounded outputs.
    For 'm', standard classifiers with LogisticRegressionCV meta.
    """
    _RIDGE_ALPHAS = np.logspace(-3, 6, 10)

    final_g = RidgeCV(alphas=_RIDGE_ALPHAS, cv=5)
    final_m = LogisticRegressionCV(
        cv=5, Cs=_CS_FAST, solver="lbfgs", max_iter=1000,
        tol=1e-2, n_jobs=N_JOBS, random_state=RANDOM_STATE, scoring="roc_auc",
        l1_ratios=(0,),
    )
    stacked_g = StackingRegressor(
        estimators=[
            ("lasso", ProbaRegressor(_lasso_classifier(n_jobs=N_JOBS_BASE))),
            ("ridge", ProbaRegressor(_ridge_classifier(n_jobs=N_JOBS_BASE))),
            ("rf", ProbaRegressor(_rf_classifier(n_jobs=N_JOBS_BASE))),
        ],
        final_estimator=final_g,
        cv=3, n_jobs=N_JOBS,
    )
    stacked_m = StackingClassifier(
        estimators=_base_learners_classifier(),
        final_estimator=final_m,
        cv=3, n_jobs=N_JOBS,
    )
    return {"g": stacked_g, "m": stacked_m}


def get_stacking_weights(stacked_model, X=None, y=None) -> dict:
    """Extract stacking weights from a fitted StackingRegressor/StackingClassifier.

    For Pipeline meta-learners, navigates through the pipeline to find coef_.
    Returns normalized (sum-to-1) absolute weight for each base learner.
    """
    try:
        final = stacked_model.final_estimator_
        # If the final estimator is a Pipeline, extract the last step's coef_
        if hasattr(final, "named_steps"):
            last_step = list(final.named_steps.values())[-1]
            coef = last_step.coef_ if hasattr(last_step, "coef_") else None
        elif hasattr(final, "coef_"):
            coef = final.coef_
        else:
            return {}
        if coef is None:
            return {}
        weights = np.abs(coef).flatten()
        if weights.sum() > 0:
            weights = weights / weights.sum()
        base_names = [name for name, _ in stacked_model.estimators]
        if len(base_names) == len(weights):
            return dict(zip(base_names, weights.round(3)))
    except Exception:
        pass
    return {}
