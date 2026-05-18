"""
ML learner creation (translated from learners.R).
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import (LinearRegression, LogisticRegressionCV, 
                                   LassoCV, RidgeCV, ElasticNetCV)
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, 
                               StackingRegressor, StackingClassifier)
import xgboost as xgb
from config import RANDOM_STATE, N_FOLDS

def create_learners(learner_type: str = "binary") -> Dict[str, Dict[str, Any]]:
    """
    Create learners for DoubleML.
    
    Args:
        learner_type: "binary" for IRM (binary treatment/outcome), 
                      "continuous" for PLR (continuous outcome).
    
    Returns:
        Dictionary with learner names as keys, each value a dict with 'g' and 'm' estimators.
    """
    learners: Dict[str, Dict[str, Any]] = {}
    
    # Common hyperparameters
    cv_folds = 5
    max_iter = 1000
    
    # Small grid of alphas for faster CV
    alphas = np.logspace(-3, -1, 10)  # 10 alphas from 0.001 to 0.1
    
    # OLS (linear regression / logistic regression)
    if learner_type == "binary":
        learners["ols"] = {
            "g": LinearRegression(),
            "m": LogisticRegressionCV(cv=cv_folds, random_state=RANDOM_STATE, max_iter=max_iter)
        }
    else:
        learners["ols"] = {
            "g": LinearRegression(),
            "m": LinearRegression()
        }
    
    # Lasso (alpha = 1)
    learners["lasso"] = {
        "g": LassoCV(
            alphas=alphas,
            cv=2,
            random_state=RANDOM_STATE,
            max_iter=max_iter,
            selection='random'
        ),
        "m": LogisticRegressionCV(
            Cs=1/alphas,  # inverse of alphas for logistic side
            cv=2,
            penalty='l1',
            solver='saga',
            random_state=RANDOM_STATE,
            max_iter=500
        )
    }
    
    # Ridge (alpha = 0)
    learners["ridge"] = {
        "g": RidgeCV(cv=cv_folds),
        "m": LogisticRegressionCV(
            cv=cv_folds, 
            penalty='l2', 
            solver='lbfgs', 
            random_state=RANDOM_STATE, 
            max_iter=max_iter
        )
    }
    
    # Elastic Net (alpha = 0.5)
    learners["enet"] = {
        "g": ElasticNetCV(l1_ratio=0.5, cv=cv_folds, random_state=RANDOM_STATE, max_iter=10000),
        "m": LogisticRegressionCV(
            cv=cv_folds, 
            penalty='l2', 
            solver='lbfgs', 
            random_state=RANDOM_STATE, 
            max_iter=max_iter
        )  # Approximation
    }
    
    # Random Forest
    learners["rf"] = {
        "g": RandomForestRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "m": RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }
    
    # XGBoost
    learners["xgb"] = {
        "g": xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            eval_metric='rmse',
            tree_method='hist',
            seed=RANDOM_STATE,
            n_jobs=-1
        ),
        "m": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',
            seed=RANDOM_STATE,
            n_jobs=-1
        )
    }
    
    return learners


def _base_learners_regressor() -> list:
    """Base learners for stacked ensemble (regression) - includes ALL learners."""
    ols = LinearRegression()
    lasso = LassoCV(
        alphas=np.logspace(-3, -1, 10),
        cv=2,
        random_state=RANDOM_STATE,
        max_iter=1000,
        selection='random'
    )
    ridge = RidgeCV(cv=5)
    enet = ElasticNetCV(l1_ratio=0.5, cv=5, random_state=RANDOM_STATE)
    rf = RandomForestRegressor(
        n_estimators=300, 
        random_state=RANDOM_STATE, 
        n_jobs=-1
    )
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        eval_metric='rmse',
        tree_method='hist',
        seed=RANDOM_STATE,
        n_jobs=-1
    )
    return [("ols", ols), ("lasso", lasso), ("ridge", ridge), ("enet", enet), ("rf", rf), ("xgb", xgb_model)]


def _base_learners_classifier() -> list:
    """Base learners for stacked ensemble (classification) - includes ALL learners."""
    ols = LogisticRegressionCV(cv=5, random_state=RANDOM_STATE, max_iter=1000)
    lasso = LogisticRegressionCV(
        Cs=np.logspace(-1, 3, 10),
        cv=2,
        penalty='l1',
        solver='saga',
        random_state=RANDOM_STATE,
        max_iter=500
    )
    ridge = LogisticRegressionCV(
        cv=5, 
        penalty='l2', 
        solver='lbfgs', 
        random_state=RANDOM_STATE, 
        max_iter=1000
    )
    enet = LogisticRegressionCV(cv=5, random_state=RANDOM_STATE, max_iter=1000)
    rf = RandomForestClassifier(
        n_estimators=300, 
        random_state=RANDOM_STATE, 
        n_jobs=-1
    )
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        seed=RANDOM_STATE,
        n_jobs=-1
    )
    return [("ols", ols), ("lasso", lasso), ("ridge", ridge), ("enet", enet), ("rf", rf), ("xgb", xgb_model)]


def create_stacked_ensemble() -> Tuple[StackingRegressor, StackingClassifier]:
    """
    Create stacked ensemble for regression (g) and classification (m).
    
    Returns:
        Tuple of (g_learner, m_learner).
    """
    # Regression stack
    base_learners_reg = _base_learners_regressor()
    g = StackingRegressor(
        estimators=base_learners_reg,
        final_estimator=RidgeCV(cv=5),
        cv=N_FOLDS,
        passthrough=False,
        n_jobs=-1
    )
    
    # Classification stack
    base_learners_clf = _base_learners_classifier()
    m = StackingClassifier(
        estimators=base_learners_clf,
        final_estimator=LogisticRegressionCV(cv=5, random_state=RANDOM_STATE),
        cv=N_FOLDS,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    return g, m