"""
Treatment predictor analysis: LASSO-logit and XGBoost variable importance.

Descriptive analysis of which covariates predict water treatment adoption.
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    HH_DATA_FILE, U5_DATA_FILE, BASE_CONFOUNDERS, U5_ADDITIONAL_CONFOUNDERS,
    RANDOM_STATE, N_JOBS, OUTPUT_DIR,
)
from data import prepare_hh_data, create_model_matrix

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score


def main():
    print("=" * 70)
    print("MICS DDML: TREATMENT PREDICTOR ANALYSIS")
    print("=" * 70)

    # Load HH data (treatment prediction uses household-level data)
    dt = prepare_hh_data(HH_DATA_FILE)

    # =========================================================================
    # Build confounder matrix
    # =========================================================================
    X, dt = create_model_matrix(dt, "SomeRiskHome")

    D = dt["water_treatment"].values.astype(float)

    # Complete cases
    mask = dt["water_treatment"].notna() & ~np.isnan(X).any(axis=1)
    X_clean = X[mask.values]
    D_clean = D[mask.values]

    print(f"\n  N observations: {len(D_clean):,}")
    print(f"  Treatment rate: {D_clean.mean():.3f}")
    print(f"  N confounders: {X_clean.shape[1]}")

    # =========================================================================
    # Part 1: LASSO-Logistic Regression
    # =========================================================================
    print("\n--- Part 1: LASSO-Logistic Regression ---")

    lasso_logit = LogisticRegressionCV(
        cv=5, penalty="l1", solver="saga", max_iter=5000,
        n_jobs=N_JOBS, random_state=RANDOM_STATE, scoring="roc_auc",
    )
    lasso_logit.fit(X_clean, D_clean)

    # AUC
    auc = roc_auc_score(D_clean, lasso_logit.predict_proba(X_clean)[:, 1])
    print(f"  In-sample AUC: {auc:.4f}")

    # Non-zero coefficients
    coef = lasso_logit.coef_.flatten()
    nonzero = np.abs(coef) > 1e-6
    print(f"  Non-zero coefficients: {nonzero.sum()}/{len(coef)}")

    # Average marginal effects (AME)
    n_bootstrap = 100
    ame_samples = np.zeros((n_bootstrap, len(coef)))
    n_obs = len(D_clean)

    for b in range(n_bootstrap):
        idx = np.random.RandomState(RANDOM_STATE + b).choice(n_obs, size=n_obs, replace=True)
        X_b = X_clean[idx]
        D_b = D_clean[idx]
        try:
            lr = LogisticRegressionCV(
                cv=5, penalty="l1", solver="saga", max_iter=5000,
                n_jobs=N_JOBS, random_state=RANDOM_STATE + b, scoring="roc_auc",
            )
            lr.fit(X_b, D_b)
            prob = lr.predict_proba(X_b)[:, 1]
            # AME = coef * prob * (1 - prob)
            ame = lr.coef_.flatten() * np.mean(prob * (1 - prob))
            ame_samples[b] = ame
        except Exception:
            ame_samples[b] = np.nan

    ame_mean = np.nanmean(ame_samples, axis=0)
    ame_se = np.nanstd(ame_samples, axis=0)

    # =========================================================================
    # Part 2: XGBoost Variable Importance
    # =========================================================================
    print("\n--- Part 2: XGBoost Variable Importance ---")

    xgb = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=RANDOM_STATE,
    )
    xgb.fit(X_clean, D_clean)

    # Cross-validated AUC
    cv_auc = cross_val_score(xgb, X_clean, D_clean, cv=5, scoring="roc_auc", n_jobs=N_JOBS)
    print(f"  5-fold CV AUC: {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")

    # Feature importance (gain)
    importance = xgb.feature_importances_

    # =========================================================================
    # Export results
    # =========================================================================
    results_dir = OUTPUT_DIR / "treatment_predictors"
    results_dir.mkdir(parents=True, exist_ok=True)

    # LASSO results
    lasso_df = pd.DataFrame({
        "coefficient": coef,
        "nonzero": nonzero,
        "ame_mean": ame_mean,
        "ame_se": ame_se,
    })
    lasso_df.to_csv(results_dir / "lasso_treatment.csv", index=False)

    # XGBoost results
    xgb_df = pd.DataFrame({
        "importance": importance,
    })
    xgb_df.to_csv(results_dir / "xgb_importance.csv", index=False)

    print(f"\nResults saved to: {results_dir}")
    print("=" * 70)
    print("TREATMENT PREDICTOR ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()