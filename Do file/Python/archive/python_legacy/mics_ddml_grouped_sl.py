from __future__ import annotations

import os
import pickle
from pathlib import Path

import doubleml as dml
import numpy as np
import pandas as pd
from joblib import hash as joblib_hash
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    RidgeCV,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

LEVELS = {
    0: "No treatment",
    1: "Boiling",
    2: "Chlorination/tablets",
    3: "Straining/settling",
}


def selected(series: pd.Series, letter: str) -> pd.Series:
    return series.astype("string").str.strip().str.upper().eq(letter.upper())


def factorize_key(frame: pd.DataFrame) -> np.ndarray:
    clean = frame.astype("string").fillna("<missing>")
    return pd.factorize(pd.MultiIndex.from_frame(clean), sort=True)[0].astype("int64")


def first_available(df: pd.DataFrame, names: list[str]) -> tuple[pd.Series, list[str]]:
    result = pd.Series(pd.NA, index=df.index, dtype="object")
    found: list[str] = []
    for name in names:
        if name in df.columns:
            found.append(name)
            result = result.where(result.notna(), df[name])
    if not found:
        raise KeyError(f"None of the required identifier columns exists: {names}")
    return result, found


def add_design_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    country = df["country_cat"].astype("string").fillna("<missing-country>")

    psu_raw, psu_sources = first_available(df, ["PSU", "psu", "Cluster_var", "HH1"])
    psu_country_count = (
        pd.DataFrame({"psu": psu_raw.astype("string"), "country": country})
        .dropna()
        .groupby("psu")["country"]
        .nunique()
    )
    psu_reused = bool((psu_country_count > 1).any())
    psu_key = (
        pd.DataFrame({"country": country, "psu": psu_raw})
        if psu_reused
        else pd.DataFrame({"psu": psu_raw})
    )
    df["_psu_id"] = factorize_key(psu_key)

    if "HHID" in df.columns:
        hh_raw = df["HHID"]
        hh_sources = ["HHID"]
    elif {"HH1", "HH2"}.issubset(df.columns):
        hh_raw = df["HH1"].astype("string") + "|" + df["HH2"].astype("string")
        hh_sources = ["HH1", "HH2"]
    else:
        hh_raw, hh_sources = first_available(df, ["HH2", "HH1"])

    hh_country_count = (
        pd.DataFrame({"hh": hh_raw.astype("string"), "country": country})
        .dropna()
        .groupby("hh")["country"]
        .nunique()
    )
    hh_reused = bool((hh_country_count > 1).any())
    hh_key = (
        pd.DataFrame({"country": country, "hh": hh_raw})
        if hh_reused
        else pd.DataFrame({"hh": hh_raw})
    )
    df["_hh_id"] = factorize_key(hh_key)

    return df, {
        "psu_source_columns": ", ".join(psu_sources),
        "psu_reused_across_countries": psu_reused,
        "household_source_columns": ", ".join(hh_sources),
        "household_reused_across_countries": hh_reused,
    }


def create_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    risk = pd.to_numeric(df["RiskHome"], errors="coerce")
    missing = risk.isna()
    df["SomeRiskHome"] = risk.isin([1, 2]).mask(missing).astype("Int8")
    df["VeryHighRiskHome"] = risk.eq(2).mask(missing).astype("Int8")

    violation = df["VeryHighRiskHome"].eq(1) & ~df["SomeRiskHome"].eq(1)
    assert not violation.any()
    assert df["SomeRiskHome"].isna().equals(df["VeryHighRiskHome"].isna())
    return df


def create_treatments(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    method_cols = [
        "WQ15A", "WQ15B", "WQ15C", "WQ15D", "WQ15E",
        "WQ15F", "WQ15G", "WQ15H", "WQ15X",
    ]
    missing = sorted(set(method_cols) - set(df.columns))
    if missing:
        raise KeyError(f"Missing water-treatment indicators: {missing}")

    used = pd.DataFrame(
        {column: selected(df[column], column[-1]) for column in method_cols},
        index=df.index,
    )
    boil = used["WQ15A"]
    chlorination = used["WQ15B"] | used["WQ15G"] | used["WQ15H"]
    straining_settling = used["WQ15C"] | used["WQ15D"] | used["WQ15F"]
    solar = used["WQ15E"]
    other = used["WQ15X"]

    recognized_any = boil | chlorination | straining_settling | solar
    other_only = other & ~recognized_any
    no_treatment = ~used.any(axis=1)

    category = pd.Series(pd.NA, index=df.index, dtype="Int8")
    category.loc[no_treatment] = 0
    category.loc[straining_settling] = 3
    category.loc[chlorination] = 2
    category.loc[boil] = 1

    df["treatment_count"] = used.sum(axis=1).astype("int16")
    df["multiple_methods"] = df["treatment_count"].gt(1)
    df["other_only"] = other_only
    df["solar_treatment"] = solar
    df["water_treatment"] = recognized_any.astype("int8")
    df["treat_cat"] = category
    df["treat_boil"] = category.eq(1).fillna(False).astype("int8")
    df["treat_chlorination_tablets"] = category.eq(2).fillna(False).astype("int8")
    df["treat_straining_settling"] = category.eq(3).fillna(False).astype("int8")

    diagnostics = {
        "rows_before_other_drop": len(df),
        "multiple_method_rows": int(df["multiple_methods"].sum()),
        "multiple_method_share": float(df["multiple_methods"].mean()),
        "other_only_rows_dropped": int(other_only.sum()),
        "solar_rows": int(solar.sum()),
    }

    df = df.loc[~other_only].copy()
    assert not df["other_only"].any()
    assert set(df["treat_cat"].dropna().unique()).issubset(set(LEVELS))
    return df, diagnostics


def prepare_sample(raw: pd.DataFrame, child: bool = False) -> tuple[pd.DataFrame, dict]:
    required = [
        "country_cat", "RiskHome", "windex5", "urban", "WS1_g", "wq27_decile",
        "Any_U5", "Girls_less_than15", "Boys_15or_less", "Toilet", "HHCHILDREN",
    ]
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = raw.copy()
    df["_row_id"] = np.arange(len(df), dtype="int64")
    df = create_outcomes(df)
    df, treatment_diagnostics = create_treatments(df)
    df, id_diagnostics = add_design_ids(df)

    for column in ["Any_U5", "Girls_less_than15", "Boys_15or_less"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype("int8")

    if child:
        needed = ["age", "male", "diarrhea"]
        missing = sorted(set(needed) - set(df.columns))
        if missing:
            raise KeyError(f"Missing U5 columns: {missing}")
        df["child_age"] = pd.to_numeric(df["age"], errors="coerce")
        df["child_sex_male"] = pd.to_numeric(df["male"], errors="coerce")
        df["diarrhea"] = pd.to_numeric(df["diarrhea"], errors="coerce").astype("Int8")

    return df, {**treatment_diagnostics, **id_diagnostics}


def dummy_block(series: pd.Series, prefix: str, drop_first: bool = True) -> pd.DataFrame:
    clean = series.astype("string").fillna("Missing")
    return pd.get_dummies(clean, prefix=prefix, drop_first=drop_first, dtype=float)


def build_controls(df: pd.DataFrame, child: bool = False) -> pd.DataFrame:
    blocks = [
        dummy_block(df["windex5"], "wealth"),
        dummy_block(df["country_cat"], "country"),
        dummy_block(df["urban"], "urban"),
        dummy_block(df["WS1_g"], "water_source"),
        dummy_block(df["Toilet"], "toilet"),
        dummy_block(df["wq27_decile"], "source_ecoli"),
        df[["Any_U5", "Girls_less_than15", "Boys_15or_less"]].astype(float),
    ]
    if child:
        blocks.extend([
            dummy_block(df["child_age"], "child_age"),
            df[["child_sex_male"]].astype(float).fillna(0),
        ])
    X = pd.concat(blocks, axis=1)
    X = X.loc[:, ~X.columns.duplicated()].astype(float)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)


def analysis_frame(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    child: bool = False,
    allowed_levels: list[int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    keep = df[outcome].notna() & df[treatment].notna()
    if allowed_levels is not None:
        keep &= df[treatment].isin(allowed_levels)

    sample = df.loc[keep].copy()
    X = build_controls(sample, child=child)
    frame = pd.concat([
        sample[["_row_id", "country_cat", "_psu_id", "_hh_id", outcome, treatment]],
        X,
    ], axis=1)
    frame[outcome] = pd.to_numeric(frame[outcome], errors="raise").astype(float)
    frame[treatment] = pd.to_numeric(frame[treatment], errors="raise").astype(int)
    frame["_psu_model_code"] = pd.factorize(frame["_psu_id"], sort=True)[0].astype(float)
    x_cols = list(X.columns) + ["_psu_model_code"]
    return frame.reset_index(drop=True), x_cols


def psu_structure(df: pd.DataFrame, sample_name: str) -> dict:
    psu_sizes = df.groupby("_psu_id").size()
    hh_sizes = df.groupby("_hh_id").size()
    return {
        "sample": sample_name,
        "observations": len(df),
        "PSUs": int(df["_psu_id"].nunique()),
        "households": int(df["_hh_id"].nunique()),
        "mean_observations_per_PSU": psu_sizes.mean(),
        "median_observations_per_PSU": psu_sizes.median(),
        "p90_observations_per_PSU": psu_sizes.quantile(0.90),
        "max_observations_per_PSU": psu_sizes.max(),
        "mean_observations_per_household": hh_sizes.mean(),
        "median_observations_per_household": hh_sizes.median(),
    }


def support_by_country(df: pd.DataFrame, sample_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = df.loc[df["treat_cat"].isin(LEVELS)].copy()
    counts = (
        sample.groupby(["country_cat", "treat_cat"], dropna=False)
        .agg(
            observations=("_row_id", "size"),
            PSUs=("_psu_id", "nunique"),
            households=("_hh_id", "nunique"),
        )
        .reset_index()
    )

    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    countries = sorted(sample["country_cat"].dropna().unique())

    for level in [1, 2, 3]:
        level_rows: list[dict] = []
        for country in countries:
            table = counts.loc[counts["country_cat"].eq(country)].set_index("treat_cat")
            n0 = int(table.loc[0, "observations"]) if 0 in table.index else 0
            g0 = int(table.loc[0, "PSUs"]) if 0 in table.index else 0
            nd = int(table.loc[level, "observations"]) if level in table.index else 0
            gd = int(table.loc[level, "PSUs"]) if level in table.index else 0
            row = {
                "sample": sample_name,
                "country": country,
                "treatment_code": level,
                "treatment_category": LEVELS[level],
                "N_no_treatment": n0,
                "PSU_no_treatment": g0,
                "N_category": nd,
                "PSU_category": gd,
                "both_categories_present": n0 > 0 and nd > 0,
                "at_least_2_PSU_each": g0 >= 2 and gd >= 2,
            }
            detail_rows.append(row)
            level_rows.append(row)

        level_frame = pd.DataFrame(level_rows)
        level_sample = sample.loc[sample["treat_cat"].eq(level)]
        summary_rows.append({
            "sample": sample_name,
            "treatment_category": LEVELS[level],
            "observations": len(level_sample),
            "PSUs": level_sample["_psu_id"].nunique(),
            "countries_with_category": level_sample["country_cat"].nunique(),
            "countries_with_category_and_no_treatment": level_frame["both_categories_present"].sum(),
            "countries_with_at_least_2_PSU_each": level_frame["at_least_2_PSU_each"].sum(),
        })

    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def grouped_sample_splitting(
    frame: pd.DataFrame,
    treatment: str,
    n_rep: int,
    n_folds: int,
    seed: int,
) -> tuple[list, list, pd.DataFrame]:
    y = frame[treatment].to_numpy(dtype=int)
    groups = frame["_psu_id"].to_numpy(dtype=int)
    households = frame["_hh_id"].to_numpy(dtype=int)
    required_levels = set(np.unique(y))

    all_smpls: list = []
    all_smpls_cluster: list = []
    audits: list[dict] = []

    for repetition in range(n_rep):
        selected = None
        for attempt in range(100):
            splitter = StratifiedGroupKFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=seed + repetition * 1000 + attempt,
            )
            candidate = list(splitter.split(np.zeros(len(frame)), y, groups))
            if all(set(np.unique(y[train])) == required_levels for train, _ in candidate):
                selected = candidate
                break
        if selected is None:
            raise RuntimeError("Unable to construct valid PSU-grouped folds.")

        seen = np.zeros(len(frame), dtype=int)
        cluster_rep: list = []
        for fold, (train, test) in enumerate(selected):
            seen[test] += 1
            train_psu = np.unique(groups[train])
            test_psu = np.unique(groups[test])
            train_hh = np.unique(households[train])
            test_hh = np.unique(households[test])
            assert np.intersect1d(train_psu, test_psu).size == 0
            assert np.intersect1d(train_hh, test_hh).size == 0

            row = {
                "repetition": repetition,
                "fold": fold,
                "train_N": len(train),
                "test_N": len(test),
                "train_PSU": len(train_psu),
                "test_PSU": len(test_psu),
                "train_households": len(train_hh),
                "test_households": len(test_hh),
            }
            for level in sorted(required_levels):
                row[f"test_share_{level}"] = float(np.mean(y[test] == level))
                row[f"train_N_{level}"] = int(np.sum(y[train] == level))
            audits.append(row)
            cluster_rep.append(([train_psu], [test_psu]))

        assert np.all(seen == 1)
        all_smpls.append(selected)
        all_smpls_cluster.append(cluster_rep)

    return all_smpls, all_smpls_cluster, pd.DataFrame(audits)


def feature_and_groups(X, group_column: int) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(X, dtype=float)
    index = group_column if group_column >= 0 else array.shape[1] + group_column
    groups = array[:, index].astype("int64")
    features = np.delete(array, index, axis=1)
    return features, groups


def inner_splits(y, groups, n_splits: int, seed: int, classification: bool):
    max_splits = min(int(n_splits), len(np.unique(groups)))
    for k in range(max_splits, 1, -1):
        if classification:
            splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            candidate = list(splitter.split(np.zeros(len(y)), y, groups))
            valid = all(np.unique(y[train]).size == 2 for train, _ in candidate)
        else:
            splitter = GroupKFold(n_splits=k)
            candidate = list(splitter.split(np.zeros(len(y)), y, groups))
            valid = True
        if valid:
            return candidate
    raise ValueError("Unable to construct grouped inner folds.")


def positive_probability(model, X) -> np.ndarray:
    classes = np.asarray(model.classes_)
    location = int(np.where(classes == 1)[0][0])
    return model.predict_proba(X)[:, location]


def convex_weights(predictions: np.ndarray, y: np.ndarray, classification: bool) -> np.ndarray:
    n_learners = predictions.shape[1]
    initial = np.repeat(1 / n_learners, n_learners)

    if classification:
        def objective(weights):
            probability = np.clip(predictions @ weights, 1e-6, 1 - 1e-6)
            return -np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))
    else:
        def objective(weights):
            return np.mean((y - predictions @ weights) ** 2)

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0, 1)] * n_learners,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1},
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if result.success:
        weights = np.clip(result.x, 0, 1)
        return weights / weights.sum()

    losses = []
    for j in range(n_learners):
        pred = predictions[:, j]
        if classification:
            pred = np.clip(pred, 1e-6, 1 - 1e-6)
            loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
        else:
            loss = np.mean((y - pred) ** 2)
        losses.append(loss)
    weights = np.zeros(n_learners)
    weights[int(np.argmin(losses))] = 1
    return weights


class ConvexSuperLearnerRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimators, cv=3, random_state=42, group_column=-1):
        self.estimators = estimators
        self.cv = cv
        self.random_state = random_state
        self.group_column = group_column

    def fit(self, X, y):
        features, groups = feature_and_groups(X, self.group_column)
        y = np.asarray(y, dtype=float)
        splits = inner_splits(y, groups, self.cv, self.random_state, classification=False)
        oof = np.full((len(y), len(self.estimators)), np.nan)
        for j, (_, estimator) in enumerate(self.estimators):
            for train, test in splits:
                fitted = clone(estimator).fit(features[train], y[train])
                oof[test, j] = fitted.predict(features[test])
        self.weights_ = convex_weights(oof, y, classification=False)
        self.learner_names_ = [name for name, _ in self.estimators]
        self.models_ = [clone(estimator).fit(features, y) for _, estimator in self.estimators]
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        features, _ = feature_and_groups(X, self.group_column)
        matrix = np.column_stack([model.predict(features) for model in self.models_])
        return matrix @ self.weights_


class ConvexSuperLearnerClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimators, cv=3, random_state=42, group_column=-1):
        self.estimators = estimators
        self.cv = cv
        self.random_state = random_state
        self.group_column = group_column

    def fit(self, X, y):
        features, groups = feature_and_groups(X, self.group_column)
        y = np.asarray(y, dtype=int)
        if np.unique(y).size != 2:
            raise ValueError("Both treatment classes are required.")
        splits = inner_splits(y, groups, self.cv, self.random_state, classification=True)
        oof = np.full((len(y), len(self.estimators)), np.nan)
        for j, (_, estimator) in enumerate(self.estimators):
            for train, test in splits:
                fitted = clone(estimator).fit(features[train], y[train])
                oof[test, j] = positive_probability(fitted, features[test])
        self.weights_ = convex_weights(oof, y, classification=True)
        self.learner_names_ = [name for name, _ in self.estimators]
        self.models_ = [clone(estimator).fit(features, y) for _, estimator in self.estimators]
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict_proba(self, X):
        features, _ = feature_and_groups(X, self.group_column)
        matrix = np.column_stack([positive_probability(model, features) for model in self.models_])
        probability = np.clip(matrix @ self.weights_, 1e-8, 1 - 1e-8)
        return np.column_stack([1 - probability, probability])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def learner_library(seed: int = 42, inner_folds: int = 3):
    reg = [
        ("ols", LinearRegression()),
        ("lasso", Pipeline([
            ("scale", StandardScaler()),
            ("model", LassoCV(cv=3, max_iter=5000, n_jobs=1, random_state=seed)),
        ])),
        ("ridge", Pipeline([
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 7), cv=3)),
        ])),
        ("elastic_net", Pipeline([
            ("scale", StandardScaler()),
            ("model", ElasticNetCV(
                cv=3, l1_ratio=[0.25, 0.5, 0.75], max_iter=5000,
                n_jobs=1, random_state=seed,
            )),
        ])),
        ("random_forest", RandomForestRegressor(
            n_estimators=250, max_depth=15, min_samples_leaf=5,
            random_state=seed, n_jobs=1,
        )),
        ("xgboost", XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=1, eval_metric="rmse",
        )),
    ]

    grid = np.logspace(-3, 3, 7)
    clf = [
        ("logit", LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)),
        ("lasso_logit", LogisticRegressionCV(
            Cs=grid, cv=3, penalty="l1", solver="liblinear",
            scoring="neg_log_loss", max_iter=2000, n_jobs=1,
            random_state=seed,
        )),
        ("ridge_logit", LogisticRegressionCV(
            Cs=grid, cv=3, penalty="l2", solver="lbfgs",
            scoring="neg_log_loss", max_iter=2000, n_jobs=1,
            random_state=seed,
        )),
        ("elastic_net_logit", LogisticRegressionCV(
            Cs=np.logspace(-2, 2, 5), cv=3, penalty="elasticnet",
            solver="saga", l1_ratios=[0.5], scoring="neg_log_loss",
            max_iter=3000, n_jobs=1, random_state=seed,
        )),
        ("random_forest", RandomForestClassifier(
            n_estimators=250, max_depth=15, min_samples_leaf=5,
            random_state=seed, n_jobs=1,
        )),
        ("xgboost", XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=1, eval_metric="logloss",
        )),
    ]
    return (
        ConvexSuperLearnerRegressor(reg, cv=inner_folds, random_state=seed, group_column=-1),
        ConvexSuperLearnerClassifier(clf, cv=inner_folds, random_state=seed, group_column=-1),
    )


def raw_oof_propensities(
    learner,
    frame: pd.DataFrame,
    x_cols: list[str],
    smpls: list,
    trim: float,
    sample_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = frame[x_cols].to_numpy(dtype=float)
    observed = frame["treat_cat"].to_numpy(dtype=int)
    records: list[pd.DataFrame] = []
    weight_records: list[dict] = []

    for level, label in LEVELS.items():
        y = (observed == level).astype(int)
        for repetition, folds in enumerate(smpls):
            prediction = np.full(len(frame), np.nan)
            for fold, (train, test) in enumerate(folds):
                fitted = clone(learner).fit(X[train], y[train])
                prediction[test] = fitted.predict_proba(X[test])[:, 1]
                for name, weight in zip(fitted.learner_names_, fitted.weights_):
                    weight_records.append({
                        "sample": sample_name,
                        "treatment_category": label,
                        "repetition": repetition,
                        "fold": fold,
                        "learner": name,
                        "weight": weight,
                    })
            if np.isnan(prediction).any():
                raise RuntimeError(f"Incomplete OOF propensity predictions for {label}.")
            records.append(pd.DataFrame({
                "sample": sample_name,
                "_row_id": frame["_row_id"].to_numpy(),
                "country": frame["country_cat"].to_numpy(),
                "observed_category": observed,
                "treatment_code": level,
                "treatment_category": label,
                "repetition": repetition,
                "propensity_raw": prediction,
                "propensity_clipped": np.clip(prediction, trim, 1 - trim),
            }))

    return pd.concat(records, ignore_index=True), pd.DataFrame(weight_records)


def propensity_summary(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (sample, label), group in oof.groupby(["sample", "treatment_category"]):
        p = group["propensity_raw"]
        rows.append({
            "sample": sample,
            "treatment_category": label,
            "P1": p.quantile(0.01),
            "P5": p.quantile(0.05),
            "median": p.quantile(0.50),
            "P95": p.quantile(0.95),
            "P99": p.quantile(0.99),
            "below_0.01_percent": 100 * p.lt(0.01).mean(),
            "below_0.025_percent": 100 * p.lt(0.025).mean(),
            "below_0.05_percent": 100 * p.lt(0.05).mean(),
            "above_0.99_percent": 100 * p.gt(0.99).mean(),
        })
    return pd.DataFrame(rows)


def save_pickle(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def model_signature(kind, frame, outcome, treatment, x_cols, smpls, learner_g, learner_m, trim):
    return joblib_hash({
        "kind": kind,
        "rows": joblib_hash(frame[["_row_id", outcome, treatment, "_psu_id", "_hh_id"] + x_cols]),
        "outcome": outcome,
        "treatment": treatment,
        "x_cols": x_cols,
        "splits": joblib_hash(smpls),
        "ml_g": joblib_hash(learner_g),
        "ml_m": joblib_hash(learner_m),
        "trim": trim,
        "levels": LEVELS,
        "doubleml": dml.__version__,
    })


def fit_irm(
    frame,
    x_cols,
    outcome,
    treatment,
    smpls,
    smpls_cluster,
    learner_g,
    learner_m,
    trim,
    cache_path: Path,
    n_jobs_cv: int,
):
    signature = model_signature(
        "IRM", frame, outcome, treatment, x_cols, smpls, learner_g, learner_m, trim
    )
    if cache_path.is_file():
        with cache_path.open("rb") as file:
            cached = pickle.load(file)
        if cached.get("signature") == signature:
            return cached["model"]

    data = dml.DoubleMLData(
        frame,
        y_col=outcome,
        d_cols=treatment,
        x_cols=x_cols,
        cluster_cols="_psu_id",
    )
    model = dml.DoubleMLIRM(
        data,
        ml_g=clone(learner_g),
        ml_m=clone(learner_m),
        n_folds=len(smpls[0]),
        n_rep=len(smpls),
        score="ATE",
        draw_sample_splitting=False,
        trimming_rule="truncate",
        trimming_threshold=trim,
    )
    model.set_sample_splitting(smpls, smpls_cluster)
    model.fit(n_jobs_cv=n_jobs_cv, store_predictions=True, store_models=False)
    save_pickle(cache_path, {"signature": signature, "model": model})
    return model


def fit_apos(
    frame,
    x_cols,
    outcome,
    treatment,
    smpls,
    smpls_cluster,
    learner_g,
    learner_m,
    trim,
    cache_path: Path,
    n_jobs_cv: int,
    n_jobs_models: int,
):
    signature = model_signature(
        "APOS", frame, outcome, treatment, x_cols, smpls, learner_g, learner_m, trim
    )
    if cache_path.is_file():
        with cache_path.open("rb") as file:
            cached = pickle.load(file)
        if cached.get("signature") == signature:
            return cached["model"], cached["contrast"]

    data = dml.DoubleMLData(
        frame,
        y_col=outcome,
        d_cols=treatment,
        x_cols=x_cols,
        cluster_cols="_psu_id",
    )
    model = dml.DoubleMLAPOS(
        data,
        ml_g=clone(learner_g),
        ml_m=clone(learner_m),
        treatment_levels=list(LEVELS),
        n_folds=len(smpls[0]),
        n_rep=len(smpls),
        draw_sample_splitting=False,
        trimming_rule="truncate",
        trimming_threshold=trim,
    )
    model.set_sample_splitting(smpls, smpls_cluster)
    model.fit(
        n_jobs_models=n_jobs_models,
        n_jobs_cv=n_jobs_cv,
        store_predictions=True,
        store_models=False,
    )
    contrast = model.causal_contrast(reference_levels=[0])
    save_pickle(cache_path, {"signature": signature, "model": model, "contrast": contrast})
    return model, contrast


def summary_value(summary: pd.Series, candidates: list[str]):
    for candidate in candidates:
        if candidate in summary.index:
            return summary[candidate]
    raise KeyError(f"None of {candidates} appears in the DoubleML summary.")


def irm_result(model, dataset: str, outcome: str) -> dict:
    row = model.summary.iloc[0]
    return {
        "method": "IRM",
        "dataset": dataset,
        "outcome": outcome,
        "comparison": "Any recognized treatment vs No treatment",
        "coef": summary_value(row, ["coef"]),
        "se": summary_value(row, ["std err", "std_err"]),
        "ci_low": summary_value(row, ["2.5 %", "2.5%"]),
        "ci_high": summary_value(row, ["97.5 %", "97.5%"]),
    }


def contrast_results(contrast, dataset: str, outcome: str) -> list[dict]:
    rows: list[dict] = []
    summary = contrast.summary.reset_index(drop=False)
    non_reference = [1, 2, 3]
    if len(summary) != len(non_reference):
        raise ValueError("Unexpected number of APOS contrasts.")
    for level, (_, row) in zip(non_reference, summary.iterrows()):
        rows.append({
            "method": "APOS contrast",
            "dataset": dataset,
            "outcome": outcome,
            "comparison": f"{LEVELS[level]} vs No treatment",
            "coef": summary_value(row, ["coef"]),
            "se": summary_value(row, ["std err", "std_err"]),
            "ci_low": summary_value(row, ["2.5 %", "2.5%"]),
            "ci_high": summary_value(row, ["97.5 %", "97.5%"]),
        })
    return rows


def eligible_countries(detail: pd.DataFrame, level: int, minimum_psu: int) -> set:
    subset = detail.loc[detail["treatment_code"].eq(level)]
    condition = (
        subset["both_categories_present"]
        if minimum_psu == 1
        else subset["at_least_2_PSU_each"]
    )
    return set(subset.loc[condition, "country"])
