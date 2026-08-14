"""Run the MICS stacked DoubleML analysis and create publication outputs.

The pipeline estimates the main IRM/APOS models, computes leave-one-control-
group-out sensitivity measures, and projects existing orthogonal scores onto
prespecified heterogeneity groups. The last two stages are post-estimation
analyses and do not change the main causal estimands.
"""

import json
import gc
from pathlib import Path
from time import perf_counter

import doubleml as dml
import numpy as np
import pandas as pd
import pyreadstat
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from xgboost import XGBClassifier, XGBRegressor
from _tables import write_water_treatment_table
from _functions import (
    EstimateCache,
    make_cache_context,
    make_estimate_key,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


seed = 42
folds = 3
repetitions = 1
# Sampled 10%
sampled = True
# Keep DoubleML's process-level parallelism disabled.  In DoubleML 0.11.3,
# APOS parallelized across treatment levels can share read-only NumPy arrays
# between workers and fail while writing psi elements.  Sequential fitting is
# slower, but avoids that failure and nested process/memory multiplication.
jobs_cv = 1
inner_folds = 3
max_fold_attempts = 100

treatment_levels = (0, 1, 2, 3, 98)
report_levels = (0, 1, 2, 3)

project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / "Data" / "3. Final"
output_dir = project_root / "Output"
checkpoint_dir = output_dir / "checkpoints"
output_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir.mkdir(parents=True, exist_ok=True)
estimate_cache = EstimateCache(checkpoint_dir)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


data_paths = {
    "HH": data_dir / "MASTER_MICS_FINAL.dta",
    "U5": data_dir / "MASTER_MICS_FINAL_U5.dta",
}
cache_context = make_cache_context(
    Path(__file__),
    data_paths,
    config={
        "cache_schema_version": 2,
        "seed": seed,
        "folds": folds,
        "inner_folds": inner_folds,
        "repetitions": repetitions,
        "treatment_levels": treatment_levels,
    },
    doubleml_version=getattr(dml, "__version__", "unknown"),
)


def make_frame(
    data, outcome, treatment, allowed_levels, keep_cluster, child=False
):
    """Build the exact model frame for one DoubleML specification.

    The function applies complete-case and treatment-support restrictions,
    expands categorical controls into dummies, and preserves the row order
    used by DoubleML. RiskSource is intentionally excluded because it is a
    prespecified heterogeneity variable rather than a main-model control.
    """

    controls = [
        "windex5",
        "urban",
        "WS1_g",
        "wq27_decile",
        "Any_U5",
        "Girls_less_than15",
        "Boys_15or_less",
        "Toilet",
    ]
    required = [outcome, treatment, "country_cat", *controls]
    if keep_cluster:
        required.append("Cluster_var")
    if child:
        required.extend(["age", "male"])

    keep = data[required].notna().all(axis=1)
    keep &= data[treatment].isin(allowed_levels)
    sample = data.loc[keep].copy()

    categorical_blocks = [
        pd.get_dummies(
            sample[column].astype("string"),
            prefix=prefix,
            drop_first=True,
            dtype=float,
        )
        for column, prefix in [
            ("windex5", "wealth"),
            ("country_cat", "country"),
            ("WS1_g", "water_source"),
            ("Toilet", "toilet"),
            ("wq27_decile", "source_ecoli"),
        ]
    ]
    numeric_block = sample[[
        "urban",
        "Any_U5",
        "Girls_less_than15",
        "Boys_15or_less",
    ]].astype(float)
    blocks = [*categorical_blocks, numeric_block]
    if child:
        blocks.extend([
            pd.get_dummies(
                sample["age"].astype("string"),
                prefix="child_age",
                drop_first=True,
                dtype=float,
            ),
            sample[["male"]].astype(float),
        ])
    x = pd.concat(blocks, axis=1)

    # The cluster code is metadata for the convex learner's inner CV. It is
    # removed before fitting the base learners and is never used as a feature.
    if keep_cluster:
        x["_cluster_model_code"] = pd.factorize(
            sample["Cluster_var"], sort=True
        )[0].astype(float)

    columns = ["country_cat", outcome, treatment]
    if keep_cluster:
        columns.insert(1, "Cluster_var")

    frame = pd.concat([sample[columns], x], axis=1).reset_index(drop=True)
    frame[outcome] = pd.to_numeric(
        frame[outcome], errors="raise"
    ).astype(float)
    frame[treatment] = pd.to_numeric(
        frame[treatment], errors="raise"
    ).astype(int)

    return frame, list(x.columns)


def make_cluster_folds(frame, treatment):
    """Create stratified cross-fitting folds that respect sampling clusters.

    Treatment support is balanced across folds while every cluster stays
    entirely in either the training or test portion of a fold. These folds
    are reused in clustered main, sensitivity, and GATE calculations.
    """

    groups = frame["Cluster_var"].to_numpy()
    target = frame[treatment].to_numpy()

    all_smpls = []
    all_smpls_cluster = []

    for repetition in range(repetitions):
        selected = None
        for attempt in range(max_fold_attempts):
            splitter = StratifiedGroupKFold(
                n_splits=folds,
                shuffle=True,
                random_state=seed + repetition * 100 + attempt,
            )
            candidate = list(splitter.split(np.zeros(len(frame)), target, groups))
            valid = all(
                np.unique(target[train]).size == np.unique(target).size
                for train, _ in candidate
            )
            if valid:
                selected = candidate
                break

        if selected is None:
            raise ValueError("Could not construct valid cluster-level folds.")

        smpls = []
        smpls_cluster = []
        for train, test in selected:
            train_groups = np.unique(groups[train])
            test_groups = np.unique(groups[test])
            if np.intersect1d(train_groups, test_groups).size:
                raise AssertionError("A cluster appears in train and test.")
            smpls.append((train, test))
            smpls_cluster.append(([train_groups], [test_groups]))

        all_smpls.append(smpls)
        all_smpls_cluster.append(smpls_cluster)

    return all_smpls, all_smpls_cluster


# ---------------------------------------------------------------------------
# Convex Super Learner
# ---------------------------------------------------------------------------


def feature_and_groups(x, group_column):
    """Remove the metadata column used to construct grouped inner folds."""

    array = np.asarray(x, dtype=float)
    index = (
        group_column
        if group_column >= 0
        else array.shape[1] + group_column
    )
    if index < 0 or index >= array.shape[1]:
        raise IndexError("Invalid cluster-column index.")
    groups = array[:, index].astype(int)
    return np.delete(array, index, axis=1), groups


def inner_splits(y, groups, n_splits, random_state, classification):
    """Create valid grouped inner folds for a convex learner."""

    max_splits = min(int(n_splits), len(np.unique(groups)))
    for k in range(max_splits, 1, -1):
        if classification:
            splitter = StratifiedGroupKFold(
                n_splits=k, shuffle=True, random_state=random_state
            )
            candidate = list(splitter.split(np.zeros(len(y)), y, groups))
            valid = all(
                np.unique(y[train]).size == 2
                for train, _ in candidate
            )
        else:
            splitter = GroupKFold(n_splits=k)
            candidate = list(splitter.split(np.zeros(len(y)), y, groups))
            valid = True
        if valid:
            return candidate
    raise ValueError("Unable to construct grouped inner folds.")


def positive_probability(model, x):
    """Return P(Y=1), independent of the estimator's class ordering."""

    classes = np.asarray(model.classes_)
    positive_index = int(np.where(classes == 1)[0][0])
    return model.predict_proba(x)[:, positive_index]


def convex_weights(predictions, target, classification):
    """Fit nonnegative weights summing to one from OOF predictions."""

    learners = predictions.shape[1]
    initial = np.repeat(1 / learners, learners)

    def loss(weights):
        fitted = predictions @ weights
        if classification:
            fitted = np.clip(fitted, 1e-6, 1 - 1e-6)
            return -np.mean(
                target * np.log(fitted)
                + (1 - target) * np.log(1 - fitted)
            )
        return np.mean((target - fitted) ** 2)

    result = minimize(
        loss,
        initial,
        method="SLSQP",
        bounds=[(0, 1)] * learners,
        constraints={
            "type": "eq",
            "fun": lambda weights: weights.sum() - 1,
        },
    )
    if result.success:
        weights = np.clip(result.x, 0, 1)
        return weights / weights.sum()

    losses = [loss(np.eye(learners)[index]) for index in range(learners)]
    weights = np.zeros(learners)
    weights[int(np.argmin(losses))] = 1
    return weights


class ConvexRegressor(RegressorMixin, BaseEstimator):
    """Cross-fitted convex ensemble for the outcome nuisance function.

    Base learners produce out-of-fold predictions, and nonnegative weights
    summing to one are selected by prediction loss. The ensemble supplies
    predictions to DoubleML; its weights are diagnostics, not causal weights.
    """

    def __init__(
        self, estimators, random_state=42, group_column=None, refit_full=False
    ):
        self.estimators = estimators
        self.random_state = random_state
        self.group_column = group_column
        self.refit_full = refit_full

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.group_column is None:
            features = x
            splits = list(KFold(
                n_splits=inner_folds,
                shuffle=True,
                random_state=self.random_state,
            ).split(features))
        else:
            features, groups = feature_and_groups(x, self.group_column)
            splits = inner_splits(
                y,
                groups,
                inner_folds,
                self.random_state,
                classification=False,
            )
        splits = list(splits)
        oof = np.full((len(y), len(self.estimators)), np.nan)

        fitted_models = []
        for column, (name, estimator) in enumerate(self.estimators):
            fold_models = []
            for train, test in splits:
                fitted = clone(estimator).fit(
                    features[train], y[train]
                )
                oof[test, column] = fitted.predict(features[test])
                fold_models.append(fitted)
            fitted_models.append((name, fold_models))

        self.weights_ = convex_weights(oof, y, classification=False)
        if self.refit_full:
            self.models_ = [
                (
                    name,
                    [clone(estimator).fit(features, y)],
                )
                for name, estimator in self.estimators
            ]
        else:
            # DoubleML only needs predictions on future outer-test folds. The
            # already-fitted inner-fold models can produce those predictions;
            # refitting the same base learners on all outer-training rows is
            # redundant for this use case.
            self.models_ = fitted_models
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        if self.group_column is None:
            features = x
        else:
            features, _ = feature_and_groups(x, self.group_column)
        predictions = np.column_stack([
            np.mean(
                [model.predict(features) for model in models], axis=0
            )
            for _, models in self.models_
        ])
        return predictions @ self.weights_


class ConvexClassifier(ClassifierMixin, BaseEstimator):
    """Cross-fitted convex ensemble for the binary treatment propensity."""

    def __init__(
        self, estimators, random_state=42, group_column=None, refit_full=False
    ):
        self.estimators = estimators
        self.random_state = random_state
        self.group_column = group_column
        self.refit_full = refit_full

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int)
        if np.unique(y).size != 2:
            raise ValueError("Both treatment classes are required.")
        if self.group_column is None:
            features = x
            splits = list(StratifiedKFold(
                n_splits=inner_folds,
                shuffle=True,
                random_state=self.random_state,
            ).split(features, y))
        else:
            features, groups = feature_and_groups(x, self.group_column)
            splits = inner_splits(
                y,
                groups,
                inner_folds,
                self.random_state,
                classification=True,
            )
        splits = list(splits)
        oof = np.full((len(y), len(self.estimators)), np.nan)

        fitted_models = []
        for column, (name, estimator) in enumerate(self.estimators):
            fold_models = []
            for train, test in splits:
                fitted = clone(estimator).fit(
                    features[train], y[train]
                )
                oof[test, column] = positive_probability(fitted, features[test])
                fold_models.append(fitted)
            fitted_models.append((name, fold_models))

        self.weights_ = convex_weights(oof, y, classification=True)
        if self.refit_full:
            self.models_ = [
                (
                    name,
                    [clone(estimator).fit(features, y)],
                )
                for name, estimator in self.estimators
            ]
        else:
            self.models_ = fitted_models
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = x.shape[1]
        return self

    def predict_proba(self, x):
        x = np.asarray(x, dtype=float)
        if self.group_column is None:
            features = x
        else:
            features, _ = feature_and_groups(x, self.group_column)
        probabilities = np.column_stack([
            np.mean(
                [positive_probability(model, features) for model in models],
                axis=0,
            )
            for _, models in self.models_
        ])
        positive = np.clip(probabilities @ self.weights_, 1e-8, 1 - 1e-8)
        return np.column_stack([1 - positive, positive])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Base learner libraries and analysis frames
# ---------------------------------------------------------------------------


regressors = [
    ("ols", LinearRegression()),
    ("lasso", Pipeline([
        ("scale", StandardScaler()),
        ("model", LassoCV(cv=3, max_iter=5_000, n_jobs=-1, random_state=seed)),
    ])),
    ("elastic_net", Pipeline([
        ("scale", StandardScaler()),
        ("model", ElasticNetCV(
            cv=3, l1_ratio=[0.25, 0.5, 0.75], max_iter=5_000,
            n_jobs=-1, random_state=seed,
        )),
    ])),
    ("random_forest", RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        random_state=seed, n_jobs=-1,
    )),
    ("xgboost", XGBRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=seed, n_jobs=-1,
        eval_metric="rmse", verbosity=0,
    )),
]


classifiers = [
    ("logit", LogisticRegression(
        C=np.inf, l1_ratio=0, solver="lbfgs", max_iter=2_000,
    )),
    ("lasso", LogisticRegressionCV(
        cv=3, l1_ratios=(1,), solver="liblinear", max_iter=2_000,
        n_jobs=-1, random_state=seed, scoring="neg_log_loss",
        use_legacy_attributes=True,
    )),
    ("elastic_net", LogisticRegressionCV(
        cv=3, l1_ratios=(0.5,), solver="saga", max_iter=3_000,
        n_jobs=-1, random_state=seed, scoring="neg_log_loss",
        use_legacy_attributes=True,
    )),
    ("random_forest", RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        random_state=seed, n_jobs=-1,
    )),
    ("xgboost", XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=seed, n_jobs=-1,
        eval_metric="logloss", verbosity=0,
    )),
]

ml_g = ConvexRegressor(regressors, random_state=seed, refit_full=False)
ml_m = ConvexClassifier(classifiers, random_state=seed, refit_full=False)
ml_g_cluster = ConvexRegressor(
    regressors, random_state=seed, group_column=-1, refit_full=False
)
ml_m_cluster = ConvexClassifier(
    classifiers, random_state=seed, group_column=-1, refit_full=False
)

common_data_columns = [
    "water_treatment",
    "WQ15_g",
    "country_cat",
    "windex5",
    "urban",
    "WS1_g",
    "wq27_decile",
    "Any_U5",
    "Girls_less_than15",
    "Boys_15or_less",
    "Toilet",
    "Cluster_var",
]
hh_columns = sorted(set([
    "SomeRiskHome",
    "VeryHighRiskHome",
    *common_data_columns,
]))
u5_columns = sorted(set([
    "diarrhea",
    *common_data_columns,
    "age",
    "male",
]))

# The source files contain hundreds of unrelated columns.  Loading all of
# them as pandas objects uses roughly 2 GB before estimation even starts.
hh, _ = pyreadstat.read_dta(
    data_dir / "MASTER_MICS_FINAL.dta", usecols=hh_columns
)
if sampled == True:
    hh = hh.sample(frac=0.1, random_state=42)

u5, _ = pyreadstat.read_dta(
    data_dir / "MASTER_MICS_FINAL_U5.dta", usecols=u5_columns
)
if sampled == True:
    u5 = u5.sample(frac=0.1, random_state=42)

analysis_specs = [
    ("HH", hh, False, "SomeRiskHome"),
    ("HH", hh, False, "VeryHighRiskHome"),
    ("U5", u5, True, "diarrhea"),
]


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


def run_estimation(progress, label, fit):
    """Run one opaque DoubleML fit while keeping the progress display useful."""

    progress.set_description(label)
    started = perf_counter()
    result = fit()
    elapsed = perf_counter() - started
    progress.update(1)
    progress.set_postfix(last=f"{elapsed:.1f}s", refresh=True)
    return result


class DiskBackedEstimate:
    """Lazy handle for a fitted estimate stored in the on-disk cache.

    The fitted DoubleML object is intentionally not kept in ``estimates``.
    Attribute access loads it only for the operation that needs it.
    This reduces memory pressure because fitted base learners can be large.
    Mutating methods must use the object they return; this is important for
    DoubleML sensitivity analysis.
    """

    def __init__(self, cache, key):
        self._cache = cache
        self._key = key

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        fitted = self._cache.load(self._key)
        return getattr(fitted, name)


def _contains_super_learner_weights(value):
    """Return whether a cached DoubleML model contains fitted SL weights."""

    if hasattr(value, "weights_"):
        return True
    if isinstance(value, dict):
        return any(_contains_super_learner_weights(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_super_learner_weights(item) for item in value)
    return False


def _cached_model_has_weights(handle, nuisance):
    """Inspect one cached model, releasing it immediately afterwards."""

    fitted = estimate_cache.load(handle._key)
    try:
        models = getattr(fitted, "models", None)
        if not isinstance(models, dict):
            return False
        return _contains_super_learner_weights(models.get(nuisance, {}))
    finally:
        del fitted
        gc.collect()


def run_or_load(
    progress, label, fit, dataset, outcome, model, clustered,
    require_super_learner_weights=False,
):
    """Load a compatible estimate or fit it once through the disk cache.

    Cache keys include source code, data files, design configuration, and the
    DoubleML version. When Super Learner diagnostics are required, this
    function verifies that ``ml_g`` weights are stored and refits under a
    separate key if an older checkpoint is incomplete.
    """

    key = make_estimate_key(
        cache_context,
        dataset,
        outcome,
        model,
        clustered,
    )
    if estimate_cache.contains(key):
        cached_handle = DiskBackedEstimate(estimate_cache, key)
        if (
            require_super_learner_weights
            and not _cached_model_has_weights(cached_handle, "ml_g0")
        ):
            # Historical cache entries may contain the causal estimate but
            # not the stored nuisance learners.  Use a distinct key so a
            # complete estimate is never overwritten or refit.
            key = make_estimate_key(
                cache_context,
                dataset,
                outcome,
                f"{model}_stored_super_learner_weights_ml_g_v2",
                clustered,
            )
            if estimate_cache.contains(key):
                stored_handle = DiskBackedEstimate(estimate_cache, key)
                if _cached_model_has_weights(stored_handle, "ml_g0"):
                    progress.set_description(f"{label} [weights cached]")
                    progress.update(1)
                    progress.set_postfix(cache="weights hit", refresh=True)
                    return stored_handle
            # Historical cache entries may contain the causal estimate or
            # only ml_m weights. Refit under a new key with ml_g stored.
            progress.set_description(f"{label} [refitting for ml_g weights]")
            fitted = run_estimation(
                progress,
                label,
                lambda: estimate_cache.get_or_fit(key, fit),
            )
            del fitted
            gc.collect()
            return DiskBackedEstimate(estimate_cache, key)
        progress.set_description(f"{label} [cached]")
        progress.update(1)
        progress.set_postfix(cache="hit", refresh=True)
        return cached_handle

    fitted = run_estimation(
        progress,
        label,
        lambda: estimate_cache.get_or_fit(key, fit),
    )
    # Joblib has already persisted the result.  Drop the in-memory copy before
    # the next nuisance model starts; downstream code uses a lazy disk handle.
    del fitted
    gc.collect()
    return DiskBackedEstimate(estimate_cache, key)


progress = tqdm(
    total=len(analysis_specs) * 4,
    desc="Estimating",
    unit="model",
    dynamic_ncols=True,
)

estimates = {}

for dataset_name, data, child, outcome in analysis_specs:
    irm_frame_cluster, irm_x_cluster = make_frame(
        data,
        outcome,
        "water_treatment",
        (0, 1),
        True,
        child=child,
    )
    irm_frame_no_cluster, irm_x_no_cluster = make_frame(
        data,
        outcome,
        "water_treatment",
        (0, 1),
        False,
        child=child,
    )
    apos_frame_cluster, apos_x_cluster = make_frame(
        data,
        outcome,
        "WQ15_g",
        treatment_levels,
        True,
        child=child,
    )
    apos_frame_no_cluster, apos_x_no_cluster = make_frame(
        data,
        outcome,
        "WQ15_g",
        treatment_levels,
        False,
        child=child,
    )

    irm_data_cluster = dml.DoubleMLData(
        irm_frame_cluster,
        y_col=outcome,
        d_cols="water_treatment",
        x_cols=irm_x_cluster,
        cluster_cols="Cluster_var",
    )
    irm_cluster = dml.DoubleMLIRM(
        irm_data_cluster,
        clone(ml_g_cluster),
        clone(ml_m_cluster),
        n_folds=folds,
        n_rep=repetitions,
        draw_sample_splitting=False,
    )
    irm_smpls, irm_smpls_cluster = make_cluster_folds(
        irm_frame_cluster, "water_treatment"
    )
    irm_cluster.set_sample_splitting(irm_smpls, irm_smpls_cluster)
    irm_cluster = run_or_load(
        progress,
        f"{dataset_name} {outcome}: IRM clustered",
        lambda model=irm_cluster: model.fit(
            n_jobs_cv=jobs_cv, store_predictions=False, store_models=True
        ),
        dataset_name,
        outcome,
        "irm",
        True,
        require_super_learner_weights=True,
    )

    irm_data_no_cluster = dml.DoubleMLData(
        irm_frame_no_cluster,
        y_col=outcome,
        d_cols="water_treatment",
        x_cols=irm_x_no_cluster,
    )
    irm_no_cluster = dml.DoubleMLIRM(
        irm_data_no_cluster,
        clone(ml_g),
        clone(ml_m),
        n_folds=folds,
        n_rep=repetitions,
    )
    irm_no_cluster = run_or_load(
        progress,
        f"{dataset_name} {outcome}: IRM unclustered",
        lambda model=irm_no_cluster: model.fit(
            n_jobs_cv=jobs_cv, store_predictions=False, store_models=True
        ),
        dataset_name,
        outcome,
        "irm",
        False,
        require_super_learner_weights=True,
    )

    # DoubleML 0.11.3 cannot attach cluster_cols to APOS. We retain the
    # cluster-level folds and compute the APOS cluster sandwich below from the
    # fitted contrast influence scores.
    apos_smpls, _ = make_cluster_folds(apos_frame_cluster, "WQ15_g")
    apos_data_cluster = dml.DoubleMLData(
        apos_frame_cluster,
        y_col=outcome,
        d_cols="WQ15_g",
        x_cols=apos_x_cluster,
    )
    apos_cluster = dml.DoubleMLAPOS(
        apos_data_cluster,
        clone(ml_g_cluster),
        clone(ml_m_cluster),
        treatment_levels=list(treatment_levels),
        n_folds=folds,
        n_rep=repetitions,
        draw_sample_splitting=False,
    )
    apos_cluster.set_sample_splitting(apos_smpls)
    apos_cluster = run_or_load(
        progress,
        f"{dataset_name} {outcome}: APOS clustered folds",
        lambda model=apos_cluster: model.fit(
            n_jobs_models=1,
            n_jobs_cv=jobs_cv,
            store_predictions=False,
            store_models=True,
        ),
        dataset_name,
        outcome,
        "apos",
        True,
        require_super_learner_weights=True,
    )

    apos_data_no_cluster = dml.DoubleMLData(
        apos_frame_no_cluster,
        y_col=outcome,
        d_cols="WQ15_g",
        x_cols=apos_x_no_cluster,
    )
    apos_no_cluster = dml.DoubleMLAPOS(
        apos_data_no_cluster,
        clone(ml_g),
        clone(ml_m),
        treatment_levels=list(treatment_levels),
        n_folds=folds,
        n_rep=repetitions,
    )
    apos_no_cluster = run_or_load(
        progress,
        f"{dataset_name} {outcome}: APOS unclustered",
        lambda model=apos_no_cluster: model.fit(
            n_jobs_models=1,
            n_jobs_cv=jobs_cv,
            store_predictions=False,
            store_models=True,
        ),
        dataset_name,
        outcome,
        "apos",
        False,
        require_super_learner_weights=True,
    )

    estimates[(dataset_name, outcome)] = {
        "irm_cluster": irm_cluster,
        "irm_no_cluster": irm_no_cluster,
        "apos_cluster": apos_cluster,
        "apos_no_cluster": apos_no_cluster,
        "irm_n": len(irm_frame_cluster),
        "apos_n": len(apos_frame_cluster),
        "outcome_mean": irm_frame_cluster[outcome].mean(),
        "clusters": irm_frame_cluster["Cluster_var"].nunique(),
        "irm_frame_cluster": irm_frame_cluster,
        "irm_frame_no_cluster": irm_frame_no_cluster,
        "apos_frame_cluster": apos_frame_cluster,
        "apos_frame_no_cluster": apos_frame_no_cluster,
    }

progress.close()


# Results
# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

from _tables import (
    write_publication_table,
    write_stacked_regression_table,
    write_super_learner_weights_tables,
    create_relative_sensitivity_tables,
    create_relative_sensitivity_comparison_tables,
    create_combined_sensitivity_tables,
    create_benchmark_sensitivity_tables,
    create_heterogeneity_comparison_tables,
)
from _functions import (
    cluster_robust_framework_se,
    cached_reduced_fit,
    estimate_gate_from_contrast,
    select_column_groups,
    sensitivity_params,
)
from scipy.stats import norm
import threading


SENSITIVITY_GROUPS = {
    # Benchmark: native DoubleML sensitivity at cf_y = cf_d = 0.03,
    # compared with dropping only the source-water E. coli deciles.
    "source_ecoli": "source_ecoli_",
}


SENSITIVITY_LABELS = {
    "source_ecoli": "Drop source-water E.coli deciles",
}


def prepare_sensitivity_inputs(estimates):
    """Rebuild model matrices for sensitivity without refitting full models.

    The fitted objects are loaded first.  We then recreate only the complete-
    case frames and column lists needed by leave-one-control-group-out fits.
    This keeps the expensive full-model stage independent from presentation
    and sensitivity bookkeeping.
    """

    for dataset, data, child, outcome in analysis_specs:
        bundle = estimates[(dataset, outcome)]
        bundle["child"] = child
        bundle["irm_frame_cluster"], bundle["irm_x_cluster"] = make_frame(
            data, outcome, "water_treatment", (0, 1), True, child=child
        )
        bundle["irm_frame_no_cluster"], bundle["irm_x_no_cluster"] = make_frame(
            data, outcome, "water_treatment", (0, 1), False, child=child
        )
        bundle["apos_frame_cluster"], bundle["apos_x_cluster"] = make_frame(
            data, outcome, "WQ15_g", treatment_levels, True, child=child
        )
        bundle["apos_frame_no_cluster"], bundle["apos_x_no_cluster"] = make_frame(
            data, outcome, "WQ15_g", treatment_levels, False, child=child
        )
        apos_contrast = bundle["apos_cluster"].causal_contrast(
            reference_levels=[0]
        )
        bundle["apos_cluster_se"] = cluster_robust_framework_se(
            apos_contrast,
            bundle["apos_frame_cluster"]["Cluster_var"].to_numpy(),
            bundle["apos_cluster"].smpls,
        )


prepare_sensitivity_inputs(estimates)

# ---------------------------------------------------------------------------
# Main publication tables
# ---------------------------------------------------------------------------
# These tables use only the already-fitted original IRM/APOS models.  They
# are deliberately written before the expensive leave-one-control-group-out
# sensitivity fits, so a long sensitivity run does not delay the first LaTeX
# outputs. The completed sensitivity stage later overwrites these same files
# with the final versions containing the sensitivity results.
early_table_paths = {
    "main": str(write_publication_table(
        output_dir,
        estimates,
        ["SomeRiskHome", "VeryHighRiskHome", "diarrhea"],
        "table_water_treatment_main.tex",
        "Stacked water-treatment effects",
        "tab:water-treatment-main",
        report_levels,
        folds,
        repetitions,
        specifications=("clustered",),
    )),
    "appendix": str(write_publication_table(
        output_dir,
        estimates,
        ["SomeRiskHome", "VeryHighRiskHome", "diarrhea"],
        "table_water_treatment_appendix.tex",
        "Stacked water-treatment effects: clustered and ordinary folds",
        "tab:water-treatment-appendix",
        report_levels,
        folds,
        repetitions,
        specifications=("clustered", "unclustered"),
    )),
}
for path in early_table_paths.values():
    print(f"Main LaTeX table saved to: {path}")

weights_table_paths = write_super_learner_weights_tables(
    output_dir,
    estimates,
    ["SomeRiskHome", "VeryHighRiskHome", "diarrhea"],
)
for weights_table_path in weights_table_paths:
    print(f"Super Learner weights table saved to: {weights_table_path}")


def run_sensitivity(estimates):
    """Legacy leave-one-control-group-out RV analysis.

    The active pipeline uses :func:`run_sensitivity_benchmark` below, which
    calls DoubleML's native ``sensitivity_benchmark()`` API. This function is
    retained for auditability and for reproducing earlier RV-relative tables;
    it is not called by the current results pipeline.
    APOS cluster-fold contrasts use the manually reconstructed cluster
    framework so both cluster SEs and RV-alpha use the same variance estimate.
    """

    rows = []
    treatment_labels = {
        1: "Boiling",
        2: "Chlorination/tablets",
        3: "Straining/settling",
    }

    # There are two reduced fits (IRM and APOS) for every available control
    # group, specification, and estimand.  Keep this progress bar explicit:
    # cached reduced fits still advance it, but never refit the original
    # models or the already-cached reduced models.
    sensitivity_specs = []
    for (dataset, outcome), bundle in estimates.items():
        if (outcome in {"SomeRiskHome", "VeryHighRiskHome"} and dataset != "HH"):
            continue
        if outcome == "diarrhea" and dataset != "U5":
            continue
        groups = select_column_groups(
            bundle["irm_x_cluster"], SENSITIVITY_GROUPS, bundle["child"]
        )
        sensitivity_specs.extend(
            (dataset, outcome, specification, group)
            for specification in ("clustered_folds", "unclustered")
            for group in groups
        )
    sensitivity_progress = tqdm(
        total=len(sensitivity_specs) * 2,
        desc="Sensitivity: reduced fits",
        unit="model",
        dynamic_ncols=True,
        leave=True,
    )
    tqdm.write(
        f"Sensitivity: {len(sensitivity_specs) * 2} reduced fits queued.",
        file=None,
    )

    def fit_with_heartbeat(cache_hit, label, key_kind, fit):
        """Run one cached reduced fit while reporting that it is alive."""

        started = perf_counter()
        stop_heartbeat = threading.Event()

        def heartbeat():
            while not stop_heartbeat.wait(10):
                sensitivity_progress.set_postfix(
                    model="cache hit" if cache_hit else "estimating",
                    elapsed=f"{perf_counter() - started:.0f}s",
                    refresh=True,
                )

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            return cached_reduced_fit(
                estimate_cache, cache_context, *key_kind, fit
            )
        finally:
            stop_heartbeat.set()
            thread.join(timeout=1)

    for (dataset, outcome), bundle in estimates.items():
        if (outcome in {"SomeRiskHome", "VeryHighRiskHome"} and dataset != "HH"):
            continue
        if outcome == "diarrhea" and dataset != "U5":
            continue
        groups = select_column_groups(
            bundle["irm_x_cluster"], SENSITIVITY_GROUPS, bundle["child"]
        )
        for specification, clustered in [
            ("clustered_folds", True),
            ("unclustered", False),
        ]:
            irm = bundle["irm_cluster"] if clustered else bundle["irm_no_cluster"]
            apos = bundle["apos_cluster"] if clustered else bundle["apos_no_cluster"]
            irm_frame = (
                bundle["irm_frame_cluster"]
                if clustered else bundle["irm_frame_no_cluster"]
            )
            irm_x = bundle["irm_x_cluster"] if clustered else bundle["irm_x_no_cluster"]
            apos_frame = (
                bundle["apos_frame_cluster"]
                if clustered else bundle["apos_frame_no_cluster"]
            )
            apos_x = (
                bundle["apos_x_cluster"]
                if clustered else bundle["apos_x_no_cluster"]
            )

            irm_rv, irm_rva = sensitivity_params(irm)
            apos_contrast = apos.causal_contrast(reference_levels=[0])
            apos_cluster_ids = (
                bundle["apos_frame_cluster"]["Cluster_var"].to_numpy()
                if clustered else None
            )
            apos_rv, apos_rva = sensitivity_params(
                apos_contrast,
                cluster_ids=apos_cluster_ids,
                smpls=apos.smpls if clustered else None,
            )
            apos_summary = apos_contrast.summary.reset_index(drop=True)
            # Store the same APOS SE used by the sensitivity calculation.
            apos_sensitivity_se = (
                cluster_robust_framework_se(
                    apos_contrast,
                    apos_cluster_ids,
                    apos.smpls,
                )
                if clustered else apos_summary["std err"].to_numpy()
            )

            # Each iteration removes one observed control group from both
            # nuisance-model matrices while preserving the original folds.
            for group, group_columns in groups.items():
                keep_irm_x = [c for c in irm_x if c not in group_columns]
                keep_apos_x = [c for c in apos_x if c not in group_columns]
                if not keep_irm_x or not keep_apos_x:
                    continue

                def fit_short_irm(
                    keep_irm_x=keep_irm_x,
                    irm=irm,
                    clustered=clustered,
                ):
                    data = dml.DoubleMLData(
                        irm_frame,
                        y_col=outcome,
                        d_cols="water_treatment",
                        x_cols=keep_irm_x,
                        cluster_cols="Cluster_var" if clustered else None,
                    )
                    model = dml.DoubleMLIRM(
                        data,
                        clone(ml_g_cluster if clustered else ml_g),
                        clone(ml_m_cluster if clustered else ml_m),
                        n_folds=folds,
                        n_rep=repetitions,
                        draw_sample_splitting=False,
                    )
                    if clustered:
                        model.set_sample_splitting(
                            irm.smpls, irm.smpls_cluster
                        )
                    else:
                        model.set_sample_splitting(irm.smpls)
                    return model.fit(
                        n_jobs_cv=jobs_cv, store_predictions=False
                    )

                def fit_short_apos(
                    keep_apos_x=keep_apos_x,
                    apos=apos,
                ):
                    data = dml.DoubleMLData(
                        apos_frame,
                        y_col=outcome,
                        d_cols="WQ15_g",
                        x_cols=keep_apos_x,
                    )
                    model = dml.DoubleMLAPOS(
                        data,
                        clone(ml_g_cluster if clustered else ml_g),
                        clone(ml_m_cluster if clustered else ml_m),
                        treatment_levels=list(treatment_levels),
                        n_folds=folds,
                        n_rep=repetitions,
                        draw_sample_splitting=False,
                    )
                    model.set_sample_splitting(apos.smpls)
                    return model.fit(
                        n_jobs_models=1,
                        n_jobs_cv=jobs_cv,
                        store_predictions=False,
                    )

                # Reduced fits are the only additional model estimations in
                # this stage; the cache avoids repeating them across runs.
                irm_cache_hit = estimate_cache.contains(make_estimate_key(
                    cache_context,
                    dataset,
                    outcome,
                    f"sensitivity_IRM_{specification}_{group}",
                    specification == "clustered_folds",
                ))
                sensitivity_progress.set_description(
                    f"Sensitivity {dataset} {outcome}: {specification} {group} IRM"
                )
                sensitivity_progress.set_postfix(
                    model="cache hit" if irm_cache_hit else "estimating",
                    refresh=True,
                )
                tqdm.write(
                    f"Sensitivity: starting {dataset} {outcome} | "
                    f"{specification} | {group} | IRM | "
                    f"{'cache hit' if irm_cache_hit else 'new fit'}",
                    file=None,
                )
                short_irm = fit_with_heartbeat(
                    irm_cache_hit,
                    "IRM",
                    (dataset, outcome, "IRM", specification, group),
                    fit_short_irm,
                )
                sensitivity_progress.update(1)
                tqdm.write(
                    f"Sensitivity: finished {dataset} {outcome} | "
                    f"{specification} | {group} | IRM",
                    file=None,
                )
                apos_cache_hit = estimate_cache.contains(make_estimate_key(
                    cache_context,
                    dataset,
                    outcome,
                    f"sensitivity_APOS_{specification}_{group}",
                    specification == "clustered_folds",
                ))
                sensitivity_progress.set_description(
                    f"Sensitivity {dataset} {outcome}: {specification} {group} APOS"
                )
                sensitivity_progress.set_postfix(
                    model="cache hit" if apos_cache_hit else "estimating",
                    refresh=True,
                )
                tqdm.write(
                    f"Sensitivity: starting {dataset} {outcome} | "
                    f"{specification} | {group} | APOS | "
                    f"{'cache hit' if apos_cache_hit else 'new fit'}",
                    file=None,
                )
                short_apos = fit_with_heartbeat(
                    apos_cache_hit,
                    "APOS",
                    (dataset, outcome, "APOS", specification, group),
                    fit_short_apos,
                )
                sensitivity_progress.update(1)
                tqdm.write(
                    f"Sensitivity: finished {dataset} {outcome} | "
                    f"{specification} | {group} | APOS",
                    file=None,
                )
                short_irm_rv, short_irm_rva = sensitivity_params(short_irm)
                short_apos_contrast = short_apos.causal_contrast(
                    reference_levels=[0]
                )
                short_apos_rv, short_apos_rva = sensitivity_params(
                    short_apos_contrast,
                    cluster_ids=apos_cluster_ids,
                    smpls=short_apos.smpls if clustered else None,
                )

                base = {
                    "dataset": dataset,
                    "outcome": outcome,
                    "learner": "stacked",
                    "specification": specification,
                    "group": group,
                    "group_label": SENSITIVITY_LABELS[group],
                }
                rows.append({
                    **base,
                    "method": "IRM",
                    "treatment": "Any Treatment",
                    "rv_q": float(irm_rv[0]),
                    "rv_qa": float(irm_rva[0]),
                    "rv_q_without": float(short_irm_rv[0]),
                    "rv_qa_without": float(short_irm_rva[0]),
                })

                for i, level in enumerate((1, 2, 3)):
                    rows.append({
                        **base,
                        "method": "APOS",
                        "treatment": treatment_labels[level],
                        "rv_q": float(apos_rv[i]),
                        "rv_qa": float(apos_rva[i]),
                        "rv_q_without": float(short_apos_rv[i]),
                        "rv_qa_without": float(short_apos_rva[i]),
                        "ate": float(apos_summary.iloc[i]["coef"]),
                        "se": float(apos_sensitivity_se[i]),
                    })

    sensitivity_progress.close()
    return pd.DataFrame(rows)


def run_sensitivity_benchmark(estimates):
    """Benchmark the observed source-water E.coli decile block.

    DoubleML refits the short model internally after removing the named
    benchmarking set. The returned statistics quantify the predictive gain
    from the observed block; they are complementary to, and not substitutes
    for, the native RV/RV-alpha analysis of unobserved confounding.
    """

    rows = []
    jobs = []
    for (dataset, outcome), bundle in estimates.items():
        if (dataset == "HH" and outcome not in {"SomeRiskHome", "VeryHighRiskHome"}) or (
            dataset == "U5" and outcome != "diarrhea"
        ):
            continue
        for specification, clustered in (("clustered_folds", True), ("unclustered", False)):
            model_names = (
                ("IRM", bundle["irm_cluster" if clustered else "irm_no_cluster"],
                 bundle["irm_x_cluster" if clustered else "irm_x_no_cluster"]),
                ("APOS", bundle["apos_cluster" if clustered else "apos_no_cluster"],
                 bundle["apos_x_cluster" if clustered else "apos_x_no_cluster"]),
            )
            for method, model, x_columns in model_names:
                benchmark_set = select_column_groups(
                    x_columns, SENSITIVITY_GROUPS, bundle["child"]
                ).get("source_ecoli", [])
                jobs.append((dataset, outcome, specification, method, model, benchmark_set))

    progress = tqdm(
        total=len(jobs), desc="Sensitivity: observed benchmark", unit="model",
        dynamic_ncols=True, leave=True,
    )
    tqdm.write(
        f"Sensitivity benchmark: {len(jobs)} models queued; "
        "benchmark set = source_ecoli_*.", file=None,
    )
    for dataset, outcome, specification, method, model, benchmark_set in jobs:
        progress.set_description(
            f"Sensitivity benchmark {dataset} {outcome}: {specification} {method}"
        )
        if not benchmark_set:
            raise ValueError(f"No source_ecoli_* columns found for {method} {dataset} {outcome}.")
        # RV and RV-alpha answer the unobserved-confounding question for the
        # complete specification. The benchmark below is a separate observed-
        # control reference and must not be confused with either RV.
        if method == "IRM":
            full_rv, full_rva = sensitivity_params(model)
        else:
            contrast = model.causal_contrast(reference_levels=[0])
            full_rv, full_rva = sensitivity_params(
                contrast,
                cluster_ids=(
                    estimates[(dataset, outcome)]["apos_frame_cluster"]["Cluster_var"].to_numpy()
                    if specification == "clustered_folds" else None
                ),
                smpls=model.smpls if specification == "clustered_folds" else None,
            )
        benchmark = model.sensitivity_benchmark(
            benchmarking_set=list(benchmark_set),
            fit_args={"n_jobs_cv": jobs_cv, "n_jobs_models": 1}
            if method == "APOS" else {"n_jobs_cv": jobs_cv},
        )
        # APOS benchmark output is indexed by all treatment levels, including
        # the reference level 0.  The causal contrast sensitivity arrays omit
        # that reference level and therefore contain only [1, 2, 3, 98].
        # Match by treatment code instead of assuming that the row number is
        # already the contrast number.
        apos_contrast_levels = [1, 2, 3, 98]
        for effect_index, (treatment, values) in enumerate(benchmark.iterrows()):
            if method == "IRM":
                parameter_index = 0
            else:
                try:
                    treatment_code = int(str(treatment).split()[0])
                except (TypeError, ValueError):
                    # Fallback for an older DoubleML index without numeric
                    # treatment labels: the reference level is the first row.
                    parameter_index = effect_index - (1 if len(benchmark) == len(full_rv) + 1 else 0)
                else:
                    if treatment_code not in apos_contrast_levels:
                        # The reference treatment has no APOS contrast and
                        # must not be written as if it were an estimated effect.
                        continue
                    parameter_index = apos_contrast_levels.index(treatment_code)
            if parameter_index >= len(full_rv):
                raise ValueError(
                    f"Could not align {method} benchmark row {treatment!r} "
                    f"with {len(full_rv)} sensitivity parameters."
                )
            rows.append({
                "dataset": dataset,
                "outcome": outcome,
                "specification": specification,
                "method": method,
                "treatment": str(treatment),
                "benchmark_set": "source_ecoli_*",
                "cf_y": float(values["cf_y"]),
                "cf_d": float(values["cf_d"]),
                "rho": float(values["rho"]),
                "delta_theta": float(values["delta_theta"]),
                "rv": float(full_rv[parameter_index]),
                "rva": float(full_rva[parameter_index]),
            })
        progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


# Observed-control benchmark based on DoubleML's native benchmarking API.
# Keep this separate from the fitted-model cache: the benchmark is a
# post-estimation object, but it is still expensive because DoubleML refits
# the short model internally for every outcome, method, and fold specification.
# Its key depends on the estimation/data context and on this benchmark stage,
# while presentation-only changes remain cache-safe.
sensitivity_cache_context = {
    **cache_context,
    "sensitivity_benchmark_cache_version": 1,
}
sensitivity_cache_key = make_estimate_key(
    sensitivity_cache_context,
    "all",
    "all",
    "sensitivity_benchmark",
    False,
)
sensitivity_checkpoint = checkpoint_dir / f"sensitivity_benchmark_{sensitivity_cache_key}.pkl"

if sensitivity_checkpoint.exists():
    try:
        sensitivity_benchmark_results = pd.read_pickle(sensitivity_checkpoint)
        if not isinstance(sensitivity_benchmark_results, pd.DataFrame):
            raise TypeError("Sensitivity checkpoint is not a DataFrame.")
        print(f"Sensitivity benchmark checkpoint loaded: {sensitivity_checkpoint}")
    except (OSError, EOFError, ValueError, TypeError, ImportError) as exc:
        print(f"Sensitivity benchmark checkpoint invalid; recomputing ({exc}).")
        sensitivity_benchmark_results = run_sensitivity_benchmark(estimates)
        sensitivity_benchmark_results.to_pickle(sensitivity_checkpoint)
else:
    sensitivity_benchmark_results = run_sensitivity_benchmark(estimates)
    sensitivity_benchmark_results.to_pickle(sensitivity_checkpoint)
    print(f"Sensitivity benchmark checkpoint saved: {sensitivity_checkpoint}")

# Keep the stable, human-facing output path used by downstream tables/scripts.
sensitivity_benchmark_results.to_pickle(
    output_dir / "results_sensitivity_benchmark.pkl"
)
sensitivity_benchmark_table_paths = create_benchmark_sensitivity_tables(
    sensitivity_benchmark_results,
    filename_prefix="table_sensitivity_benchmark",
    output_dir=output_dir,
)
# Keep the publication-layer interface stable. The regression tables do not
# consume sensitivity rows directly, but they still receive this object.
sensitivity_results = sensitivity_benchmark_results
relative_table_paths = sensitivity_benchmark_table_paths


# ---------------------------------------------------------------------------
# Prespecified heterogeneity projections
# ---------------------------------------------------------------------------
# These are post-estimation GATE projections of the existing stacked APOS
# contrasts.  They do not refit the outcome or propensity nuisance learners.
HETEROGENEITY_GROUPS = {
    "source_ecoli": {
        "column": "wq27_decile",
        "label": "Initial source-water E. coli decile",
        "labels": {},
    },
}

# RiskSource is needed only for the post-estimation heterogeneity projection.
# It is intentionally loaded lazily because the main estimation input keeps a
# narrow set of columns and the fitted-model cache should not depend on this
# presentation-only variable.
_heterogeneity_source_cache = {}


def prepare_heterogeneity_groups(
    data, outcome, child, keep_cluster, treatment_column, allowed_levels
):
    """Recreate the fitted sample order and attach source-water E. coli deciles.

    The GATE analysis is intentionally restricted to the observed initial
    source-water contamination deciles used in the main controls. The narrow
    model frames are rebuilt in the same row order so the decile labels align
    with the already-fitted influence scores.
    """

    controls = [
        "windex5", "urban", "WS1_g", "wq27_decile", "Any_U5",
        "Girls_less_than15", "Boys_15or_less", "Toilet",
    ]
    required = [outcome, treatment_column, "country_cat", *controls]
    if keep_cluster:
        required.append("Cluster_var")
    if child:
        required.extend(["age", "male"])
    keep = data[required].notna().all(axis=1)
    keep &= data[treatment_column].isin(allowed_levels)

    keep &= data["wq27_decile"].notna()
    groups = data.loc[keep, ["wq27_decile"]].copy()
    return groups.reset_index(drop=True)


def run_heterogeneity_projections(estimates):
    """Project existing IRM/APOS scores onto prespecified risk groups.

    The IRM framework supplies the binary any-treatment score. APOS supplies
    treatment-specific contrasts relative to level zero. Neither branch
    refits nuisance learners; the function only estimates descriptive GATE
    projections and their clustered or ordinary standard errors.
    """

    rows = []
    projection_specs = [
        (dataset, outcome, specification)
        for dataset, _, _, outcome in analysis_specs
        if not (
            (dataset == "HH" and outcome not in {"SomeRiskHome", "VeryHighRiskHome"})
            or (dataset == "U5" and outcome != "diarrhea")
        )
        for specification in ("clustered_folds", "unclustered")
    ]
    heterogeneity_progress = tqdm(
        total=len(projection_specs) * len(HETEROGENEITY_GROUPS),
        desc="Heterogeneity: GATE projections",
        unit="group",
        dynamic_ncols=True,
        leave=True,
    )
    tqdm.write(
        "Heterogeneity: "
        f"{len(projection_specs) * len(HETEROGENEITY_GROUPS)} projections queued.",
        file=None,
    )
    for dataset, data, child, outcome in analysis_specs:
        if dataset == "HH" and outcome not in {"SomeRiskHome", "VeryHighRiskHome"}:
            continue
        if dataset == "U5" and outcome != "diarrhea":
            continue

        bundle = estimates[(dataset, outcome)]
        for specification, clustered in [
            ("clustered_folds", True),
            ("unclustered", False),
        ]:
            apos_groups = prepare_heterogeneity_groups(
                data, outcome, child, keep_cluster=clustered,
                treatment_column="WQ15_g", allowed_levels=treatment_levels,
            )
            irm_groups = prepare_heterogeneity_groups(
                data, outcome, child, keep_cluster=clustered,
                treatment_column="water_treatment", allowed_levels=(0, 1),
            )
            apos_frame = (
                bundle["apos_frame_cluster"]
                if clustered else bundle["apos_frame_no_cluster"]
            )
            irm_frame = (
                bundle["irm_frame_cluster"]
                if clustered else bundle["irm_frame_no_cluster"]
            )
            if len(apos_groups) != len(apos_frame) or len(irm_groups) != len(irm_frame):
                raise ValueError(
                    f"Heterogeneity-group rows are not aligned for {dataset} {outcome} "
                    f"({specification})."
                )
            apos = (
                bundle["apos_cluster"]
                if clustered else bundle["apos_no_cluster"]
            )
            contrast = apos.causal_contrast(reference_levels=[0])
            cluster_ids = (
                apos_frame["Cluster_var"].to_numpy() if clustered else None
            )
            irm_cluster_ids = (
                irm_frame["Cluster_var"].to_numpy() if clustered else None
            )
            irm = bundle["irm_cluster"] if clustered else bundle["irm_no_cluster"]
            # IRM itself is already the binary-treatment framework.  GATE
            # projection needs its framework scores, whereas APOS requires an
            # explicit causal contrast first.
            irm_framework = irm.framework

            for group_name, spec in HETEROGENEITY_GROUPS.items():
                heterogeneity_progress.set_description(
                    f"Heterogeneity {dataset} {outcome}: "
                    f"{specification} {group_name}"
                )
                # Use human-readable labels while retaining the original
                # decile values for the projection design matrix.
                group_labels = {
                    str(value): f"Decile {value}"
                    for value in pd.Series(
                        apos_groups[spec["column"]]
                    ).dropna().unique()
                }
                # IRM contributes the binary any-treatment effect.
                irm_result = estimate_gate_from_contrast(
                    irm_framework,
                    "Any Treatment",
                    0,
                    irm_groups[spec["column"]],
                    irm_cluster_ids,
                    group_labels=group_labels,
                )
                irm_result.insert(0, "dataset", dataset)
                irm_result.insert(1, "outcome", outcome)
                irm_result.insert(2, "method", "IRM stacked")
                irm_result.insert(3, "specification", specification)
                irm_result.insert(4, "group", group_name)
                irm_result.insert(5, "heterogeneity_label", spec["label"])
                irm_result.insert(6, "treatment_label", "Any Treatment")
                rows.append(irm_result)

                # APOS contributes the three method-specific contrasts.
                values = apos_groups[spec["column"]]
                for effect_index, treatment_level in enumerate((1, 2, 3)):
                    result = estimate_gate_from_contrast(
                        contrast,
                        treatment_level,
                        effect_index,
                        values,
                        cluster_ids,
                        group_labels=group_labels,
                    )
                    result.insert(0, "dataset", dataset)
                    result.insert(1, "outcome", outcome)
                    result.insert(2, "method", "APOS stacked")
                    result.insert(3, "specification", specification)
                    result.insert(4, "group", group_name)
                    result.insert(5, "heterogeneity_label", spec["label"])
                    result.insert(
                        6,
                        "treatment_label",
                        {1: "Boiling", 2: "Chlorination/tablets", 3: "Straining/settling"}
                        [treatment_level],
                    )
                    rows.append(result)
                heterogeneity_progress.update(1)
                heterogeneity_progress.set_postfix(
                    status="completed", refresh=True
                )

    heterogeneity_progress.close()
    tqdm.write("Heterogeneity: all GATE projections completed.", file=None)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


heterogeneity_results = run_heterogeneity_projections(estimates)
heterogeneity_results.to_pickle(
    output_dir / "results_heterogeneity_gates.pkl"
)
heterogeneity_table_paths = []
heterogeneity_table_paths.extend(create_heterogeneity_comparison_tables(
    heterogeneity_results,
    output_dir=output_dir,
    filename_prefix="table_heterogeneity_main",
    specifications=("clustered_folds",),
))
heterogeneity_table_paths.extend(create_heterogeneity_comparison_tables(
    heterogeneity_results,
    output_dir=output_dir,
    filename_prefix="table_heterogeneity_appendix",
    specifications=("clustered_folds", "unclustered"),
))


# ---------------------------------------------------------------------------
# Publication outputs
# ---------------------------------------------------------------------------
# The publication layer consumes the fitted objects and sensitivity results;
# it does not estimate new models.



irm_table_parts = []
apos_table_parts = []

for (dataset_name, outcome), models in estimates.items():
    prefix = f"{dataset_name.lower()}_{outcome}"

    apos_cluster_all = models["apos_cluster"].causal_contrast(
        reference_levels=[0]
    ).summary
    apos_no_cluster_all = models["apos_no_cluster"].causal_contrast(
        reference_levels=[0]
    ).summary

    # APOS estimates all levels, including 98, but reports only 0-3.
    expected_contrasts = len(treatment_levels) - 1
    if len(apos_cluster_all) != expected_contrasts:
        raise AssertionError("APOS did not estimate every treatment level.")
    if len(apos_no_cluster_all) != expected_contrasts:
        raise AssertionError("APOS did not estimate every treatment level.")

    apos_cluster_report = apos_cluster_all.iloc[:len(report_levels) - 1].copy()
    apos_cluster_report["std err"] = models["apos_cluster_se"][:len(apos_cluster_report)]
    apos_cluster_report["t"] = (
        apos_cluster_report["coef"] / apos_cluster_report["std err"]
    )
    apos_cluster_report["P>|t|"] = 2 * norm.sf(apos_cluster_report["t"].abs())
    apos_cluster_report["2.5 %"] = (
        apos_cluster_report["coef"] - 1.96 * apos_cluster_report["std err"]
    )
    apos_cluster_report["97.5 %"] = (
        apos_cluster_report["coef"] + 1.96 * apos_cluster_report["std err"]
    )
    apos_no_cluster_report = apos_no_cluster_all.iloc[:len(report_levels) - 1]
    for specification, summary in [
        ("clustered", models["irm_cluster"].summary),
        ("unclustered", models["irm_no_cluster"].summary),
    ]:
        table = summary.reset_index()
        table.insert(0, "dataset", dataset_name)
        table.insert(1, "outcome", outcome)
        table.insert(2, "specification", specification)
        irm_table_parts.append(table)

    for specification, summary in [
        ("clustered_folds", apos_cluster_report),
        ("unclustered", apos_no_cluster_report),
    ]:
        table = summary.reset_index()
        table = table.rename(columns={table.columns[3]: "contrast"})
        table.insert(0, "dataset", dataset_name)
        table.insert(1, "outcome", outcome)
        table.insert(2, "specification", specification)
        apos_table_parts.append(table)

    print(f"\n{dataset_name} — {outcome} — IRM clustered:")
    print(models["irm_cluster"].summary)
    print(f"\n{dataset_name} — {outcome} — IRM unclustered:")
    print(models["irm_no_cluster"].summary)
    print(f"\n{dataset_name} — {outcome} — APOS clustered folds, reported levels 0-3:")
    print(apos_cluster_report)
    print(f"\n{dataset_name} — {outcome} — APOS unclustered folds, reported levels 0-3:")
    print(apos_no_cluster_report)


irm_table = pd.concat(irm_table_parts, ignore_index=True)
apos_table = pd.concat(apos_table_parts, ignore_index=True)

table_paths = {
    "main": str(write_stacked_regression_table(
        output_dir,
        estimates,
        sensitivity_results,
        ["SomeRiskHome", "VeryHighRiskHome", "diarrhea"],
        "table_water_treatment_main.tex",
        "Stacked water-treatment effects",
        "tab:water-treatment-main",
        report_levels,
        folds,
        repetitions,
        specifications=("clustered",),
    )),
    "appendix": str(write_stacked_regression_table(
        output_dir,
        estimates,
        sensitivity_results,
        ["SomeRiskHome", "VeryHighRiskHome", "diarrhea"],
        "table_water_treatment_appendix.tex",
        "Stacked water-treatment effects: clustered and ordinary folds",
        "tab:water-treatment-appendix",
        report_levels,
        folds,
        repetitions,
        specifications=("clustered", "unclustered"),
    )),
}

cache_manifest = {
    "cache_policy": (
        "Estimation source and data changes invalidate checkpoints; "
        "presentation code after '# Results' does not."
    ),
    "cache_schema_version": cache_context["cache_schema_version"],
    "source_sha256": cache_context["source_sha256"],
    "data_sha256": cache_context["data_sha256"],
    "checkpoint_dir": str(checkpoint_dir),
    "analysis_specs": [
        {"dataset": dataset, "outcome": outcome}
        for dataset, _, _, outcome in analysis_specs
    ],
    "treatment_levels_estimated": list(treatment_levels),
    "treatment_levels_reported": list(report_levels),
    "tables": [
        *table_paths.values(),
        str(output_dir / "results_sensitivity_benchmark.pkl"),
        str(output_dir / "results_heterogeneity_gates.pkl"),
        *[str(path) for path in heterogeneity_table_paths],
        *[str(path) for path in relative_table_paths],
    ],
}
(output_dir / "cache_manifest.json").write_text(
    json.dumps(cache_manifest, indent=2),
    encoding="utf-8",
)
