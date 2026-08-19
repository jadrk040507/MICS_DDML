"""Readable MICS DoubleML pipeline.

The file is organized like an applied-econometrics do-file. Read the numbered
sections from top to bottom. The helper functions only avoid repeating the
same technical code for the household and under-five samples.

Outputs:
    Output/checkpoints/*.pkl            fitted IRM/APOS models
    Output/results_*.pkl                result data frames
    Output/table_*.tex                 LaTeX tables
"""

from pathlib import Path
from copy import deepcopy
import gc
import json
import os

import doubleml as dml
import joblib
import numpy as np
import pandas as pd
from joblib import parallel_backend
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
from xgboost import XGBClassifier, XGBRegressor
from _functions import (
    LazyCheckpointBundle,
    cluster_robust_framework_se,
    estimate_gate_from_contrast,
    sensitivity_params,
)
from _tables import (
    create_benchmark_sensitivity_tables,
    create_heterogeneity_comparison_tables,
    write_sensitivity_summary_table,
    write_publication_table,
    write_super_learner_weights_tables,
)


# ============================================================
# 1. Analysis choices and paths
# ============================================================

SEED = 42
# Full analysis is the default. Set MICS_SAMPLED=1 only for a smoke test.
SAMPLED = os.environ.get("MICS_SAMPLED", "0").lower() in {
    "1", "true", "yes", "on"
}
SAMPLE_FRAC = 0.05
SAMPLE_SEED = SEED

FOLDS = 2 if SAMPLED else 5
REPETITIONS = 1 if SAMPLED else 3
INNER_FOLDS = 2 if SAMPLED else 3


def _env_int(name, default):
    """Read a positive integer environment override."""

    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


# Keep parallelism explicit. APOS can parallelize across treatment-level
# models, while RF/XGBoost can parallelize inside each nuisance fit. Running
# both levels wide at the same time is the easiest way to exhaust RAM.
APOS_WORKERS = _env_int("MICS_APOS_WORKERS", 1)
LEARNER_JOBS = _env_int("MICS_LEARNER_JOBS", 1 if SAMPLED else 2)

TREATMENT_LEVELS = (0, 1, 2, 3, 98)
REPORTED_LEVELS = (0, 1, 2, 3)

PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "Data" / "3. Final"
OUTPUT_DIR = PROJECT / "Output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
TABLE_DIR = OUTPUT_DIR
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_TAG = (
    f"_sample{int(SAMPLE_FRAC * 100):02d}{'_fast2' if SAMPLED else ''}"
    if SAMPLED else ""
)
HEAVY_APOS_CHECKPOINT_BYTES = 512 * 1024 * 1024


# ============================================================
# 2. Variables and data preparation
# ============================================================

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


def make_frame(data, outcome, treatment, child=False, cluster=True,
               allowed_levels=None):
    """Create the model frame and dummy-variable matrix for one specification."""

    controls = list(COMMON_CONTROLS)
    if child:
        controls += ["age", "male"]

    required = [outcome, treatment, *controls]
    if "country_cat" in data.columns:
        required.append("country_cat")
    if cluster:
        required.append("Cluster_var")

    frame = data[required].copy()
    frame = frame.dropna()
    if allowed_levels is not None:
        frame = frame[frame[treatment].isin(allowed_levels)].copy()

    categorical = ["windex5", "WS1_g", "wq27_decile", "Toilet"]
    if "country_cat" in frame.columns:
        categorical.append("country_cat")
    numeric = [v for v in controls if v not in categorical]

    x = pd.get_dummies(frame[numeric + categorical], drop_first=True, dtype=float)
    x = x.astype(float).reset_index(drop=True)
    frame = frame.reset_index(drop=True)

    # This column is only metadata for the convex learner's inner folds. The
    # ConvexRegressor/ConvexClassifier remove it before fitting any learner,
    # so Cluster_var is never used as a causal predictor.
    if cluster:
        x["_cluster_model_code"] = pd.factorize(
            frame["Cluster_var"], sort=True
        )[0].astype(float)

    columns = [outcome, treatment]
    if cluster:
        columns.append("Cluster_var")
    model_frame = pd.concat([frame[columns], x], axis=1)
    return model_frame, list(x.columns)


# ============================================================
# 3. Technical helpers — usually do not edit
# ============================================================

def convex_weights(predictions, target, classification=False):
    """Choose non-negative weights that sum to one using OOF predictions."""

    n_learners = predictions.shape[1]
    initial = np.repeat(1 / n_learners, n_learners)

    def loss(weights):
        fitted = predictions @ weights
        if classification:
            fitted = np.clip(fitted, 1e-8, 1 - 1e-8)
            return -np.mean(
                target * np.log(fitted) + (1 - target) * np.log(1 - fitted)
            )
        return np.mean((target - fitted) ** 2)

    result = minimize(
        loss,
        initial,
        method="SLSQP",
        bounds=[(0, 1)] * n_learners,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )
    if result.success:
        weights = np.clip(result.x, 0, 1)
        return weights / weights.sum()

    # If optimization fails, keep the single learner with the lowest loss.
    losses = [loss(np.eye(n_learners)[i]) for i in range(n_learners)]
    weights = np.zeros(n_learners)
    weights[int(np.argmin(losses))] = 1
    return weights


def _positive_probability(model, x):
    """Return the probability of treatment 1."""

    positive = int(np.where(np.asarray(model.classes_) == 1)[0][0])
    return model.predict_proba(x)[:, positive]


class ConvexRegressor(RegressorMixin, BaseEstimator):
    """Convex combination with cluster-aware inner folds when requested.

    ``group_column=-1`` means the last design-matrix column is metadata for
    grouped inner cross-fitting. It is removed before OLS, LASSO, Elastic Net,
    random forest, and XGBoost are fitted, so the cluster code is never a
    predictor.
    """

    def __init__(self, estimators, random_state=42, group_column=None):
        self.estimators = estimators
        self.random_state = random_state
        self.group_column = group_column

    def _features_and_groups(self, x):
        x = np.asarray(x, dtype=float)
        if self.group_column is None:
            return x, None
        index = self.group_column if self.group_column >= 0 else x.shape[1] + self.group_column
        groups = x[:, index].astype(int)
        return np.delete(x, index, axis=1), groups

    def fit(self, x, y):
        x, groups = self._features_and_groups(x)
        y = np.asarray(y, dtype=float)
        if groups is None:
            splits = KFold(
                INNER_FOLDS, shuffle=True, random_state=self.random_state
            ).split(x)
        else:
            splits = GroupKFold(INNER_FOLDS).split(x, y, groups)
        splits = list(splits)
        oof = np.zeros((len(y), len(self.estimators)))
        self.models_ = []

        for j, (name, estimator) in enumerate(self.estimators):
            fold_models = []
            for train, test in splits:
                fitted = clone(estimator).fit(x[train], y[train])
                oof[test, j] = fitted.predict(x[test])
                fold_models.append(fitted)
            self.models_.append((name, fold_models))

        self.weights_ = convex_weights(oof, y, classification=False)
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, x):
        x, _ = self._features_and_groups(x)
        predictions = np.column_stack([
            np.mean([m.predict(x) for m in models], axis=0)
            for _, models in self.models_
        ])
        return predictions @ self.weights_


class ConvexClassifier(ClassifierMixin, BaseEstimator):
    """Convex treatment ensemble with optional cluster-aware inner folds."""

    def __init__(self, estimators, random_state=42, group_column=None):
        self.estimators = estimators
        self.random_state = random_state
        self.group_column = group_column

    def _features_and_groups(self, x):
        x = np.asarray(x, dtype=float)
        if self.group_column is None:
            return x, None
        index = self.group_column if self.group_column >= 0 else x.shape[1] + self.group_column
        groups = x[:, index].astype(int)
        return np.delete(x, index, axis=1), groups

    def fit(self, x, y):
        x, groups = self._features_and_groups(x)
        y = np.asarray(y, dtype=int)
        if groups is None:
            splits = StratifiedKFold(
                INNER_FOLDS, shuffle=True, random_state=self.random_state
            ).split(x, y)
        else:
            splits = StratifiedGroupKFold(
                INNER_FOLDS, shuffle=True, random_state=self.random_state
            ).split(x, y, groups)
        splits = list(splits)
        oof = np.zeros((len(y), len(self.estimators)))
        self.models_ = []

        for j, (name, estimator) in enumerate(self.estimators):
            fold_models = []
            for train, test in splits:
                fitted = clone(estimator).fit(x[train], y[train])
                oof[test, j] = _positive_probability(fitted, x[test])
                fold_models.append(fitted)
            self.models_.append((name, fold_models))

        self.weights_ = convex_weights(oof, y, classification=True)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = x.shape[1]
        return self

    def predict_proba(self, x):
        x, _ = self._features_and_groups(x)
        predictions = np.column_stack([
            np.mean([_positive_probability(m, x) for m in models], axis=0)
            for _, models in self.models_
        ])
        positive = np.clip(predictions @ self.weights_, 1e-8, 1 - 1e-8)
        return np.column_stack([1 - positive, positive])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


# ============================================================
# 4. All learners used by the convex ensemble
# ============================================================

REGRESSORS = [
    ("ols", LinearRegression()),
    ("lasso", Pipeline([
        ("scale", StandardScaler()),
        ("model", LassoCV(cv=3, max_iter=3000, random_state=SEED)),
    ])),
    ("elastic_net", Pipeline([
        ("scale", StandardScaler()),
        ("model", ElasticNetCV(
            cv=3,
            l1_ratio=(0.25, 0.5, 0.75),
            max_iter=3000,
            random_state=SEED,
        )),
    ])),
    ("random_forest", RandomForestRegressor(
        n_estimators=150, max_depth=15, min_samples_leaf=5,
        random_state=SEED, n_jobs=LEARNER_JOBS,
    )),
    ("xgboost", XGBRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=SEED, n_jobs=LEARNER_JOBS,
        eval_metric="rmse", verbosity=0,
    )),
]

CLASSIFIERS = [
    ("logit", LogisticRegression(max_iter=2000)),
    ("lasso", LogisticRegressionCV(
        cv=3, l1_ratios=(1,), solver="liblinear", max_iter=2000,
        scoring="neg_log_loss", random_state=SEED,
        use_legacy_attributes=False,
    )),
    ("elastic_net", LogisticRegressionCV(
        cv=3, solver="saga", l1_ratios=(0.5,),
        max_iter=3000, scoring="neg_log_loss", random_state=SEED,
        use_legacy_attributes=False,
    )),
    ("random_forest", RandomForestClassifier(
        n_estimators=150, max_depth=15, min_samples_leaf=5,
        random_state=SEED, n_jobs=LEARNER_JOBS,
    )),
    ("xgboost", XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=SEED, n_jobs=LEARNER_JOBS,
        eval_metric="logloss", verbosity=0,
    )),
]

# The sampled run is a smoke test of the complete pipeline, not a model
# comparison. Keep two learners so the convex ensemble is exercised without
# spending minutes on every saga/ElasticNet fold.
if SAMPLED:
    REGRESSORS = REGRESSORS[:2]
    CLASSIFIERS = CLASSIFIERS[:2]


# ============================================================
# 5. Checkpoints and estimation functions
# ============================================================

def _is_apos_checkpoint(name):
    """Return True for APOS checkpoints, including the iid specification."""

    return "_APOS" in name


def _move_legacy_heavy_checkpoint(path):
    """Move an old full-model APOS checkpoint aside before it can exhaust RAM."""

    legacy_path = path.with_name(f"{path.stem}.legacy{path.suffix}")
    counter = 1
    while legacy_path.exists():
        legacy_path = path.with_name(
            f"{path.stem}.legacy{counter}{path.suffix}"
        )
        counter += 1
    path.rename(legacy_path)
    print(
        f"Moved oversized legacy checkpoint aside: "
        f"{path.name} -> {legacy_path.name}",
        flush=True,
    )


def checkpoint_ready_to_load(name):
    """Return True when a checkpoint can be loaded without known RAM hazards."""

    path = checkpoint_path(name)
    if (
        path.exists()
        and _is_apos_checkpoint(name)
        and path.stat().st_size > HEAVY_APOS_CHECKPOINT_BYTES
    ):
        _move_legacy_heavy_checkpoint(path)
    return path.exists()


def compact_fitted_model(model):
    """Remove fitted nuisance learners after their convex weights are saved."""

    model.convex_weights = collect_convex_weights(model)

    # APOS stores one DoubleML model per treatment level in ``_modellist``.
    # Each of those submodels can carry fitted nuisance learners even when the
    # APOS wrapper's public ``models`` property is empty.  Keep frameworks,
    # scores, summaries, and sample splits, but drop fitted base learners.
    if hasattr(model, "_models"):
        model._models = None
    for submodel in getattr(model, "_modellist", []) or []:
        if hasattr(submodel, "_models"):
            submodel._models = None


def load_or_fit(name, fit_function):
    """Load a fitted model if it exists; otherwise fit and save it."""

    path = checkpoint_path(name)
    checkpoint_ready_to_load(name)

    if path.exists():
        print(f"Loading checkpoint: {path.name}", flush=True)
        return joblib.load(path)

    print(f"Estimating: {name}", flush=True)
    fitted = fit_function()

    # Most fit functions return a DoubleML model directly.  The clustered
    # APOS fit additionally returns its cluster-robust standard errors in a
    # dictionary, so extract the model before attaching checkpoint metadata.
    fitted_model = fitted["model"] if isinstance(fitted, dict) else fitted

    # The fitted DoubleML framework retains the influence scores and summary
    # needed for tables, sensitivity, and GATE. The fitted base learners are
    # not needed after their convex weights have been extracted, so remove
    # them before saving. This is especially important for APOS, whose fitted
    # treatment-level submodels otherwise create multi-GB checkpoints.
    compact_fitted_model(fitted_model)
    joblib.dump(fitted, path, compress=3)
    print(f"Saved checkpoint: {path.name}", flush=True)
    return fitted


def checkpoint_path(name):
    """Return the checkpoint path without loading the checkpoint."""

    return CHECKPOINT_DIR / f"{name}{CHECKPOINT_TAG}.pkl"


def collect_convex_weights(model):
    """Collect and average convex weights before fitted learners are removed."""

    collected = {}

    def visit(value, nuisance="unknown"):
        if isinstance(value, dict):
            for key, item in value.items():
                next_nuisance = key if str(key).startswith("ml_") else nuisance
                visit(item, next_nuisance)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, nuisance)
        elif isinstance(value, np.ndarray):
            for item in value.flat:
                visit(item, nuisance)
        elif hasattr(value, "weights_") and hasattr(value, "estimators"):
            names = [name for name, _ in value.estimators]
            row = dict(zip(names, np.asarray(value.weights_, dtype=float)))
            collected.setdefault(nuisance, []).append(row)

    # Read the private backing store while fitted learners are still present.
    # The public ``models`` property may be empty/read-only in some DoubleML
    # versions even though ``_models`` contains the fitted nuisances.
    model_store = getattr(model, "_models", None)
    if model_store is None:
        model_store = getattr(model, "models", {})
    visit(model_store)
    for submodel in getattr(model, "_modellist", []) or []:
        submodel_store = getattr(submodel, "_models", None)
        if submodel_store is None:
            submodel_store = getattr(submodel, "models", {})
        visit(submodel_store)
    averaged = {}
    for nuisance, rows in collected.items():
        names = list(dict.fromkeys(name for row in rows for name in row))
        averaged[nuisance] = {
            name: float(np.mean([row[name] for row in rows if name in row]))
            for name in names
        }
    return averaged


def fit_irm(frame, x_columns, outcome, treatment, clustered):
    """Fit one interactive model for a binary treatment."""

    data = dml.DoubleMLData(
        frame,
        y_col=outcome,
        d_cols=treatment,
        x_cols=x_columns,
        cluster_cols="Cluster_var" if clustered else None,
    )
    model = dml.DoubleMLIRM(
        data,
        ConvexRegressor(
            REGRESSORS,
            random_state=SEED,
            group_column=-1 if clustered else None,
        ),
        ConvexClassifier(
            CLASSIFIERS,
            random_state=SEED,
            group_column=-1 if clustered else None,
        ),
        n_folds=FOLDS,
        n_rep=REPETITIONS,
        draw_sample_splitting=True,
    )
    return model.fit(
        n_jobs_cv=1,
        store_predictions=False,
        store_models=True,
    )


def fit_apos(frame, x_columns, outcome):
    """Fit one multi-valued APOS model for treatment levels 0,1,2,3,98."""

    data = dml.DoubleMLData(
        frame,
        y_col=outcome,
        d_cols="WQ15_g",
        x_cols=x_columns,
    )
    model = dml.DoubleMLAPOS(
        data,
        ConvexRegressor(REGRESSORS, random_state=SEED, group_column=None),
        ConvexClassifier(CLASSIFIERS, random_state=SEED, group_column=None),
        treatment_levels=list(TREATMENT_LEVELS),
        n_folds=FOLDS,
        n_rep=REPETITIONS,
        draw_sample_splitting=True,
    )
    with parallel_backend("threading"):
        return model.fit(
            n_jobs_models=APOS_WORKERS,
            n_jobs_cv=1,
            store_predictions=False,
            store_models=False,
        )


def make_cluster_splits(frame, treatment):
    """Make repeated folds that keep every sampling cluster together."""

    groups = frame["Cluster_var"].to_numpy()
    target = frame[treatment].to_numpy()
    all_repetitions = []

    for repetition in range(REPETITIONS):
        splitter = StratifiedGroupKFold(
            n_splits=FOLDS,
            shuffle=True,
            random_state=SEED + repetition,
        )
        splits = list(splitter.split(np.zeros(len(frame)), target, groups))
        all_repetitions.append(splits)

    return all_repetitions


def fit_apos_clustered(frame, x_columns, outcome, splits):
    """Fit APOS on cluster-level folds and calculate cluster-robust SEs.

    The APOS point estimates come from DoubleML. The standard errors are
    replaced after estimation by the one-way cluster sandwich in
    ``_functions.cluster_robust_framework_se``.
    """

    data = dml.DoubleMLData(
        frame,
        y_col=outcome,
        d_cols="WQ15_g",
        x_cols=x_columns,
    )
    model = dml.DoubleMLAPOS(
        data,
        ConvexRegressor(REGRESSORS, random_state=SEED, group_column=-1),
        ConvexClassifier(CLASSIFIERS, random_state=SEED, group_column=-1),
        treatment_levels=list(TREATMENT_LEVELS),
        n_folds=FOLDS,
        n_rep=REPETITIONS,
        draw_sample_splitting=False,
    )
    model.set_sample_splitting(splits)
    with parallel_backend("threading"):
        model = model.fit(
            n_jobs_models=APOS_WORKERS,
            n_jobs_cv=1,
            store_predictions=False,
            store_models=False,
        )
    contrast = model.causal_contrast(reference_levels=[0])
    cluster_se = cluster_robust_framework_se(
        contrast,
        frame["Cluster_var"].to_numpy(),
        splits,
    )
    return {"model": model, "cluster_se": cluster_se}


def _add_metadata(summary, dataset, outcome, method, specification, n, clusters):
    """Add labels used by the result tables."""

    table = summary.reset_index()
    table.insert(0, "dataset", dataset)
    table.insert(1, "outcome", outcome)
    table.insert(2, "method", method)
    table.insert(3, "specification", specification)
    table["n"] = n
    table["clusters"] = clusters
    return table


# ============================================================
# 6. Load data and estimate IRM/APOS models
# ============================================================

def load_analysis_data(path, outcome, child):
    """Load only the columns needed for one outcome and one sample."""

    columns = set(COMMON_CONTROLS)
    columns.update({
        outcome,
        "water_treatment",
        "WQ15_g",
        "country_cat",
        "Cluster_var",
    })
    if child:
        columns.update({"age", "male"})

    # convert_categoricals=False preserves numeric Stata codes for treatment
    # levels. Loading selected columns keeps unrelated survey variables out of
    # memory; only one outcome's data frame is alive at a time.
    data = pd.read_stata(
        path,
        columns=sorted(columns),
        convert_categoricals=False,
    )
    if SAMPLED:
        data = data.sample(
            frac=SAMPLE_FRAC,
            random_state=SAMPLE_SEED,
        ).reset_index(drop=True)
        print(
            f"Quick sample: {len(data):,} observations "
            f"({SAMPLE_FRAC:.0%} of loaded data)"
        )
    return data

analysis_specs = [
    ("HH", DATA_DIR / "MASTER_MICS_FINAL.dta", False, "SomeRiskHome"),
    ("HH", DATA_DIR / "MASTER_MICS_FINAL.dta", False, "VeryHighRiskHome"),
    ("U5", DATA_DIR / "MASTER_MICS_FINAL_U5.dta", True, "diarrhea"),
]

irm_results = []
apos_results = []
weight_results = []
estimates = {}

for dataset, data_path, child, outcome in analysis_specs:
    print(f"Loading columns for {dataset} — {outcome}", flush=True)
    data = load_analysis_data(data_path, outcome, child)
    print(f"\nPreparing {dataset} — {outcome}", flush=True)

    # IRM uses the binary treatment indicator.
    frame, x_columns = make_frame(
        data, outcome, "water_treatment", child=child,
        cluster=True, allowed_levels=(0, 1),
    )
    clusters = frame["Cluster_var"].nunique()
    irm_models = {}
    irm_frames = {}

    for clustered in (True, False):
        if clustered:
            irm_frame = frame
            irm_x = x_columns
        else:
            irm_frame, irm_x = make_frame(
                data, outcome, "water_treatment", child=child,
                cluster=False, allowed_levels=(0, 1),
            )

        name = f"{dataset}_{outcome}_IRM_{'clustered' if clustered else 'iid'}"
        model = load_or_fit(
            name,
            lambda f=irm_frame, x=irm_x, c=clustered: fit_irm(
                f, x, outcome, "water_treatment", c
            ),
        )
        irm_models["cluster" if clustered else "no_cluster"] = model
        irm_frames["cluster" if clustered else "no_cluster"] = irm_frame
        irm_results.append(_add_metadata(
            model.summary,
            dataset, outcome, "IRM",
            "clustered" if clustered else "iid",
            len(irm_frame), clusters,
        ))

        # The checkpoint keeps only these averaged weights, not the fitted
        # base learners themselves.
        for nuisance, learner_weights in getattr(model, "convex_weights", {}).items():
            for learner, weight in learner_weights.items():
                weight_results.append({
                    "dataset": dataset,
                    "outcome": outcome,
                    "model": name,
                    "nuisance": nuisance,
                    "learner": learner,
                    "weight": float(weight),
                })

    # Keep only the small columns needed by the publication tables.  The
    # complete IRM design matrices are no longer needed once both IRM fits
    # have finished; release them before allocating the APOS matrices.
    irm_table_frame_cluster = irm_frames["cluster"][
        [outcome, "water_treatment", "Cluster_var"]
    ].copy()
    irm_table_frame_iid = irm_frames["no_cluster"][
        [outcome, "water_treatment"]
    ].copy()
    del frame, irm_frame, irm_frames
    del irm_models, model
    gc.collect()

    # APOS is estimated twice, just like IRM:
    #   1. clustered folds + custom cluster-robust SEs;
    #   2. ordinary observation-level folds + ordinary DoubleML SEs.
    #
    # DoubleMLAPOS in our version does not accept cluster_cols directly. The
    # clustered APOS specification therefore keeps clusters together in every
    # fold and replaces the default SE with our cluster sandwich below.
    apos_frame, apos_x = make_frame(
        data, outcome, "WQ15_g", child=child,
        cluster=True, allowed_levels=TREATMENT_LEVELS,
    )
    apos_splits = make_cluster_splits(apos_frame, "WQ15_g")
    apos_name = f"{dataset}_{outcome}_APOS"
    apos_table_frame_cluster = apos_frame[
        [outcome, "WQ15_g", "Cluster_var"]
    ].copy()
    apos_n = len(apos_frame)
    apos_clusters = apos_frame["Cluster_var"].nunique()
    apos_checkpoint_exists = checkpoint_ready_to_load(apos_name)
    if apos_checkpoint_exists:
        # Existing full APOS checkpoints can be several GB.  Do not keep the
        # raw data or design matrix alive while unpickling one of them.
        del data, apos_frame, apos_x, apos_splits
        gc.collect()
    apos_model = load_or_fit(
        apos_name,
        lambda: fit_apos_clustered(
            apos_frame, apos_x, outcome, apos_splits
        ),
    )
    apos_fitted = apos_model["model"]
    apos_cluster_se = apos_model["cluster_se"]
    apos_summary = apos_fitted.causal_contrast(reference_levels=[0]).summary
    apos_results.append(_add_metadata(
        apos_summary,
        dataset, outcome, "APOS", "clustered_folds",
        apos_n, apos_clusters,
    ))
    if not apos_checkpoint_exists:
        del apos_frame, apos_splits
    # APOS clustered is now represented by its checkpoint path in
    # ``estimates``.  Do not retain the fitted DoubleML object while loading
    # the iid APOS checkpoint.
    del apos_model, apos_fitted, apos_summary
    gc.collect()

    # The unclustered APOS specification uses ordinary observation-level
    # sample splitting and the standard DoubleML standard errors.
    if apos_checkpoint_exists:
        data = load_analysis_data(data_path, outcome, child)
    apos_frame_iid, apos_x_iid = make_frame(
        data, outcome, "WQ15_g", child=child,
        cluster=False, allowed_levels=TREATMENT_LEVELS,
    )
    apos_iid_name = f"{dataset}_{outcome}_APOS_iid"
    print(f"Preparing {apos_iid_name}", flush=True)
    apos_table_frame_iid = apos_frame_iid[
        [outcome, "WQ15_g"]
    ].copy()
    apos_iid_n = len(apos_frame_iid)
    apos_iid_checkpoint_exists = checkpoint_ready_to_load(apos_iid_name)
    if apos_iid_checkpoint_exists:
        del data, apos_frame_iid, apos_x_iid
        gc.collect()
    apos_iid = load_or_fit(
        apos_iid_name,
        lambda: fit_apos(apos_frame_iid, apos_x_iid, outcome),
    )
    apos_iid_summary = apos_iid.causal_contrast(
        reference_levels=[0]
    ).summary
    apos_results.append(_add_metadata(
        apos_iid_summary,
        dataset, outcome, "APOS", "iid",
        apos_iid_n, None,
    ))
    if not apos_iid_checkpoint_exists:
        del apos_frame_iid
    del apos_iid, apos_iid_summary
    gc.collect()
    gc.collect()

    # This is the structure expected by the existing publication-table code.
    # Keep only the columns needed for table statistics. The full design
    # matrices are not retained in the in-memory results dictionary.
    checkpoint_paths = {
        "irm_cluster": checkpoint_path(
            f"{dataset}_{outcome}_IRM_clustered"
        ),
        "irm_no_cluster": checkpoint_path(
            f"{dataset}_{outcome}_IRM_iid"
        ),
        "apos_cluster": checkpoint_path(
            f"{dataset}_{outcome}_APOS"
        ),
        "apos_no_cluster": checkpoint_path(
            f"{dataset}_{outcome}_APOS_iid"
        ),
        "apos_cluster_se": checkpoint_path(
            f"{dataset}_{outcome}_APOS"
        ),
    }
    estimates[(dataset, outcome)] = LazyCheckpointBundle(
        checkpoint_paths,
        {
        # ``_tables.py`` uses the explicit irm_frame_* names for publication
        # tables. Keep the shorter aliases below for the later diagnostics.
            "irm_frame_cluster": irm_table_frame_cluster,
            "irm_frame_no_cluster": irm_table_frame_iid,
            "frame_cluster": irm_table_frame_cluster,
            "frame_no_cluster": irm_table_frame_iid,
            "apos_frame_cluster": apos_table_frame_cluster,
            "apos_frame_no_cluster": apos_table_frame_iid,
        },
    )

    print(
        f"Completed {dataset} — {outcome}: "
        "IRM clustered, IRM iid, APOS clustered, APOS iid",
        flush=True,
    )

    # Release every large, outcome-local frame before loading the next one.
    # ``estimates`` retains only the compact model objects and table frames
    # needed by the post-estimation code; the design matrices are not kept.
    for _name in (
        "data", "apos_model", "apos_fitted", "apos_iid", "apos_x",
        "apos_x_iid", "x_columns", "apos_frame", "apos_frame_iid",
        "apos_splits",
    ):
        if _name in locals():
            del locals()[_name]
    gc.collect()

expected_estimates = {
    (dataset, outcome) for dataset, _, _, outcome in analysis_specs
}
missing_estimates = expected_estimates.difference(estimates)
if missing_estimates:
    raise RuntimeError(
        "Estimation stopped before all analysis specifications completed. "
        f"Missing: {sorted(missing_estimates)}"
    )
print(
    f"All {len(expected_estimates)} analysis specifications completed; "
    "12 model specifications are available.",
    flush=True,
)


# ============================================================
# 7. Tables and final saved results
# ============================================================

irm_results = pd.concat(irm_results, ignore_index=True)
apos_results = pd.concat(apos_results, ignore_index=True)
pd.to_pickle(irm_results, OUTPUT_DIR / "results_irm.pkl")
pd.to_pickle(apos_results, OUTPUT_DIR / "results_apos.pkl")

if weight_results:
    weights = pd.DataFrame(weight_results)
    pd.to_pickle(weights, OUTPUT_DIR / "results_convex_weights.pkl")
else:
    weights = pd.DataFrame()

# Reuse the project's publication-table builder. It adds readable treatment
# labels, panels, significance stars, standard errors, sample statistics, and
# methodological notes. The table code lives in _tables.py so formatting is
# kept separate from estimation.
write_publication_table(
    TABLE_DIR,
    estimates,
    ["SomeRiskHome", "VeryHighRiskHome", "diarrhea"],
    "table_water_treatment_main.tex",
    "Stacked water-treatment effects",
    "tab:water-treatment-main",
    REPORTED_LEVELS,
    FOLDS,
    REPETITIONS,
    specifications=("clustered",),
)

# Appendix: show the same estimates under both fold constructions.
write_publication_table(
    TABLE_DIR,
    estimates,
    ["SomeRiskHome", "VeryHighRiskHome", "diarrhea"],
    "table_water_treatment_appendix.tex",
    "Stacked water-treatment effects: clustered and ordinary folds",
    "tab:water-treatment-appendix",
    REPORTED_LEVELS,
    FOLDS,
    REPETITIONS,
    specifications=("clustered", "unclustered"),
)

write_super_learner_weights_tables(
    TABLE_DIR,
    estimates,
    ["SomeRiskHome", "VeryHighRiskHome", "diarrhea"],
    weights=weights,
)

manifest = {
    "seed": SEED,
    "sampled": SAMPLED,
    "sample_frac": SAMPLE_FRAC if SAMPLED else None,
    "folds": FOLDS,
    "repetitions": REPETITIONS,
    "inner_folds": INNER_FOLDS,
    "apos_workers": APOS_WORKERS,
    "learner_jobs": LEARNER_JOBS,
    "learners_outcome": [name for name, _ in REGRESSORS],
    "learners_treatment": [name for name, _ in CLASSIFIERS],
    "treatment_levels": list(TREATMENT_LEVELS),
    "checkpoints": sorted(path.name for path in CHECKPOINT_DIR.glob("*.pkl")),
    "tables": sorted(path.name for path in TABLE_DIR.glob("*.tex")),
}
(OUTPUT_DIR / "manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)


# ============================================================
# 8. Sensitivity analysis — run last
# ============================================================
# This section does not refit the main models. It loads the fitted
# checkpoints, computes DoubleML's robustness values for cf_y = cf_d = 0.03,
# and saves the results separately. Keeping it last means the main estimates
# and publication table are available even if this optional stage is slow.

sensitivity_rows = []
for (dataset, outcome), bundle in estimates.items():
    irm = bundle["irm_cluster"]
    irm_rv, irm_rva = sensitivity_params(irm)
    del irm
    apos_model = bundle["apos_cluster"]
    apos = apos_model.causal_contrast(reference_levels=[0])
    apos_rv, apos_rva = sensitivity_params(
        apos,
        cluster_ids=bundle["apos_frame_cluster"]["Cluster_var"].to_numpy(),
        smpls=apos_model.smpls,
    )
    del apos, apos_model

    sensitivity_rows.append({
        "dataset": dataset,
        "outcome": outcome,
        "specification": "clustered_folds",
        "method": "IRM",
        "treatment": "Any Treatment",
        "rv": float(irm_rv[0]),
        "rva": float(irm_rva[0]),
    })
    for index, level in enumerate((1, 2, 3)):
        sensitivity_rows.append({
            "dataset": dataset,
            "outcome": outcome,
            "specification": "clustered_folds",
            "method": "APOS",
            "treatment": str(level),
            "rv": float(apos_rv[index]),
            "rva": float(apos_rva[index]),
        })

sensitivity_results = pd.DataFrame(sensitivity_rows)
pd.to_pickle(
    sensitivity_results,
    OUTPUT_DIR / "results_sensitivity.pkl",
)
# Keep sensitivity outputs separate from the regression tables.  The compact
# summary is the main sensitivity table; the benchmark-style details can be
# added later without changing the main estimates.
write_sensitivity_summary_table(
    sensitivity_results,
    TABLE_DIR,
    filename="table_sensitivity_main.tex",
)


# ============================================================
# 9. Heterogeneity and GATE — run last
# ============================================================
# GATE is a projection of already-estimated orthogonal scores. It does not
# refit the nuisance learners. The prespecified grouping variable is the
# initial source-water E. coli decile, wq27_decile.

gate_rows = []
for (dataset, outcome), bundle in estimates.items():
    data_path = (
        DATA_DIR / "MASTER_MICS_FINAL_U5.dta"
        if dataset == "U5" else DATA_DIR / "MASTER_MICS_FINAL.dta"
    )
    child = dataset == "U5"
    data = load_analysis_data(data_path, outcome, child)

    for clustered, specification in (
        (True, "clustered_folds"),
        (False, "unclustered"),
    ):
        treatment = "water_treatment"
        allowed = (0, 1)
        gate_frame, _ = make_frame(
            data, outcome, treatment, child=child,
            cluster=clustered, allowed_levels=allowed,
        )
        # Recover the grouping values in the same complete-case order as the
        # model frame. The frame stores dummies, so this column is rebuilt
        # directly from the selected rows.
        controls = list(COMMON_CONTROLS) + (["age", "male"] if child else [])
        required = [outcome, treatment, *controls, "country_cat"]
        if clustered:
            required.append("Cluster_var")
        keep = data[required].notna().all(axis=1)
        keep &= data[treatment].isin(allowed)
        groups = data.loc[keep, "wq27_decile"].reset_index(drop=True)

        irm = bundle["irm_cluster" if clustered else "irm_no_cluster"]
        irm_cluster_ids = (
            gate_frame["Cluster_var"].to_numpy() if clustered else None
        )
        irm_gate = estimate_gate_from_contrast(
            irm.framework,
            "Any Treatment",
            0,
            groups,
            irm_cluster_ids,
            group_labels={str(i): f"Decile {i}" for i in range(1, 11)},
        )
        irm_gate.insert(0, "dataset", dataset)
        irm_gate.insert(1, "outcome", outcome)
        irm_gate.insert(2, "method", "IRM stacked")
        irm_gate.insert(3, "specification", specification)
        irm_gate.insert(4, "group", "source_ecoli")
        irm_gate.insert(5, "heterogeneity_label", "Initial source-water E. coli decile")
        irm_gate.insert(6, "treatment_label", "Any Treatment")
        gate_rows.append(irm_gate)
        del irm, irm_gate, gate_frame, groups

        apos = bundle["apos_cluster" if clustered else "apos_no_cluster"]
        apos_contrast = apos.causal_contrast(reference_levels=[0])
        apos_frame = (
            bundle["apos_frame_cluster"] if clustered
            else bundle["apos_frame_no_cluster"]
        )
        apos_required = [outcome, "WQ15_g", *controls, "country_cat"]
        if clustered:
            apos_required.append("Cluster_var")
        apos_keep = data[apos_required].notna().all(axis=1)
        apos_keep &= data["WQ15_g"].isin(TREATMENT_LEVELS)
        apos_groups = data.loc[apos_keep, "wq27_decile"].reset_index(drop=True)
        apos_cluster_ids = (
            apos_frame["Cluster_var"].to_numpy() if clustered else None
        )
        for effect_index, level in enumerate((1, 2, 3)):
            apos_gate = estimate_gate_from_contrast(
                apos_contrast,
                level,
                effect_index,
                apos_groups,
                apos_cluster_ids,
                group_labels={str(i): f"Decile {i}" for i in range(1, 11)},
            )
            apos_gate.insert(0, "dataset", dataset)
            apos_gate.insert(1, "outcome", outcome)
            apos_gate.insert(2, "method", "APOS stacked")
            apos_gate.insert(3, "specification", specification)
            apos_gate.insert(4, "group", "source_ecoli")
            apos_gate.insert(5, "heterogeneity_label", "Initial source-water E. coli decile")
            apos_gate.insert(6, "treatment_label", {
                1: "Boiling",
                2: "Chlorination/tablets",
                3: "Straining/settling",
            }[level])
            gate_rows.append(apos_gate)
        del apos, apos_contrast, apos_frame, apos_groups, apos_cluster_ids

    del data

gate_results = pd.concat(gate_rows, ignore_index=True)
pd.to_pickle(gate_results, OUTPUT_DIR / "results_heterogeneity_gates.pkl")
create_heterogeneity_comparison_tables(
    gate_results,
    output_dir=TABLE_DIR,
    filename_prefix="table_gate_main",
    specifications=("clustered_folds",),
)
create_heterogeneity_comparison_tables(
    gate_results,
    output_dir=TABLE_DIR,
    filename_prefix="table_gate_appendix",
    specifications=("clustered_folds", "unclustered"),
)

print("\nAnalysis finished.")
print(f"Checkpoints: {CHECKPOINT_DIR}")
print(f"Tables:      {TABLE_DIR}")
