"""Shared data preparation, cross-fitting, Super Learner, and inference tools."""

import os
from time import perf_counter

import doubleml as dml
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

def controls_for_sample(common_controls, child_controls, child):
    """Return controls in their prespecified modeling order."""
    controls = list(common_controls)
    if child:
        controls.extend(child_controls)
    return controls


def complete_case_sample(
    data,
    outcome,
    treatment,
    controls,
    *,
    cluster=True,
    cluster_column="Cluster_var",
    country_column="country_cat",
    allowed_levels=None,
    extra_columns=(),
):
    """Select model-complete rows while retaining optional metadata."""
    required = [outcome, treatment, *controls, country_column]
    if cluster:
        required.append(cluster_column)

    frame = data[required].copy().dropna()
    if allowed_levels is not None:
        frame = frame[frame[treatment].isin(allowed_levels)].copy()
    for column in extra_columns:
        frame[column] = data.loc[frame.index, column]
    return frame.reset_index(drop=True)


def make_frame(
    data,
    outcome,
    treatment,
    controls,
    *,
    categorical_controls,
    cluster=True,
    cluster_column="Cluster_var",
    country_column="country_cat",
    allowed_levels=None,
):
    """Create the encoded DoubleML frame and ordered predictor names."""
    frame = complete_case_sample(
        data,
        outcome,
        treatment,
        controls,
        cluster=cluster,
        cluster_column=cluster_column,
        country_column=country_column,
        allowed_levels=allowed_levels,
    )

    categorical = list(categorical_controls)
    numeric = [name for name in controls if name not in categorical]
    controls_frame = frame[numeric + categorical].copy()
    x = pd.get_dummies(
        controls_frame,
        columns=categorical,
        drop_first=True,
        dtype=float,
    ).astype(float).reset_index(drop=True)
    if cluster:
        x["_cluster_model_code"] = pd.factorize(
            frame[cluster_column], sort=True
        )[0].astype(float)

    columns = [outcome, treatment]
    if cluster:
        columns.append(cluster_column)
    model_frame = pd.concat([frame[columns], x], axis=1)
    return model_frame, list(x.columns)


def load_analysis_data(
    path,
    outcome,
    controls,
    *,
    country_codes=None,
    quick_sample=False,
    sample_fraction=0.05,
    sample_seed=42,
):
    """Load only analysis columns, filter countries, then draw a sample."""
    country_column = "country_cat"
    columns = set(controls)
    columns.update({
        outcome,
        "water_treatment",
        "WQ15_g",
        country_column,
        "Cluster_var",
        "RiskSource",
    })
    data = pd.read_stata(
        path,
        columns=sorted(columns),
        convert_categoricals=False,
    )
    if country_codes is not None:
        data = data[data[country_column].isin(country_codes)].copy()
        print(f"Selected countries: {len(data):,} observations before sampling")
    if quick_sample:
        data = data.sample(
            frac=sample_fraction,
            random_state=sample_seed,
        ).reset_index(drop=True)
        print(
            f"Quick sample: {len(data):,} observations "
            f"({sample_fraction:.0%} of loaded data)"
        )
    return data

def validate_splits(splits, target, groups=None):
    """Check sample coverage, treatment support, and optional PSU isolation."""
    expected_levels = np.unique(target)
    expected_rows = np.arange(len(target))
    for repetition in splits:
        tested_rows = np.concatenate([test for _, test in repetition])
        if not np.array_equal(np.sort(tested_rows), expected_rows):
            raise ValueError("Test folds do not partition the analysis sample.")
        for train, test in repetition:
            if not np.array_equal(np.unique(target[train]), expected_levels):
                raise ValueError("A training fold is missing a treatment level.")
            if not np.array_equal(np.unique(target[test]), expected_levels):
                raise ValueError("A test fold is missing a treatment level.")
            if groups is not None:
                overlap = np.intersect1d(groups[train], groups[test])
                if overlap.size:
                    raise ValueError("A sampling cluster appears in train and test.")


def make_iid_splits(frame, treatment, *, n_folds, repetitions, seed):
    """Create deterministic stratified folds for observation-level fitting."""
    target = frame[treatment].to_numpy()
    all_repetitions = []
    for repetition in range(repetitions):
        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed + repetition,
        )
        all_repetitions.append(
            list(splitter.split(np.zeros(len(frame)), target))
        )
    validate_splits(all_repetitions, target)
    return all_repetitions


def make_cluster_splits(
    frame, treatment, *, n_folds, repetitions, seed,
    cluster_column="Cluster_var",
):
    """Create stratified folds that keep every sampling cluster together."""
    groups = frame[cluster_column].to_numpy()
    target = frame[treatment].to_numpy()
    all_repetitions = []
    for repetition in range(repetitions):
        splitter = StratifiedGroupKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed + repetition,
        )
        all_repetitions.append(
            list(splitter.split(np.zeros(len(frame)), target, groups))
        )
    validate_splits(all_repetitions, target, groups=groups)
    return all_repetitions


def make_cluster_split_metadata(
    frame, splits, *, cluster_column="Cluster_var",
):
    """Translate observation folds into DoubleML's PSU-fold metadata."""
    groups = frame[cluster_column].to_numpy()
    return [
        [
            ([np.unique(groups[train])], [np.unique(groups[test])])
            for train, test in repetition
        ]
        for repetition in splits
    ]

def estimate_gate_from_contrast(
    contrast,
    treatment_level,
    effect_index,
    group_values,
    cluster_ids,
    group_labels,
    level=0.95,
    n_rep_boot=500,
):
    """Project a fitted orthogonal signal onto prespecified GATE groups.

    Parameters
    ----------
    contrast : DoubleMLFramework
        Fitted IRM effect or APOS treatment contrast.
    treatment_level : object
        Treatment label copied into the returned rows.
    effect_index : int
        Contrast column projected from the J effects in ``contrast``.
    group_values : array-like
        Numeric group code for every observation.
    cluster_ids : array-like or None
        PSU identifiers for clustered covariance; ``None`` requests HC0.
    group_labels : dict[str, str]
        Mapping from numeric group strings to readable labels.
    level : float, default=0.95
        Pointwise and joint confidence level.
    n_rep_boot : int, default=500
        Gaussian draws used for joint intervals.

    Returns
    -------
    pandas.DataFrame
        One row per observed group with coefficient, SE, p-value, intervals,
        sample counts, and median BLP R-squared.
    """

    # Use the same N observations x J effects x R repetitions layout as the
    # clustered-inference section above.
    scaled_influence_scores = score_array_with_named_dimensions(
        contrast.scaled_psi
    )
    # DoubleML stores scaled_psi = theta - orthogonal_signal for these linear
    # scores. GATEs must project the orthogonal signal itself, as the native
    # IRM gate() method does, rather than project the centered influence term.
    all_thetas = np.asarray(contrast.all_thetas, dtype=float)
    # all_thetas is J x R. np.newaxis inserts an observation dimension, making
    # it 1 x J x R; NumPy then broadcasts each theta across all N observations
    # when subtracting the N x J x R influence-score array.
    orthogonal_signals = (
        all_thetas[np.newaxis, :, :] - scaled_influence_scores
    )

    values = pd.Series(group_values).reset_index(drop=True)
    valid_observation = values.notna().to_numpy()
    values = values.loc[valid_observation].reset_index(drop=True)
    orthogonal_signals = orthogonal_signals[valid_observation, :, :]

    # Normalize numeric group codes to strings before creating indicators.
    numeric_values = pd.to_numeric(values)
    values = numeric_values.astype(int).astype("string")

    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)[valid_observation]

    group_indicators = pd.get_dummies(
        values.astype("string"),
        prefix="group",
        dtype=float,
    )
    observed_groups = group_indicators.sum(axis=0) > 0
    group_indicators = group_indicators.loc[:, observed_groups]

    blp = dml.DoubleMLBLP(
        orthogonal_signals[:, effect_index, :],
        basis=group_indicators,
        is_gate=True,
    )
    if cluster_ids is None:
        blp.fit(cov_type="HC0")
    else:
        blp.fit(
            cov_type="cluster",
            cov_kwds={"groups": cluster_ids},
        )

    pointwise_interval = blp.confint(joint=False, level=float(level))
    joint_interval = blp.confint(
        joint=True,
        level=float(level),
        n_rep_boot=int(n_rep_boot),
    )

    labels = group_labels
    r_squared_by_repetition = [
        float(fitted_model.rsquared)
        for fitted_model in blp._blp_model
    ]
    median_r_squared = float(np.median(r_squared_by_repetition))

    rows = []
    for indicator_name in group_indicators.columns:
        raw_value = indicator_name.removeprefix("group_")
        readable_label = labels[raw_value]

        group_mask = values.astype("string").eq(raw_value).to_numpy()
        if cluster_ids is None:
            number_of_group_clusters = np.nan
        else:
            number_of_group_clusters = pd.Series(
                cluster_ids[group_mask]
            ).nunique()

        result = blp.summary.loc[indicator_name]
        rows.append({
            "treatment_level": treatment_level,
            "group_value": raw_value,
            "group_label": readable_label,
            "n": int(group_indicators[indicator_name].sum()),
            "n_psu": (
                int(number_of_group_clusters)
                if pd.notna(number_of_group_clusters)
                else np.nan
            ),
            "r2": median_r_squared,
            "coef": float(result["coef"]),
            "se": float(result["std err"]),
            "pval": float(result["P>|t|"]),
            "ci_lower": float(pointwise_interval.loc[indicator_name].iloc[0]),
            "ci_upper": float(pointwise_interval.loc[indicator_name].iloc[-1]),
            "ci_lower_joint": float(
                joint_interval.loc[indicator_name].iloc[0]
            ),
            "ci_upper_joint": float(
                joint_interval.loc[indicator_name].iloc[-1]
            ),
        })

    return pd.DataFrame(rows)


def summary_with_clustered_inference(summary, clustered_inference):
    """Replace ordinary APOS inference with repeated clustered inference.

    This copy is saved in ``results_apos.pkl``. The result file and LaTeX
    tables therefore report the same standard errors, t statistics, p-values,
    and confidence intervals.

    Parameters
    ----------
    summary : pandas.DataFrame
        Ordinary DoubleML contrast summary providing coefficients and rows.
    clustered_inference : dict[str, numpy.ndarray]
        Output from ``cluster_robust_framework_inference``.

    Returns
    -------
    pandas.DataFrame
        Summary with clustered SEs, p-values, and confidence intervals.
    """

    adjusted = summary.copy()
    standard_errors = np.asarray(clustered_inference["se"], dtype=float)

    coefficients = adjusted["coef"].to_numpy(dtype=float)
    t_statistics = coefficients / standard_errors

    adjusted["std err"] = standard_errors
    adjusted["t"] = t_statistics
    adjusted["P>|t|"] = np.asarray(
        clustered_inference["pval"],
        dtype=float,
    )
    adjusted["2.5 %"] = np.asarray(
        clustered_inference["ci_lower"],
        dtype=float,
    )
    adjusted["97.5 %"] = np.asarray(
        clustered_inference["ci_upper"],
        dtype=float,
    )
    return adjusted

def convex_weights(predictions, target, classification=False):
    """Choose nonnegative Super Learner weights that add to one."""

    predictions = np.asarray(predictions, dtype=float)
    target = np.asarray(target, dtype=float)
    if (
        predictions.ndim != 2
        or predictions.shape[0] == 0
        or predictions.shape[1] == 0
    ):
        raise ValueError(
            "Predictions must be a nonempty observations-by-learners matrix"
        )
    if target.ndim != 1 or len(target) != predictions.shape[0]:
        raise ValueError("Target must be a one-dimensional vector aligned with predictions")
    if not np.isfinite(predictions).all() or not np.isfinite(target).all():
        raise ValueError("Predictions and target must contain only finite values")
    if classification and not np.isin(target, [0, 1]).all():
        raise ValueError("Classification targets must contain only 0 and 1")

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
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1},
    )
    if not result.success:
        raise RuntimeError(f"Convex weight optimization failed: {result.message}")
    weights = np.asarray(result.x, dtype=float)
    if not np.isfinite(weights).all():
        raise RuntimeError("Convex weight optimization returned nonfinite weights")
    weights = np.clip(weights, 0, 1)
    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise RuntimeError("Convex weight optimization returned invalid weights")
    return weights / weight_sum


def positive_probability(model, x):
    """Return the probability assigned to class 1."""

    positive = int(np.where(np.asarray(model.classes_) == 1)[0][0])
    return model.predict_proba(x)[:, positive]


def _verbose_learner_logging():
    """Enable per-learner logs only when explicitly requested."""
    return os.getenv("DDML_VERBOSE_LEARNERS", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


class _ConvexBase:
    """Shared configuration and cluster-column handling."""

    def __init__(
        self,
        estimators,
        random_state=42,
        group_column=None,
        inner_folds=3,
    ):
        self.estimators = estimators
        self.random_state = random_state
        self.group_column = group_column
        self.inner_folds = inner_folds

    def _features_and_groups(self, x):
        x = np.asarray(x, dtype=float)
        if self.group_column is None:
            return x, None
        index = (
            self.group_column
            if self.group_column >= 0
            else x.shape[1] + self.group_column
        )
        groups = x[:, index].astype(int)
        return np.delete(x, index, axis=1), groups


class ConvexRegressor(_ConvexBase, RegressorMixin, BaseEstimator):
    """Cross-fitted convex ensemble for an outcome regression."""

    def fit(self, x, y):
        x, groups = self._features_and_groups(x)
        y = np.asarray(y, dtype=float)
        if groups is None:
            splits = KFold(
                self.inner_folds,
                shuffle=True,
                random_state=self.random_state,
            ).split(x)
        else:
            splits = GroupKFold(self.inner_folds).split(x, y, groups)
        splits = list(splits)
        out_of_fold = np.zeros((len(y), len(self.estimators)))
        self.models_ = []
        verbose = _verbose_learner_logging()

        for learner_number, (name, estimator) in enumerate(self.estimators):
            fold_models = []
            if verbose:
                started = perf_counter()
                print(
                    f"[Super Learner regression] {name} "
                    f"({learner_number + 1}/{len(self.estimators)}): fitting "
                    f"{len(splits)} inner folds on {len(y):,} rows.",
                    flush=True,
                )
            for train, test in splits:
                fitted = clone(estimator).fit(x[train], y[train])
                out_of_fold[test, learner_number] = fitted.predict(x[test])
                fold_models.append(fitted)
            if verbose:
                print(
                    f"[Super Learner regression] {name}: done in "
                    f"{perf_counter() - started:.1f}s.",
                    flush=True,
                )
            self.models_.append((name, fold_models))

        self.weights_ = convex_weights(
            out_of_fold,
            y,
            classification=False,
        )
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, x):
        x, _ = self._features_and_groups(x)
        predictions_by_learner = []
        for _, fold_models in self.models_:
            fold_predictions = [model.predict(x) for model in fold_models]
            predictions_by_learner.append(np.mean(fold_predictions, axis=0))
        predictions = np.column_stack(predictions_by_learner)
        return predictions @ self.weights_


class ConvexClassifier(_ConvexBase, ClassifierMixin, BaseEstimator):
    """Cross-fitted convex ensemble for a binary treatment model."""

    def fit(self, x, y):
        x, groups = self._features_and_groups(x)
        y = np.asarray(y, dtype=int)
        if groups is None:
            splits = StratifiedKFold(
                self.inner_folds,
                shuffle=True,
                random_state=self.random_state,
            ).split(x, y)
        else:
            splits = StratifiedGroupKFold(
                self.inner_folds,
                shuffle=True,
                random_state=self.random_state,
            ).split(x, y, groups)
        splits = list(splits)
        out_of_fold = np.zeros((len(y), len(self.estimators)))
        self.models_ = []
        verbose = _verbose_learner_logging()

        for learner_number, (name, estimator) in enumerate(self.estimators):
            fold_models = []
            if verbose:
                started = perf_counter()
                print(
                    f"[Super Learner classification] {name} "
                    f"({learner_number + 1}/{len(self.estimators)}): fitting "
                    f"{len(splits)} inner folds on {len(y):,} rows.",
                    flush=True,
                )
            for train, test in splits:
                fitted = clone(estimator).fit(x[train], y[train])
                out_of_fold[test, learner_number] = positive_probability(
                    fitted,
                    x[test],
                )
                fold_models.append(fitted)
            if verbose:
                print(
                    f"[Super Learner classification] {name}: done in "
                    f"{perf_counter() - started:.1f}s.",
                    flush=True,
                )
            self.models_.append((name, fold_models))

        self.weights_ = convex_weights(
            out_of_fold,
            y,
            classification=True,
        )
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = x.shape[1]
        return self

    def predict_proba(self, x):
        x, _ = self._features_and_groups(x)
        predictions_by_learner = []
        for _, fold_models in self.models_:
            fold_predictions = [
                positive_probability(model, x)
                for model in fold_models
            ]
            predictions_by_learner.append(np.mean(fold_predictions, axis=0))
        predictions = np.column_stack(predictions_by_learner)
        positive = np.clip(predictions @ self.weights_, 1e-8, 1 - 1e-8)
        return np.column_stack([1 - positive, positive])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)



def collect_convex_weights(model):
    """Average stored Super Learner weights over folds and repetitions.

    Parameters
    ----------
    model : fitted DoubleML model
        Model whose nested nuisance learners contain ``weights_`` arrays.

    Returns
    -------
    dict[str, dict[str, float]]
        Average learner weight for each nuisance model.
    """

    collected = {}

    def visit(value, nuisance="unknown"):
        """Recursively find fitted convex learners inside DoubleML storage."""

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

    visit(model._models)
    averaged = {}
    for nuisance, rows in collected.items():
        names = list(dict.fromkeys(name for row in rows for name in row))
        averaged[nuisance] = {
            name: float(np.mean([row[name] for row in rows if name in row]))
            for name in names
        }
    return averaged


def score_array_with_named_dimensions(score_values):
    """Return scores with dimensions (observations, effects, repetitions).

    DoubleML normally already uses this three-dimensional layout. The checks
    below make the layout explicit and also handle simpler one- or two-
    dimensional inputs in the intuitive way. Keeping this conversion in one
    named function is easier to follow than repeating ``np.atleast_3d``.

    Parameters
    ----------
    score_values : array-like
        Influence-score values with observations in the first dimension.

    Returns
    -------
    numpy.ndarray
        Numeric array with shape N observations x J effects x R repetitions.
    """

    scores = np.asarray(score_values, dtype=float)
    if scores.ndim == 1:
        # One effect and one repetition: N becomes N x 1 x 1.
        scores = scores[:, np.newaxis, np.newaxis]
    elif scores.ndim == 2:
        # Several effects and one repetition: N x J becomes N x J x 1.
        scores = scores[:, :, np.newaxis]
    elif scores.ndim != 3:
        raise ValueError(
            "Scores must have one, two, or three dimensions; "
            f"received shape {scores.shape}."
        )
    return scores


def sum_rows_within_psu(values, cluster_ids):
    """Sum observation-level array rows within each sampling PSU.

    Parameters
    ----------
    values
        Array whose first dimension represents observations. Any remaining
        dimensions, such as effects and repetitions, are preserved.
    cluster_ids
        PSU identifier for every observation, in the same row order.

    Returns
    -------
    tuple
        The PSU sums and the corresponding unique PSU labels.
    """

    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    if values.shape[0] != len(cluster_ids):
        raise ValueError(
            "Scores and cluster IDs must contain the same number of rows."
        )

    # return_inverse=True converts arbitrary labels into safe array positions.
    # Example: PSU labels [101, 101, 205] become positions [0, 0, 1].
    unique_clusters, cluster_position = np.unique(
        cluster_ids,
        return_inverse=True,
    )
    psu_sums = np.zeros(
        (len(unique_clusters), *values.shape[1:]),
        dtype=float,
    )

    # A normal assignment would overwrite repeated positions. np.add.at adds
    # every observation to its PSU, including when many rows share that PSU.
    np.add.at(psu_sums, cluster_position, values)
    return psu_sums, unique_clusters


def cluster_robust_framework_inference(framework, cluster_ids, level=0.95):
    """Calculate repeated-cross-fitting inference clustered by PSU.

    The cluster sandwich is evaluated separately for every repetition. Point
    estimates, p-values, and confidence limits are then aggregated using the
    same repetition-wise rules as ``DoubleMLFramework``.

    Parameters
    ----------
    framework : DoubleMLFramework
        Fitted treatment contrast containing ``scaled_psi`` and
        repetition-specific coefficients.
    cluster_ids : array-like
        PSU identifier for every observation, in framework row order.
    level : float, default=0.95
        Confidence level for the reported interval.

    Returns
    -------
    dict[str, numpy.ndarray]
        Final ``coef``, ``se``, ``pval``, and confidence limits, plus
        repetition-specific ``se_rep`` and ``pval_rep`` arrays.
    """

    # Shape: N observations x J treatment contrasts x R repetitions.
    influence_scores = score_array_with_named_dimensions(
        framework.scaled_psi
    )

    # One-way cluster sandwich:
    # sum_g (sum_i in g influence_i)^2 / N^2.
    cluster_scores, _ = sum_rows_within_psu(
        influence_scores,
        cluster_ids,
    )

    # Keep one clustered SE for every contrast and repetition (shape J x R).
    # These are se_rep: they are intermediate repeated-cross-fitting results,
    # not the single SE eventually printed in the tables.
    standard_errors_by_repetition = np.sqrt(
        np.sum(cluster_scores ** 2, axis=0) / len(cluster_ids) ** 2
    )

    # all_thetas also has shape J x R. Each column contains the estimates from
    # one independently drawn cross-fitting split.
    coefficients_by_repetition = np.asarray(
        framework.all_thetas,
        dtype=float,
    )

    # The reported coefficient is the median estimate across repetitions
    # (shape J). With R=1 this is simply the estimate from that one run.
    coefficients = np.median(coefficients_by_repetition, axis=1)

    # Convert the J x R values in se_rep into one reported SE per contrast.
    # DoubleML does this by taking the median upper 95% bound and solving
    # backwards for the SE around the median coefficient.
    aggregated_upper = np.median(
        coefficients_by_repetition
        + 1.96 * standard_errors_by_repetition,
        axis=1,
    )
    standard_errors = (aggregated_upper - coefficients) / 1.96

    # pval_rep has shape J x R: one two-sided normal p-value for each
    # contrast and repetition, calculated with theta_rep / se_rep.
    p_values_by_repetition = 2 * norm.sf(np.abs(
        coefficients_by_repetition / standard_errors_by_repetition
    ))

    # pval has shape J and is the median p-value across repetitions. This is
    # the p-value saved in results_apos.pkl and used for table stars.
    p_values = np.median(p_values_by_repetition, axis=1)

    # Confidence limits are also constructed within each repetition first and
    # then aggregated by their median, matching DoubleMLFramework.confint().
    critical_value = norm.ppf(1 - (1 - float(level)) / 2)
    ci_lower = np.median(
        coefficients_by_repetition
        - critical_value * standard_errors_by_repetition,
        axis=1,
    )
    ci_upper = np.median(
        coefficients_by_repetition
        + critical_value * standard_errors_by_repetition,
        axis=1,
    )

    return {
        "coef": coefficients,
        "se": standard_errors,  # J reported SEs used in results and tables.
        "se_rep": standard_errors_by_repetition,  # J x R intermediate SEs.
        "pval": p_values,  # J reported p-values used for significance stars.
        "pval_rep": p_values_by_repetition,  # J x R intermediate p-values.
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def build_clustered_sensitivity_framework(framework, cluster_ids):
    """Prepare PSU-level scores for DoubleML's sensitivity calculations.

    Aggregating the scores once makes the sensitivity calculations both exact
    for the observation-weighted sandwich and much faster than repeatedly
    scanning every observation for every cluster.

    Parameters
    ----------
    framework : DoubleMLFramework
        Observation-level fitted model or treatment contrast.
    cluster_ids : array-like
        PSU identifier for each framework row.

    Returns
    -------
    DoubleMLFramework
        A result framework whose rows represent PSU-level pseudo-observations.
    """

    # These classes are result containers, not new estimators:
    # - DoubleMLCore holds estimates, SEs, influence scores, and sensitivity
    #   ingredients as arrays.
    # - DoubleMLFramework wraps that core and supplies methods such as
    #   sensitivity_analysis() and confint().
    # They are imported here, instead of at the top of the file, because this
    # specialized conversion is the only place that needs them.
    from doubleml.double_ml_framework import DoubleMLCore, DoubleMLFramework

    original_core = framework.dml_core
    cluster_scores, unique_clusters = sum_rows_within_psu(
        original_core.scaled_psi,
        cluster_ids,
    )
    number_of_clusters = len(unique_clusters)
    number_of_observations = len(cluster_ids)

    # Multiplication by G/N turns each cluster sum into a pseudo-observation.
    # The ordinary variance of these G pseudo-observations is exactly
    # sum_g(score_g^2) / N^2.
    cluster_scale = number_of_clusters / number_of_observations
    cluster_scores *= cluster_scale
    standard_errors_by_repetition = np.sqrt(
        np.mean(cluster_scores ** 2, axis=0) / number_of_clusters
    )

    sensitivity_elements = {
        key: value.copy()
        for key, value in original_core.sensitivity_elements.items()
    }
    # psi_max_bias is another observation-level score used only by sensitivity
    # analysis. It must be summed by PSU in exactly the same way as scaled_psi.
    cluster_bias_scores, _ = sum_rows_within_psu(
        sensitivity_elements["psi_max_bias"],
        cluster_ids,
    )
    cluster_bias_scores *= cluster_scale
    sensitivity_elements["psi_max_bias"] = cluster_bias_scores

    # Build a new result container whose "observations" are PSUs. No model is
    # fitted here; we are only repackaging already estimated score arrays so
    # DoubleML can run its standard sensitivity formulas at the PSU level.
    clustered_core = DoubleMLCore(
        all_thetas=original_core.all_thetas,
        all_ses=standard_errors_by_repetition,
        var_scaling_factors=np.full_like(
            original_core.var_scaling_factors,
            number_of_clusters,
        ),
        scaled_psi=cluster_scores,
        is_cluster_data=False,
        sensitivity_elements=sensitivity_elements,
    )
    treatment_names = None
    if framework.treatment_names is not None:
        treatment_names = list(framework.treatment_names)

    return DoubleMLFramework(
        clustered_core,
        treatment_names=treatment_names,
    )


def sensitivity_params(model_or_contrast, cluster_ids=None):
    """Calculate robustness values for an IRM or APOS estimand.

    Parameters
    ----------
    model_or_contrast : DoubleML model or framework
        Fitted estimand with sensitivity elements.
    cluster_ids : array-like or None
        PSU identifiers. When supplied, sensitivity uses PSU-level scores.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        RV and RV-alpha, one value per treatment contrast.
    """

    if cluster_ids is not None:
        model_or_contrast = build_clustered_sensitivity_framework(
            model_or_contrast,
            cluster_ids,
        )

    analyzed = model_or_contrast.sensitivity_analysis(
        cf_y=0.03,
        cf_d=0.03,
        rho=1.0,
        level=0.95,
    )
    parameters = analyzed.sensitivity_params
    # reshape(-1) flattens scalar/row/column outputs into the same predictable
    # one-dimensional vector: one robustness value per treatment contrast.
    robustness_value = np.asarray(
        parameters["rv"],
        dtype=float,
    ).reshape(-1)
    robustness_value_alpha = np.asarray(
        parameters["rva"],
        dtype=float,
    ).reshape(-1)
    return robustness_value, robustness_value_alpha


def format_coefficient(coefficient, standard_error, p_value=None):
    """Format one coefficient with conventional significance stars.

    Parameters
    ----------
    coefficient : float
        Point estimate.
    standard_error : float
        Reported SE, used only for a fallback p-value when necessary.
    p_value : float or None
        Preferred p-value, already aggregated across repetitions.

    Returns
    -------
    str
        Three-decimal coefficient followed by zero to three stars.
    """

    if standard_error == 0 or np.isnan(standard_error):
        return f"{coefficient:.3f}"

    if p_value is None:
        p_value = 2 * norm.sf(abs(coefficient / standard_error))

    if p_value < 0.01:
        stars = "***"
    elif p_value < 0.05:
        stars = "**"
    elif p_value < 0.10:
        stars = "*"
    else:
        stars = ""
    return f"{coefficient:.3f}{stars}"
