"""Shared Super Learner engine used by the ATE and ATT analyses.

This module contains only prediction machinery.  The causal estimand remains
explicit in ``ate.py`` or ``att.py``; sharing this code therefore removes
duplication without turning ATE and ATT into the same analysis.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)


def convex_weights(predictions, target, classification=False):
    """Choose nonnegative Super Learner weights that add to one."""

    predictions = np.asarray(predictions, dtype=float)
    target = np.asarray(target)
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
    weights = np.clip(result.x, 0, 1)
    return weights / weights.sum()


def positive_probability(model, x):
    """Return the probability assigned to class 1."""

    positive = int(np.where(np.asarray(model.classes_) == 1)[0][0])
    return model.predict_proba(x)[:, positive]


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

        for learner_number, (name, estimator) in enumerate(self.estimators):
            fold_models = []
            for train, test in splits:
                fitted = clone(estimator).fit(x[train], y[train])
                out_of_fold[test, learner_number] = fitted.predict(x[test])
                fold_models.append(fitted)
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

        for learner_number, (name, estimator) in enumerate(self.estimators):
            fold_models = []
            for train, test in splits:
                fitted = clone(estimator).fit(x[train], y[train])
                out_of_fold[test, learner_number] = positive_probability(
                    fitted,
                    x[test],
                )
                fold_models.append(fitted)
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
