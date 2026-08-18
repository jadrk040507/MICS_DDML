"""Statistical and infrastructure helpers for the MICS DoubleML pipeline.

The functions cover clustered inference, sensitivity and GATE post-estimation,
and reproducible disk caching for expensive fitted models. The cache is an
implementation detail and must not change the estimand, sample, folds, or
variance estimator used by the analysis.
"""

import hashlib
import joblib
from pathlib import Path

import numpy as np
from joblib import Memory, hash as joblib_hash


OUTCOME_LABELS = {
    "SomeRiskHome": "Some Risk Home (E.coli 1--100 CFU)",
    "VeryHighRiskHome": "Very High Risk Home (E.coli $\\geq$ 101 CFU)",
    "diarrhea": "Diarrhea (under-5)",
}


class LazyCheckpointBundle:
    """Load one fitted checkpoint only when a downstream step needs it."""

    def __init__(self, paths, metadata):
        self.paths = paths
        self.metadata = metadata

    def __getitem__(self, key):
        if key in self.metadata:
            return self.metadata[key]
        loaded = joblib.load(self.paths[key])
        if key == "apos_cluster":
            return loaded["model"]
        if key == "apos_cluster_se":
            return loaded["cluster_se"]
        return loaded

    def get(self, key, default=None):
        if key in self.metadata:
            return self.metadata[key]
        if key not in self.paths:
            return default
        return self[key]


def relative_robustness_value(reduced_value, full_value, denominator_floor=1e-8):
    """Compare robustness after removing a control group.

    The returned ratio is reduced-specification RV divided by full-
    specification RV: one means unchanged robustness, below one means lower
    robustness, and above one means higher robustness.

    A ratio is not informative when either value is missing/non-finite or when
    the full-specification RV is effectively zero.  In those cases ``None`` is
    returned so table builders can display an em dash instead of a misleading
    extreme ratio.
    """

    if reduced_value is None or full_value is None:
        return None
    try:
        reduced = float(reduced_value)
        full = float(full_value)
    except (TypeError, ValueError):
        return None
    if not (reduced == reduced and full == full):  # NaN check
        return None
    if abs(full) < denominator_floor:
        return None
    return reduced / full


def cluster_robust_framework_se(framework, cluster_ids, smpls):
    """Compute one-way cluster-robust SEs from DoubleML influence scores.

    Why this function is needed
    ---------------------------
    ``DoubleMLIRM`` accepts ``cluster_cols`` directly. Older versions of
    ``DoubleMLAPOS`` do not. For APOS, we therefore fit the model with folds
    that keep each sampling cluster together and calculate the cluster
    variance from the fitted orthogonal-score influence functions.

    Estimator
    ---------
    Let ``psi_i(theta)`` be the scaled DoubleML influence score and let ``c(i)``
    identify the sampling cluster. For each cross-fitting repetition, we sum
    the influence scores within each cluster:

        Gamma_c = sum_{i in cluster c} psi_i(theta)

    The one-way cluster sandwich is proportional to the average of
    ``Gamma_c^2`` across clusters. The implementation below computes that
    quantity separately for every target parameter and repetition, then takes
    the square root to obtain a standard error. The final repetition summary
    follows DoubleML's median-confidence-bound aggregation.

    Required conditions
    -------------------
    * ``cluster_ids`` has one entry per observation in ``scaled_psi``.
    * ``smpls`` contains one list of train/test folds per repetition.
    * Every cluster appears in exactly one test fold within each repetition.
      This is what makes the cluster score sums non-overlapping.
    * The model must have been fitted with the same sample splitting supplied
      in ``smpls``.

    This function changes standard errors only. It does not change the APOS
    point estimates, nuisance predictions, treatment levels, or estimand.
    """

    psi = np.asarray(framework.scaled_psi, dtype=float)
    if psi.ndim == 2:
        psi = psi[:, :, None]
    cluster_ids = np.asarray(cluster_ids)
    if psi.shape[0] != cluster_ids.shape[0]:
        raise ValueError("cluster_ids must match the framework score rows.")
    if len(smpls) != psi.shape[2]:
        raise ValueError("smpls must contain one split list per repetition.")

    n_clusters = np.unique(cluster_ids).size
    if n_clusters < 2:
        raise ValueError("At least two clusters are required for clustered SEs.")

    all_ses = np.empty((psi.shape[1], psi.shape[2]), dtype=float)
    for rep, rep_smpls in enumerate(smpls):
        # The one-way cluster formula below requires cluster-level test folds:
        # every cluster must contribute to exactly one test fold per repetition.
        tested_clusters = np.concatenate([
            np.unique(cluster_ids[test_indices])
            for _, test_indices in rep_smpls
        ])
        _, test_counts = np.unique(tested_clusters, return_counts=True)
        if tested_clusters.size != n_clusters or np.any(test_counts != 1):
            raise ValueError(
                "Cluster-robust scores require each cluster to appear in "
                "exactly one test fold per repetition."
            )

        # A cluster must never be split between the training and test portion
        # of one fold. This is a second, explicit guard in addition to the
        # test-fold coverage check above.
        for train_indices, test_indices in rep_smpls:
            train_clusters = set(np.unique(cluster_ids[train_indices]))
            test_clusters = set(np.unique(cluster_ids[test_indices]))
            if train_clusters.intersection(test_clusters):
                raise ValueError(
                    "A sampling cluster appears in both train and test "
                    "within one fold."
                )

        gamma = np.zeros(psi.shape[1], dtype=float)
        for _, test_indices in rep_smpls:
            test_clusters = np.unique(cluster_ids[test_indices])
            if test_clusters.size == 0:
                continue
            for cluster in test_clusters:
                in_test_fold = np.zeros(cluster_ids.shape[0], dtype=bool)
                in_test_fold[test_indices] = True
                mask = (cluster_ids == cluster) & in_test_fold
                cluster_score = psi[mask, :, rep].sum(axis=0)
                gamma += cluster_score ** 2 / test_clusters.size
        sigma2 = gamma / n_clusters
        all_ses[:, rep] = np.sqrt(np.maximum(sigma2, 0.0))

    theta = np.asarray(framework.all_thetas, dtype=float)
    point = np.median(theta, axis=1)
    upper = np.median(theta + 1.96 * all_ses, axis=1)
    return (upper - point) / 1.96


def as_clustered_framework(framework, cluster_ids, smpls):
    """Attach cluster-level inference metadata to a DoubleML framework.

    DoubleML 0.11.3 does not expose the required clustered APOS framework
    directly. This preserves fitted point estimates and sensitivity elements,
    replaces standard errors with the custom cluster sandwich, and supplies
    cluster-level fold metadata for the native sensitivity calculation.
    """

    from doubleml.double_ml_framework import DoubleMLCore, DoubleMLFramework

    cluster_ids = np.asarray(cluster_ids)
    n_clusters = np.unique(cluster_ids).size
    clustered_se = cluster_robust_framework_se(framework, cluster_ids, smpls)
    all_ses = np.repeat(clustered_se[:, None], framework.n_rep, axis=1)
    smpls_cluster = []
    for rep_smpls in smpls:
        rep_clusters = []
        for train_indices, test_indices in rep_smpls:
            rep_clusters.append((
                [np.unique(cluster_ids[train_indices])],
                [np.unique(cluster_ids[test_indices])],
            ))
        smpls_cluster.append(rep_clusters)

    core = framework.dml_core
    clustered_core = DoubleMLCore(
        all_thetas=core.all_thetas,
        all_ses=all_ses,
        var_scaling_factors=np.full_like(core.var_scaling_factors, n_clusters),
        scaled_psi=core.scaled_psi,
        is_cluster_data=True,
        cluster_dict={
            "smpls": smpls,
            "smpls_cluster": smpls_cluster,
            "cluster_vars": cluster_ids.reshape(-1, 1),
            "n_folds_per_cluster": 1,
        },
        sensitivity_elements=core.sensitivity_elements,
    )
    return DoubleMLFramework(
        clustered_core,
        treatment_names=list(framework.treatment_names)
        if framework.treatment_names is not None else None,
    )


def sha256_file(path):
    """Return a stable SHA-256 digest for a local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_cache_context(source_path, data_paths, config, doubleml_version):
    """Create cache inputs while excluding presentation-only code."""

    source_path = Path(source_path)
    source_digest = hashlib.sha256()
    source_text = source_path.read_text(encoding="utf-8")
    estimation_source = source_text.split("\n# Results\n", 1)[0]
    source_digest.update(estimation_source.encode("utf-8"))
    source_digest.update(
        str(config.get("cache_schema_version", 1)).encode("ascii")
    )

    return {
        "source_sha256": source_digest.hexdigest(),
        "data_sha256": {
            name: sha256_file(path) for name, path in data_paths.items()
        },
        "doubleml_version": doubleml_version,
        **config,
    }


def make_estimate_key(context, dataset, outcome, model, clustered):
    """Create a stable key for one estimation specification."""

    payload = {
        **context,
        "dataset": dataset,
        "outcome": outcome,
        "model": model,
        "clustered": clustered,
    }
    return joblib_hash(payload)


class EstimateCache:
    """Disk-backed cache for fitted DoubleML objects.

    ``joblib.Memory`` owns serialization, metadata, and cache discovery. The
    callable is deliberately ignored when hashing: the explicit estimation
    key remains the authority, so presentation-only code cannot invalidate a
    fitted model.
    """

    def __init__(self, directory):
        self.memory = Memory(location=Path(directory), verbose=0)
        self._cached_fit = self.memory.cache(_fit_callable, ignore=["fit"])

    def contains(self, key):
        return self._cached_fit.check_call_in_cache(key, fit=None)

    def get_or_fit(self, key, fit):
        return self._cached_fit(key, fit)

    def load(self, key):
        """Load a previously cached fit without retaining a second copy."""

        if not self.contains(key):
            raise KeyError(f"No cached estimate exists for key {key}.")
        # ``fit`` is ignored by the cached callable, so None addresses the
        # existing cache entry without executing a new estimation.
        return self._cached_fit(key, None)


def _fit_callable(key, fit):
    """Execute a fit; persistence is supplied by ``EstimateCache``."""

    del key
    return fit()


def select_column_groups(x_columns, group_patterns, child=False):
    """Select model-matrix columns using named groups of prefixes/names."""

    groups = {}
    for group, patterns in group_patterns.items():
        if group == "child_age_sex" and not child:
            continue
        patterns = (patterns,) if isinstance(patterns, str) else patterns
        columns = [
            column for column in x_columns
            if any(
                column == pattern or column.startswith(pattern)
                for pattern in patterns
            )
        ]
        if columns:
            groups[group] = columns
    return groups


def cached_reduced_fit(cache, context, dataset, outcome, kind,
                       specification, group, fit):
    """Fit a reduced specification once and retrieve it from the cache later."""

    key = make_estimate_key(
        context,
        dataset,
        outcome,
        f"sensitivity_{kind}_{specification}_{group}",
        specification == "clustered_folds",
    )
    return cache.get_or_fit(key, fit)


def sensitivity_params(model_or_contrast, cluster_ids=None, smpls=None):
    """Compute RV and RV-alpha for an IRM or APOS estimand.

    Clustered APOS contrasts are first rebuilt with the cluster-robust
    framework so the sensitivity calculation uses the same variance scale as
    the reported clustered effect.
    """

    if cluster_ids is not None:
        if smpls is None:
            raise ValueError("smpls is required for clustered sensitivity.")
        model_or_contrast = as_clustered_framework(
            model_or_contrast, cluster_ids, smpls
        )

    # DoubleML mutates the fitted framework in place and returns that same
    # framework.  This distinction matters for DiskBackedEstimate: each
    # attribute access reloads a fresh cached copy, so calling
    # ``model_or_contrast.sensitivity_analysis()`` and then reading
    # ``model_or_contrast.sensitivity_params`` would discard the mutation.
    analyzed = model_or_contrast.sensitivity_analysis(
        cf_y=0.03, cf_d=0.03, rho=1.0, level=0.95
    )
    params = analyzed.sensitivity_params
    if params is None:
        raise ValueError("DoubleML sensitivity parameters are unavailable.")
    return (
        np.asarray(params["rv"], dtype=float).reshape(-1),
        np.asarray(params["rva"], dtype=float).reshape(-1),
    )


def estimate_gate_from_contrast(contrast, treatment_level, effect_index,
                                group_values, cluster_ids=None,
                                group_labels=None, level=0.95,
                                n_rep_boot=500):
    """Estimate group-specific effects by projecting an existing score.

    This post-estimation BLP/GATE projection reuses an orthogonal score and
    does not refit nuisance learners. Group indicators make each coefficient
    the average treatment effect for one group. Clustered projections use a
    one-way cluster covariance; ordinary projections use HC0 covariance.
    The reported BLP R-squared is descriptive, not a causal-model fit.
    """

    import doubleml as dml
    import pandas as pd

    scaled_psi = np.asarray(contrast.scaled_psi, dtype=float)
    if scaled_psi.ndim == 2:
        scaled_psi = scaled_psi[:, :, None]
    if effect_index < 0 or effect_index >= scaled_psi.shape[1]:
        raise IndexError("effect_index is outside the contrast score array.")
    values = pd.Series(group_values).reset_index(drop=True)
    valid = values.notna().to_numpy()
    values = values.loc[valid].reset_index(drop=True)
    scaled_psi = scaled_psi[valid, :, :]
    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)[valid]
    groups = pd.get_dummies(values.astype("string"), prefix="group", dtype=float)
    groups = groups.loc[:, groups.sum(axis=0) > 0]
    if groups.shape[1] < 1:
        raise ValueError("The grouping variable has no observed groups.")
    if scaled_psi.shape[0] != len(values):
        raise ValueError("Grouping variable, clusters, and APOS scores must be aligned.")
    if cluster_ids is not None and len(values) != len(cluster_ids):
        raise ValueError("Grouping variable, clusters, and APOS scores must be aligned.")

    blp = dml.DoubleMLBLP(
        scaled_psi[:, effect_index, :],
        basis=groups,
        is_gate=True,
    )
    if cluster_ids is None:
        blp.fit(cov_type="HC0")
    else:
        blp.fit(
            cov_type="cluster",
            cov_kwds={"groups": np.asarray(cluster_ids)},
        )
    ci_point = blp.confint(joint=False, level=float(level))
    ci_joint = blp.confint(
        joint=True,
        level=float(level),
        n_rep_boot=int(n_rep_boot),
    )

    labels = group_labels or {}
    r2_by_rep = [float(model.rsquared) for model in blp._blp_model]
    r2 = float(np.median(r2_by_rep))
    rows = []
    for column in groups.columns:
        raw_value = column.removeprefix("group_")
        label = labels.get(raw_value)
        if label is None:
            try:
                label = labels.get(str(int(float(raw_value))))
            except (TypeError, ValueError):
                label = None
        if label is None:
            label = raw_value
        group_mask = values.astype("string").eq(raw_value)
        n_psu = (
            pd.Series(np.asarray(cluster_ids)[group_mask.to_numpy()]).nunique()
            if cluster_ids is not None else np.nan
        )
        summary_row = blp.summary.loc[column]
        rows.append({
            "treatment_level": treatment_level,
            "group_value": raw_value,
            "group_label": label,
            "n": int(groups[column].sum()),
            "n_psu": int(n_psu) if n_psu == n_psu else np.nan,
            "r2": r2,
            "coef": float(summary_row["coef"]),
            "se": float(summary_row["std err"]),
            "pval": float(summary_row["P>|t|"]),
            "ci_lower": float(ci_point.loc[column].iloc[0]),
            "ci_upper": float(ci_point.loc[column].iloc[-1]),
            "ci_lower_joint": float(ci_joint.loc[column].iloc[0]),
            "ci_upper_joint": float(ci_joint.loc[column].iloc[-1]),
        })
    return pd.DataFrame(rows)
