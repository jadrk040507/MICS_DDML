"""Joint influence-based inference for ATE, ATT, ATU, and treatment share."""

import numpy as np
import pandas as pd


def joint_effects(a, t, influence_a, influence_t, treated, clusters):
    """Return (ATE, ATT, ATU, treatment share) and clustered covariance.

    ATU is derived as (ATE - p * ATT) / (1 - p). Its influence function
    includes the estimated treatment share and the covariance with both
    estimated treatment effects.
    """
    d = np.asarray(treated, dtype=float)
    ia = np.asarray(influence_a, dtype=float)
    it = np.asarray(influence_t, dtype=float)
    cluster_values = np.asarray(clusters)
    if d.ndim != 1 or not np.isin(d, [0, 1]).all():
        raise ValueError("Treatment must be a binary vector")
    if ia.shape != d.shape or it.shape != d.shape or cluster_values.shape != d.shape:
        raise ValueError("Influences, treatment, and clusters must align")
    if (not np.isfinite(np.r_[a, t, ia, it]).all()
            or pd.isna(cluster_values).any()):
        raise ValueError("Nonfinite inputs or missing clusters")

    p = float(d.mean())
    if not 0 < p < 1:
        raise ValueError("Both target populations are required")

    atu = (a - p * t) / (1 - p)
    influence_p = d - p
    influence_atu = (ia - p * it + (atu - t) * influence_p) / (1 - p)
    influences = np.column_stack((ia, it, influence_atu, influence_p))
    codes, labels = pd.factorize(cluster_values)
    if len(labels) < 2:
        raise ValueError("At least two PSUs are required")
    cluster_sums = np.zeros((len(labels), influences.shape[1]))
    np.add.at(cluster_sums, codes, influences)
    covariance = cluster_sums.T @ cluster_sums / len(d) ** 2
    return np.array([a, t, atu, p], dtype=float), covariance


def linear_combination(theta, covariance, weights):
    """Return a linear contrast and its standard error from joint covariance."""
    theta = np.asarray(theta, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (4,) or not np.isfinite(weights).all():
        raise ValueError("Provide four finite weights: ATE, ATT, ATU, p")
    if theta.shape != (4,) or covariance.shape != (4, 4):
        raise ValueError("Expected four estimates and a 4 by 4 covariance")
    if not np.isfinite(theta).all() or not np.isfinite(covariance).all():
        raise ValueError("Estimates and covariance must be finite")
    estimate = float(weights @ theta)
    variance = float(weights @ covariance @ weights)
    return estimate, float(np.sqrt(max(0.0, variance)))
