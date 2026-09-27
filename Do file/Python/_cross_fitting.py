"""Shared, reproducible fold construction for ATE and ATT."""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


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
