"""Reproducible observation and PSU fold construction."""
import unittest

import numpy as np
import pandas as pd

from _cross_fitting import (
    make_cluster_split_metadata,
    make_cluster_splits,
    make_iid_splits,
    validate_splits,
)


class CrossFittingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iid_frame = pd.DataFrame({
            "D": np.tile([0, 1, 2], 30),
        })
        cls.cluster_frame = pd.DataFrame({
            "D": np.tile([0, 1], 24),
            "Cluster_var": np.repeat(np.arange(24), 2),
        })

    def test_iid_splits_are_reproducible(self):
        first = make_iid_splits(
            self.iid_frame, "D", n_folds=3, repetitions=2, seed=42
        )
        second = make_iid_splits(
            self.iid_frame, "D", n_folds=3, repetitions=2, seed=42
        )
        for rep_a, rep_b in zip(first, second):
            for (train_a, test_a), (train_b, test_b) in zip(rep_a, rep_b):
                np.testing.assert_array_equal(train_a, train_b)
                np.testing.assert_array_equal(test_a, test_b)

    def test_cluster_splits_have_no_psu_overlap(self):
        splits = make_cluster_splits(
            self.cluster_frame, "D", n_folds=3, repetitions=2, seed=42
        )
        groups = self.cluster_frame["Cluster_var"].to_numpy()
        for repetition in splits:
            for train, test in repetition:
                self.assertFalse(set(groups[train]) & set(groups[test]))

    def test_each_fold_retains_all_treatment_levels(self):
        frame = self.iid_frame
        splits = make_iid_splits(
            frame, "D", n_folds=3, repetitions=2, seed=42
        )
        expected = np.unique(frame["D"])
        for repetition in splits:
            for train, test in repetition:
                np.testing.assert_array_equal(np.unique(frame.D.iloc[train]), expected)
                np.testing.assert_array_equal(np.unique(frame.D.iloc[test]), expected)
        validate_splits(splits, frame.D.to_numpy())

    def test_cluster_metadata_matches_observation_splits(self):
        frame = self.cluster_frame
        splits = make_cluster_splits(
            frame, "D", n_folds=3, repetitions=2, seed=42
        )
        metadata = make_cluster_split_metadata(frame, splits)
        groups = frame.Cluster_var.to_numpy()
        for rep_splits, rep_metadata in zip(splits, metadata):
            for (train, test), (metadata_train, metadata_test) in zip(
                rep_splits, rep_metadata
            ):
                np.testing.assert_array_equal(metadata_train[0], np.unique(groups[train]))
                np.testing.assert_array_equal(metadata_test[0], np.unique(groups[test]))


if __name__ == "__main__":
    unittest.main()
