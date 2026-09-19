"""Regression checks for repeated-cross-fitting inference and file isolation."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import norm

import ate


class FakeFramework:
    def __init__(self, scaled_psi, all_thetas):
        self.scaled_psi = scaled_psi
        self.all_thetas = all_thetas


class RepeatedInferenceTests(unittest.TestCase):
    def test_clustered_inference_aggregates_pvalues_and_intervals_by_repetition(self):
        cluster_ids = np.array([0, 0, 1, 1, 2, 2])
        se_rep = np.array([[0.05, 0.20, 0.05]])
        scaled_psi = np.zeros((len(cluster_ids), 1, 3))
        scaled_psi[0, 0, :] = len(cluster_ids) * se_rep[0]
        theta_rep = np.array([[0.01, 0.20, 0.40]])
        framework = FakeFramework(scaled_psi, theta_rep)

        result = ate.cluster_robust_framework_inference(
            framework,
            cluster_ids,
        )

        theta = np.median(theta_rep, axis=1)
        expected_p = np.median(
            2 * norm.sf(np.abs(theta_rep / se_rep)),
            axis=1,
        )
        critical = norm.ppf(0.975)
        expected_lower = np.median(
            theta_rep - critical * se_rep,
            axis=1,
        )
        expected_upper = np.median(
            theta_rep + critical * se_rep,
            axis=1,
        )
        aggregated_upper = np.median(
            theta_rep + 1.96 * se_rep,
            axis=1,
        )
        expected_se = (aggregated_upper - theta) / 1.96

        np.testing.assert_allclose(result["se_rep"], se_rep)
        np.testing.assert_allclose(result["se"], expected_se)
        np.testing.assert_allclose(result["pval"], expected_p)
        np.testing.assert_allclose(result["ci_lower"], expected_lower)
        np.testing.assert_allclose(result["ci_upper"], expected_upper)

    def test_significance_stars_use_the_reported_pvalue(self):
        self.assertEqual(
            ate.format_coefficient(1.0, 0.1, p_value=0.20),
            "1.000",
        )
        self.assertEqual(
            ate.format_coefficient(0.01, 1.0, p_value=0.009),
            "0.010***",
        )

    def test_sample_pickles_have_a_separate_suffix(self):
        full = ate.result_pickle_path("results_apos", quick_sample=False)
        sample = ate.result_pickle_path("results_apos", quick_sample=True)
        self.assertEqual(full.name, "results_apos.pkl")
        self.assertEqual(sample.name, "results_apos_sample05.pkl")
        self.assertNotEqual(Path(full), Path(sample))

    def test_checkpoint_store_separates_and_roundtrips_full_and_sample(self):
        """Generic checkpoints share one interface without sharing filenames."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            with patch.object(ate, "CHECKPOINT_DIR", temporary_path):
                full_store = ate.CheckpointStore(quick_sample=False)
                sample_store = ate.CheckpointStore(quick_sample=True)

                self.assertEqual(full_store.path("example").name, "example.pkl")
                self.assertEqual(
                    sample_store.path("example").name,
                    "example_sample05.pkl",
                )

                full_store.save("example", [{"run": "full"}])
                sample_store.save("example", [{"run": "sample"}])

                self.assertEqual(full_store.load("example"), [{"run": "full"}])
                self.assertEqual(
                    sample_store.load("example"),
                    [{"run": "sample"}],
                )


if __name__ == "__main__":
    unittest.main()
