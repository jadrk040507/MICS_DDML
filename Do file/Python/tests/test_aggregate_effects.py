"""Repeated inference must aggregate estimates before reporting uncertainty."""

from types import SimpleNamespace
import unittest

import numpy as np
from scipy.stats import norm

from analysis import cluster_robust_framework_inference


class AggregateEffectsTests(unittest.TestCase):
    def test_clustered_repetitions_use_median_coefficients_and_pvalues(self):
        coefficients = np.array([[1.0, 5.0, 3.0]])
        errors = np.array([[2.0, 0.1, 0.2]])
        scores = np.zeros((2, 1, 3))
        scores[0, 0, :] = 2 * errors[0]
        result = cluster_robust_framework_inference(
            SimpleNamespace(scaled_psi=scores, all_thetas=coefficients),
            np.array([0, 1]),
        )
        self.assertEqual(result["coef"][0], 3.0)
        self.assertAlmostEqual(
            result["pval"][0],
            np.median(2 * norm.sf(np.abs(coefficients / errors))),
        )
        self.assertNotAlmostEqual(result["se"][0], np.median(errors))


if __name__ == "__main__":
    unittest.main()
