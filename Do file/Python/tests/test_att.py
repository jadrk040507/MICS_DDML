"""Regression checks for the ATT-specific MICS pipeline."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

import att


class AttPipelineTests(unittest.TestCase):
    def test_att_outputs_are_isolated_from_ate_outputs(self):
        self.assertEqual(att.ESTIMAND, "ATT")
        self.assertEqual(att.OUTPUT_DIR.name, "ATT")
        self.assertEqual(att.CHECKPOINT_DIR.parent, att.OUTPUT_DIR)

    def test_multivalued_att_weights_target_any_treated_household(self):
        levels = np.tile(np.asarray(att.TREATMENT_LEVELS), 20)
        frame = pd.DataFrame({
            "WQ15_g": levels,
            "x": np.linspace(-2, 2, len(levels)),
        })
        classifiers = [("logit", LogisticRegression(max_iter=1000))]

        with (
            patch.object(att, "FOLDS", 2),
            patch.object(att, "REPETITIONS", 1),
            patch.object(att, "INNER_FOLDS", 2),
            patch.object(att, "CLASSIFIERS", classifiers),
        ):
            splits = att.make_iid_splits(frame, "WQ15_g")
            weights = att.make_att_weights(
                frame,
                ["x"],
                splits,
                clustered=False,
            )

        treated = levels != 0
        self.assertEqual(weights["weights_bar"].shape, (len(frame), 1))
        self.assertTrue(np.all(weights["weights"][~treated] == 0))
        self.assertTrue(np.all(weights["weights"][treated] > 0))
        self.assertAlmostEqual(float(np.mean(weights["weights"])), 1.0)
        self.assertTrue(np.isfinite(weights["weights_bar"]).all())

    def test_binary_irm_uses_atte_score(self):
        rng = np.random.default_rng(42)
        n = 120
        x = rng.normal(size=n)
        treatment = (rng.uniform(size=n) < 0.5).astype(int)
        outcome = treatment + x + rng.normal(size=n)
        frame = pd.DataFrame({"y": outcome, "d": treatment, "x": x})

        with (
            patch.object(att, "FOLDS", 2),
            patch.object(att, "REPETITIONS", 1),
            patch.object(att, "INNER_FOLDS", 2),
            patch.object(att, "REGRESSORS", [("ols", LinearRegression())]),
            patch.object(
                att,
                "CLASSIFIERS",
                [("logit", LogisticRegression(max_iter=1000))],
            ),
        ):
            fitted = att.fit_irm(
                frame,
                ["x"],
                outcome="y",
                treatment="d",
                clustered=False,
            )

        self.assertEqual(fitted.score, "ATTE")
        self.assertTrue(np.isfinite(fitted.coef).all())

    def test_group_att_uses_weighted_ratio_normalization(self):
        psi_a = -np.asarray([2.0, 0.0, 0.0, 2.0])[:, None, None]
        psi_b = np.asarray([3.0, 1.0, 1.0, 5.0])[:, None, None]
        groups = pd.Series([1, 1, 2, 2])

        result = att.estimate_att_gate_from_scores(
            psi_a=psi_a,
            psi_b=psi_b,
            treatment_level="treated",
            group_values=groups,
            cluster_ids=None,
            group_labels={"1": "Low", "2": "High"},
            n_rep_boot=20,
        )

        np.testing.assert_allclose(result["coef"].to_numpy(), [2.0, 3.0])
        self.assertTrue(np.isfinite(result[["se", "pval"]]).all().all())


if __name__ == "__main__":
    unittest.main()
