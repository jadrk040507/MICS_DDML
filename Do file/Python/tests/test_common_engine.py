"""Equivalence checks for the Super Learner extracted from ATE and ATT."""

import ast
from pathlib import Path
import unittest

import numpy as np
from types import SimpleNamespace
from unittest.mock import patch
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

import _ate_impl as ate
import _att_impl as att
import _ddml_engine as ddml_engine


class CommonEngineEquivalenceTests(unittest.TestCase):
    def test_shared_inference_helpers_are_imported_by_both_estimands(self):
        for name in ("summary_with_clustered_inference", "estimate_gate_from_contrast"):
            with self.subTest(helper=name):
                shared = getattr(ddml_engine, name)
                self.assertIs(getattr(ate, name), shared)
                self.assertIs(getattr(att, name), shared)

    def test_convex_weights_reject_nonfinite_predictions(self):
        predictions = np.array([[np.nan, 0.2], [0.3, 0.4]])
        target = np.array([0.0, 1.0])

        with self.assertRaisesRegex(ValueError, "finite"):
            ddml_engine.convex_weights(predictions, target)

    def test_convex_weights_reject_unsuccessful_optimization(self):
        failed = SimpleNamespace(
            success=False,
            x=np.array([0.5, 0.5]),
            message="iteration limit reached",
        )
        with patch.object(ddml_engine, "minimize", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "iteration limit"):
                ddml_engine.convex_weights(
                    np.array([[0.0, 1.0], [1.0, 0.0]]),
                    np.array([0.0, 1.0]),
                )

    def test_both_estimands_use_the_same_engine_objects(self):
        self.assertIs(ate.ConvexRegressor, ddml_engine.ConvexRegressor)
        self.assertIs(att.ConvexRegressor, ddml_engine.ConvexRegressor)
        self.assertIs(ate.ConvexClassifier, ddml_engine.ConvexClassifier)
        self.assertIs(att.ConvexClassifier, ddml_engine.ConvexClassifier)
        self.assertIs(ate.convex_weights, ddml_engine.convex_weights)
        self.assertIs(att.convex_weights, ddml_engine.convex_weights)
        shared_functions = (
            "collect_convex_weights",
            "score_array_with_named_dimensions",
            "sum_rows_within_psu",
            "cluster_robust_framework_inference",
            "build_clustered_sensitivity_framework",
            "sensitivity_params",
            "format_coefficient",
        )
        for name in shared_functions:
            self.assertIs(getattr(ate, name), getattr(ddml_engine, name))
            self.assertIs(getattr(att, name), getattr(ddml_engine, name))

    def test_estimand_scripts_no_longer_define_prediction_engines(self):
        forbidden = {
            "convex_weights",
            "ConvexRegressor",
            "ConvexClassifier",
            "collect_convex_weights",
            "score_array_with_named_dimensions",
            "sum_rows_within_psu",
            "cluster_robust_framework_inference",
            "build_clustered_sensitivity_framework",
            "sensitivity_params",
            "format_coefficient",
        }
        for module in (ate, att):
            source = Path(module.__file__).read_text(encoding="utf-8")
            definitions = {
                node.name
                for node in ast.parse(source).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            }
            self.assertTrue(forbidden.isdisjoint(definitions))

    def test_convex_weights_match_pre_extraction_reference(self):
        predictions = np.array([
            [0.1, 0.3, 0.2],
            [0.3, 0.2, 0.4],
            [0.8, 0.6, 0.7],
            [0.9, 0.7, 0.8],
            [0.4, 0.5, 0.3],
            [0.6, 0.4, 0.5],
        ])
        regression_target = np.array([0.15, 0.25, 0.75, 0.85, 0.45, 0.55])
        classification_target = np.array([0, 0, 1, 1, 0, 1])

        np.testing.assert_allclose(
            ddml_engine.convex_weights(predictions, regression_target),
            [0.725443687, 0.274556313, 0.0],
            rtol=0,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            ddml_engine.convex_weights(
                predictions,
                classification_target,
                classification=True,
            ),
            [1.0, 0.0, 0.0],
            rtol=0,
            atol=1e-8,
        )

    def test_regressor_matches_pre_extraction_predictions(self):
        x = np.column_stack([
            np.arange(18, dtype=float),
            np.tile([0.0, 1.0, 2.0], 6),
        ])
        y = 0.5 * x[:, 0] - 0.2 * x[:, 1] + np.sin(x[:, 0]) / 10
        learner = ddml_engine.ConvexRegressor(
            [("ols", LinearRegression()), ("ridge", Ridge(alpha=1.0))],
            random_state=11,
            inner_folds=3,
        ).fit(x, y)

        np.testing.assert_allclose(learner.weights_, [0.5, 0.5], atol=1e-12)
        np.testing.assert_allclose(
            learner.predict(x[:4]),
            [0.01564669, 0.32306645, 0.63048622, 1.50800753],
            rtol=0,
            atol=1e-8,
        )

    def test_classifier_matches_pre_extraction_probabilities(self):
        x = np.column_stack([
            np.arange(18, dtype=float),
            np.tile([0.0, 1.0, 2.0], 6),
        ])
        y = np.array([0, 1, 0, 1, 0, 1] * 3)
        learner = ddml_engine.ConvexClassifier(
            [
                (
                    "small",
                    LogisticRegression(
                        C=0.1,
                        solver="liblinear",
                        random_state=11,
                    ),
                ),
                (
                    "large",
                    LogisticRegression(
                        C=10,
                        solver="liblinear",
                        random_state=11,
                    ),
                ),
            ],
            random_state=11,
            inner_folds=3,
        ).fit(x, y)

        np.testing.assert_allclose(learner.weights_, [1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(
            learner.predict_proba(x[:4]),
            [
                [0.50570650, 0.49429350],
                [0.50739420, 0.49260580],
                [0.50909169, 0.49090831],
                [0.49322463, 0.50677537],
            ],
            rtol=0,
            atol=1e-8,
        )


if __name__ == "__main__":
    unittest.main()

