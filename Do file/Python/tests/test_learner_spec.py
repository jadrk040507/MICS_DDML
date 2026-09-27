"""The production learner library keeps nested tuning bounded."""
import unittest

import _ate_impl as ate
import _att_impl as att
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor


class LearnerSpecificationTests(unittest.TestCase):
    def test_fast_candidate_libraries_are_compact_and_consistent(self):
        expected_regressors = ["ols", "ridge", "random_forest", "xgboost"]
        expected_classifiers = ["logit", "random_forest", "xgboost"]
        for module in (ate, att):
            with self.subTest(module=module.__name__):
                self.assertEqual(
                    [name for name, _ in module.REGRESSORS], expected_regressors
                )
                self.assertEqual(
                    [name for name, _ in module.CLASSIFIERS], expected_classifiers
                )
                self.assertEqual((module.FOLDS, module.REPETITIONS), (5, 3))
                for _, model in module.REGRESSORS + module.CLASSIFIERS:
                    if isinstance(model, (RandomForestRegressor, RandomForestClassifier,
                                          XGBRegressor, XGBClassifier)):
                        self.assertEqual(model.n_estimators, 50)


if __name__ == "__main__":
    unittest.main()
