"""Learner-level console output is opt-in to keep production logs compact."""
import os
from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LinearRegression

from _ddml_engine import ConvexRegressor


class CompactProgressTests(unittest.TestCase):
    def fit_with_verbosity(self, setting):
        x = np.arange(24, dtype=float).reshape(12, 2)
        y = x[:, 0] * .4 + x[:, 1] * .2
        learner = ConvexRegressor(
            [("ols", LinearRegression())], inner_folds=2
        )
        output = StringIO()
        with patch.dict(os.environ, {"DDML_VERBOSE_LEARNERS": setting}):
            with redirect_stdout(output):
                learner.fit(x, y)
        return output.getvalue()

    def test_default_learner_fit_is_quiet(self):
        self.assertEqual(self.fit_with_verbosity("0"), "")

    def test_verbose_learner_fit_can_be_enabled(self):
        output = self.fit_with_verbosity("1")
        self.assertIn("regression] ols", output)
        self.assertIn("done in", output)


if __name__ == "__main__":
    unittest.main()
