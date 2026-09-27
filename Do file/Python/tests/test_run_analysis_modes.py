"""Tests for the single public analysis command."""

import unittest

import numpy as np
from unittest.mock import patch

try:
    import analysis
    import run_analysis
except ModuleNotFoundError as error:
    if error.name != "run_analysis":
        raise
    run_analysis = None


class RunAnalysisModeTests(unittest.TestCase):
    def calls_for(self, *args, **kwargs):
        with patch("analysis.run_analysis") as execute:
            run_analysis.run(*args, **kwargs)
        return [
            (call.args[0].estimand, call.kwargs["fold_mode"], call.kwargs["stage"])
            for call in execute.call_args_list
        ]

    def test_default_runs_clustered_ate_then_att_in_stage_order(self):
        self.assertEqual(
            self.calls_for(),
            [("ATE", "clustered", "all"), ("ATT", "clustered", "all")],
        )

    def test_all_expands_clustered_before_unclustered(self):
        self.assertEqual(
            self.calls_for("all", estimand="ate", stage="effects"),
            [
                ("ATE", "clustered", "effects"),
                ("ATE", "unclustered", "effects"),
            ],
        )

    def test_estimand_and_stage_filters(self):
        self.assertEqual(
            self.calls_for(estimand="att", stage="gate"),
            [("ATT", "clustered", "gate")],
        )

    def test_each_execution_uses_the_analysis_seed_and_restores_state(self):
        draws = []

        def record_draw(*_args, **_kwargs):
            draws.append(np.random.random())

        np.random.seed(999)
        before = np.random.get_state()
        with patch("analysis.run_analysis", side_effect=record_draw):
            run_analysis.run(estimand="both", stage="gate")
        after = np.random.get_state()

        expected = np.random.RandomState(analysis.SEED).random_sample()
        self.assertEqual(draws, [expected, expected])
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_invalid_cli_value_exits_with_argparse_error(self):
        with self.assertRaises(SystemExit) as raised:
            run_analysis.main(["invalid"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
