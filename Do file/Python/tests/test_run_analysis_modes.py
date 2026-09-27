"""Tests for the single public analysis command."""

import unittest
from unittest.mock import patch

try:
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

    def test_invalid_cli_value_exits_with_argparse_error(self):
        with self.assertRaises(SystemExit) as raised:
            run_analysis.main(["invalid"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
