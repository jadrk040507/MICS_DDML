"""Tests for the unified ATE and ATT workflow."""

import unittest
from unittest.mock import patch

try:
    import analysis
except ModuleNotFoundError as error:
    if error.name != "analysis":
        raise
    analysis = None


class AnalysisWorkflowTests(unittest.TestCase):
    def test_ate_and_att_use_same_estimation_function(self):
        self.assertTrue(callable(getattr(analysis, "estimate_one_outcome", None)))
        self.assertTrue(callable(getattr(analysis, "estimate_all_models", None)))

    def test_att_spec_selects_atte_and_weight_strategy(self):
        spec = analysis.get_analysis_spec("att")
        self.assertEqual(spec.score, "ATTE")
        self.assertTrue(spec.uses_att_weights)
        self.assertTrue(spec.att_gate_strategy)

    def test_ate_spec_does_not_construct_att_weights(self):
        spec = analysis.get_analysis_spec("ate")
        with patch.object(analysis, "make_att_weights") as make_weights:
            result = analysis._weights_for_apos(spec, None, None, None, False)
        self.assertIsNone(result)
        make_weights.assert_not_called()

    def test_fold_modes_keep_canonical_estimand_output_directory(self):
        for estimand in ("ate", "att"):
            spec = analysis.get_analysis_spec(estimand)
            for fold_mode in ("clustered", "unclustered", "both"):
                with analysis.use_analysis_spec(spec, fold_mode):
                    self.assertEqual(analysis.OUTPUT_DIR, spec.output_dir)

    def test_analysis_rejects_unknown_estimand(self):
        with self.assertRaisesRegex(ValueError, "estimand"):
            analysis.get_analysis_spec("atu")


if __name__ == "__main__":
    unittest.main()
