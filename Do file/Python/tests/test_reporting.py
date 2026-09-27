"""Tests for the shared reporting stage boundary."""

from types import SimpleNamespace
import unittest

try:
    import reporting
except ModuleNotFoundError as error:
    if error.name != "reporting":
        raise
    reporting = None


class ReportingTests(unittest.TestCase):
    def test_reporting_exports_all_three_stage_functions(self):
        for name in ("save_effect_outputs", "run_sensitivity", "run_gate", "write_manifest"):
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(reporting, name, None)))

    def test_single_fold_mode_never_requests_absent_bundle(self):
        calls = []
        def writer(estimates, **kwargs):
            calls.append(kwargs["fold_modes"])
            self.assertEqual(estimates["clustered"], "available")
            return "written"
        spec = SimpleNamespace(effect_writer=writer)
        result = reporting.save_effect_outputs(
            spec,
            {"clustered": "available"},
            quick_sample=False,
            fold_mode="clustered",
        )
        self.assertEqual(result, "written")
        self.assertEqual(calls, [("clustered",)])


if __name__ == "__main__":
    unittest.main()
