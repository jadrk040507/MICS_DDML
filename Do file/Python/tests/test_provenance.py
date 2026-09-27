"""Checkpoint provenance must change when research inputs change."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analysis as ate
import analysis as att
import artifacts
from artifacts import build_checkpoint_provenance


class CheckpointProvenanceTests(unittest.TestCase):
    def test_model_provenance_tracks_shared_workflow_helpers(self):
        expected = {"analysis_script", "ddml", "artifacts"}
        for module in (ate, att):
            with self.subTest(module=module.__name__):
                _, details = module.checkpoint_provenance()
                self.assertTrue(expected.issubset(details["files"]))
                for name in expected:
                    self.assertEqual(details["files"][name]["status"], "present")

    def test_sensitivity_helper_change_invalidates_only_sensitivity_fingerprint(self):
        builder = getattr(artifacts, "build_sensitivity_provenance", None)
        self.assertTrue(callable(builder), "sensitivity provenance builder is missing")
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "sensitivity_scale.py"
            helper.write_text("formula = 'old'", encoding="utf-8")
            model_fingerprint = "a" * 64
            first, first_details = builder(2, model_fingerprint, {"scale": helper})
            helper.write_text("formula = 'new'", encoding="utf-8")
            second, second_details = builder(2, model_fingerprint, {"scale": helper})

        self.assertEqual(
            first_details["settings"]["model_checkpoint_fingerprint"],
            model_fingerprint,
        )
        self.assertEqual(model_fingerprint, "a" * 64)
        self.assertNotEqual(first, second)
        self.assertNotEqual(
            first_details["files"]["scale"]["sha256"],
            second_details["files"]["scale"]["sha256"],
        )

    def test_sensitivity_checkpoints_use_their_own_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            store = artifacts.CheckpointStore(
                Path(directory),
                estimand="ATE",
                quick_sample=False,
                model_fingerprint="m" * 64,
                sensitivity_fingerprint="s" * 64,
            )
            sensitivity_path = store.path("sensitivity_groups_v1_example")
            model_path = store.path("HH_example_IRM_clustered")

        self.assertIn(f"_{'s' * 12}", sensitivity_path.name)
        self.assertIn(f"_{'m' * 12}", model_path.name)

    def test_worker_configuration_invalidates_model_checkpoint(self):
        for module in (ate, att):
            with self.subTest(module=module.__name__):
                with patch.object(module, "LEARNER_JOBS", 2):
                    two_workers, _ = module.checkpoint_provenance()
                with patch.object(module, "LEARNER_JOBS", 3):
                    three_workers, _ = module.checkpoint_provenance()
                self.assertNotEqual(two_workers, three_workers)

    def test_fingerprint_is_stable_for_identical_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data.dta"
            data.write_bytes(b"same data")
            arguments = (2, {"analysis_data": data}, {"folds": 5})
            first, first_details = build_checkpoint_provenance(*arguments)
            second, second_details = build_checkpoint_provenance(*arguments)
            self.assertEqual(first, second)
            self.assertEqual(first_details, second_details)

    def test_data_change_invalidates_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data.dta"
            data.write_bytes(b"old data")
            old, _ = build_checkpoint_provenance(
                2,
                {"analysis_data": data},
                {"folds": 5},
            )
            data.write_bytes(b"new data with a different length")
            new, _ = build_checkpoint_provenance(
                2,
                {"analysis_data": data},
                {"folds": 5},
            )
            self.assertNotEqual(old, new)

    def test_setting_change_invalidates_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data.dta"
            data.write_bytes(b"same data")
            five_folds, _ = build_checkpoint_provenance(
                2,
                {"analysis_data": data},
                {"folds": 5},
            )
            ten_folds, _ = build_checkpoint_provenance(
                2,
                {"analysis_data": data},
                {"folds": 10},
            )
            self.assertNotEqual(five_folds, ten_folds)


if __name__ == "__main__":
    unittest.main()
