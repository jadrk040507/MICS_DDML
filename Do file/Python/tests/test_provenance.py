"""Checkpoint provenance must change when research inputs change."""

from pathlib import Path
import tempfile
import unittest

from provenance import build_checkpoint_provenance


class CheckpointProvenanceTests(unittest.TestCase):
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
