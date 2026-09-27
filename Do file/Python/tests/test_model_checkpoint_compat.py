import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from _model_checkpoint_compat import legacy_model_path


class LegacyModelCheckpointTest(unittest.TestCase):
    def test_matching_fitted_model_reused_but_sensitivity_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            script = project / "Do file/Python/archive/replaced_2026_09_23/ate.py"
            script.parent.mkdir(parents=True)
            script.write_bytes(b"old fitting implementation")
            files = {
                key: {"status": "present", "sha256": key}
                for key in ("data_HH", "data_U5", "environment_lock", "shared_engine")
            }
            files["analysis_script"] = {"sha256": hashlib.sha256(script.read_bytes()).hexdigest()}
            provenance = {"checkpoint_schema_version": 2, "settings": {"folds": 5}, "files": files}
            output = project / "Output/ATE"
            output.mkdir(parents=True)
            (output / "manifest.json").write_text(json.dumps({
                "checkpoint_fingerprint": "a" * 64,
                "checkpoint_provenance": provenance,
            }))
            model = output / "checkpoints/HH_SomeRiskHome_IRM_clustered_aaaaaaaaaaaa.pkl"
            model.parent.mkdir()
            model.write_bytes(b"expensive model")
            self.assertEqual(legacy_model_path(project, "ATE", "HH_SomeRiskHome_IRM_clustered", provenance, False), model)
            self.assertIsNone(legacy_model_path(project, "ATE", "sensitivity_HH_SomeRiskHome_clustered_folds_IRM", provenance, False))
            self.assertIsNone(legacy_model_path(project, "ATE", "HH_SomeRiskHome_IRM_clustered", provenance, True))
            altered = {**provenance, "settings": {"folds": 3}}
            self.assertIsNone(legacy_model_path(project, "ATE", "HH_SomeRiskHome_IRM_clustered", altered, False))
