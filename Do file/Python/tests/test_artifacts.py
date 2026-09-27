"""Current-only artifact persistence tests."""

from pathlib import Path
import tempfile
import unittest

import joblib

try:
    import artifacts
except ModuleNotFoundError as error:
    if error.name != "artifacts":
        raise
    artifacts = None


class ArtifactTests(unittest.TestCase):
    def test_artifacts_exports_current_checkpoint_api(self):
        required = (
            "build_checkpoint_provenance",
            "build_sensitivity_provenance",
            "atomic_dump",
            "valid_sensitivity_rows",
            "OutcomeCheckpointBundle",
            "CheckpointStore",
        )
        for name in required:
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(artifacts, name, None)))

    def make_store(self, directory):
        return artifacts.CheckpointStore(
            directory,
            estimand="ATE",
            quick_sample=False,
            model_fingerprint="m" * 64,
            sensitivity_fingerprint="s" * 64,
        )

    def test_missing_current_checkpoint_does_not_search_legacy_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy" / "model.pkl"
            legacy.parent.mkdir()
            joblib.dump({"legacy": True}, legacy)
            store = self.make_store(root / "current")
            expected = root / "current" / f"model_{'m' * 12}.pkl"
            self.assertEqual(store.path("model"), expected)
            self.assertFalse(store.exists("model"))

    def test_corrupt_current_checkpoint_raises_load_error(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(Path(directory))
            path = store.path("model")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not a joblib payload")
            with self.assertRaises(Exception):
                store.load("model")


if __name__ == "__main__":
    unittest.main()
