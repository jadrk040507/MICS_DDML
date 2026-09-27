"""Tests for lazy outcome checkpoint access."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

from _checkpoint_io import OutcomeCheckpointBundle


class OutcomeCheckpointBundleTests(unittest.TestCase):
    def test_bundle_returns_table_frame_without_loading_model(self):
        frame = pd.DataFrame({"y": [1, 2]})
        bundle = OutcomeCheckpointBundle({"irm": Path("missing.pkl")}, {"frame": frame})
        with patch("_checkpoint_io.joblib.load") as load:
            self.assertIs(bundle["frame"], frame)
        load.assert_not_called()

    def test_bundle_loads_model_on_demand(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            model = {"estimate": 0.5}
            joblib.dump(model, path)
            bundle = OutcomeCheckpointBundle({"irm": path}, {})
            self.assertEqual(bundle["irm"], model)

    def test_bundle_unwraps_clustered_apos_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "apos.pkl"
            model = {"fitted": True}
            joblib.dump({"model": model}, path)
            bundle = OutcomeCheckpointBundle({"apos_cluster": path}, {})
            self.assertEqual(bundle["apos_cluster"], model)


if __name__ == "__main__":
    unittest.main()
