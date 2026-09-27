"""The Python workflow must be complete and minimal in a fresh checkout."""

import importlib
from pathlib import Path
import unittest


PYTHON_DIR = Path(__file__).resolve().parents[1]
ACTIVE_MODULES = {
    "run_analysis.py",
    "analysis.py",
    "ddml.py",
    "reporting.py",
    "artifacts.py",
}
REMOVED_NAMES = (
    "_ate_impl",
    "_att_impl",
    "_analysis_runner",
    "_model_checkpoint_compat",
    "_compare_ate_att_atu",
    "_joint_inference",
)


class RepositoryLayoutTests(unittest.TestCase):
    def test_python_root_contains_only_five_active_modules(self):
        self.assertEqual(
            {path.name for path in PYTHON_DIR.glob("*.py")},
            ACTIVE_MODULES,
        )

    def test_no_active_import_mentions_removed_modules(self):
        for filename in ACTIVE_MODULES:
            source = (PYTHON_DIR / filename).read_text(encoding="utf-8")
            for removed in REMOVED_NAMES:
                with self.subTest(filename=filename, removed=removed):
                    self.assertNotIn(removed, source)

    def test_clean_import_of_all_active_modules(self):
        for filename in sorted(ACTIVE_MODULES):
            with self.subTest(filename=filename):
                importlib.import_module(filename.removesuffix(".py"))


if __name__ == "__main__":
    unittest.main()
