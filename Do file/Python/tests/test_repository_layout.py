"""The documented Python workflow must be complete in a fresh checkout."""

from pathlib import Path
import subprocess
import unittest


PYTHON_DIR = Path(__file__).resolve().parents[1]
PROJECT = PYTHON_DIR.parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_documented_workflow_files_are_tracked(self):
        required = [
            "run_analysis.py",
            "analysis.py",
            "ddml.py",
            "reporting.py",
            "artifacts.py",
        ]
        relative = [str(path.relative_to(PROJECT)) for path in map(PYTHON_DIR.joinpath, required)]
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", *relative],
            cwd=PROJECT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
