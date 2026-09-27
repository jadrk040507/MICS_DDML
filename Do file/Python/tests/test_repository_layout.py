"""The documented Python workflow must be complete in a fresh checkout."""

from pathlib import Path
import subprocess
import unittest


PYTHON_DIR = Path(__file__).resolve().parents[1]
PROJECT = PYTHON_DIR.parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_documented_workflow_files_are_tracked(self):
        required = [
            "01_run_analysis.py",
            "02_ate_c.py",
            "03_att_c.py",
            "04_sensitivity_c.py",
            "05_gate_c.py",
            "06_ate_u.py",
            "07_att_u.py",
            "08_sensitivity_u.py",
            "09_gate_u.py",
            "_analysis_runner.py",
            "_compare_ate_att_atu.py",
            "_joint_inference.py",
            "_model_checkpoint_compat.py",
            "_provenance.py",
            "_sensitivity_groups.py",
            "_sensitivity_scale.py",
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
