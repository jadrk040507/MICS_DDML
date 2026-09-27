import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from doubleml import DoubleMLBLP

import _analysis_runner as run_analysis


class RunAnalysisModesTest(unittest.TestCase):
    def test_default_runs_only_clustered_stages_in_order(self):
        calls = []
        modules = {
            name: types.SimpleNamespace(
                SEED=42,
                main=lambda name=name, **kw: calls.append((name, kw)),
            )
            for name in ("_ate_impl", "_att_impl")
        }
        with patch.dict(sys.modules, modules):
            run_analysis.run()
        self.assertEqual(calls, [
            ("_ate_impl", {"fold_mode": "clustered", "stage": "effects"}),
            ("_att_impl", {"fold_mode": "clustered", "stage": "effects"}),
            ("_ate_impl", {"fold_mode": "clustered", "stage": "sensitivity"}),
            ("_att_impl", {"fold_mode": "clustered", "stage": "sensitivity"}),
            ("_ate_impl", {"fold_mode": "clustered", "stage": "gate"}),
            ("_att_impl", {"fold_mode": "clustered", "stage": "gate"}),
        ])

    def test_unclustered_does_not_call_clustered(self):
        calls = []
        modules = {
            name: types.SimpleNamespace(
                SEED=42,
                main=lambda name=name, **kw: calls.append((name, kw)),
            )
            for name in ("_ate_impl", "_att_impl")
        }
        with patch.dict(sys.modules, modules):
            run_analysis.run("unclustered")
        self.assertEqual(len(calls), 6)
        self.assertTrue(all(kw["fold_mode"] == "unclustered" for _, kw in calls))
