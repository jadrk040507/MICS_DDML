"""Check risk GATEs against known effects and preserve estimation row alignment."""
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

import _ate_impl as ate
import _att_impl as att


class GateGroupTests(unittest.TestCase):
    def test_groupings_preserve_rows_and_recover_known_effects(self):
        n = 120
        risk = np.tile([0, 1, 2], n // 3)
        signal = risk + 1.0 + np.random.default_rng(42).normal(0, 0.1, n)
        data = pd.DataFrame({column: np.ones(n) for column in ate.COMMON_CONTROLS})
        data['wq27_decile'] = risk + 1
        data['RiskSource'] = risk.astype(float)
        data['country_cat'] = 1
        data['Cluster_var'] = np.repeat(np.arange(n // 4), 4)
        data['water_treatment'] = np.tile([0, 1], n // 2)
        data['WQ15_g'] = np.tile([0, 1, 2, 3], n // 4)
        data['SomeRiskHome'] = signal
        # Nonconsecutive source indices and a dropped outcome exercise score alignment.
        data.index = np.arange(n) * 2
        data.loc[0, 'SomeRiskHome'] = np.nan
        data.loc[2, 'RiskSource'] = np.nan
        kept_signal = signal[1:]
        scores = kept_signal[:, None, None]
        framework = SimpleNamespace(all_thetas=np.zeros((1, 1)), scaled_psi=-scores)
        contrast = SimpleNamespace(all_thetas=np.zeros((3, 1)), scaled_psi=-np.repeat(scores, 3, axis=1))
        for module in (ate, att):
            sample = module.complete_case_sample(data, 'SomeRiskHome', 'water_treatment', extra_columns=('RiskSource',))
            self.assertEqual(len(sample), n - 1)
            self.assertTrue(pd.isna(sample.RiskSource.iloc[0]))
            irm = SimpleNamespace(framework=framework, psi_elements={'psi_a': -np.ones_like(scores), 'psi_b': scores})
            models = [SimpleNamespace(treatment_level=level, psi_elements={
                'psi_a': -np.ones_like(scores), 'psi_b': scores if level else np.zeros_like(scores),
            }) for level in (0, 1, 2, 3)]
            apos = SimpleNamespace(causal_contrast=lambda **kwargs: contrast, modellist=models)
            for clustered in (False, True):
                with self.subTest(module=module.__name__, clustered=clustered):
                    bundle = {'irm_cluster': irm, 'irm_no_cluster': irm, 'apos_cluster': apos, 'apos_no_cluster': apos}
                    rows = module.gate_for_specification(bundle, data, 'HH', 'SomeRiskHome', False, 'clustered_folds' if clustered else 'unclustered', clustered)
                    result = pd.concat(rows, ignore_index=True)
                    self.assertEqual(set(result.group), {'source_ecoli', 'source_risk'})
                    risk_rows = result.loc[result.group.eq('source_risk')]
                    self.assertEqual(len(risk_rows), 12)
                    expected = {
                        str(group): data.loc[data.SomeRiskHome.notna() & data.RiskSource.eq(group), 'SomeRiskHome'].mean()
                        for group in (0, 1, 2)
                    }
                    np.testing.assert_allclose(risk_rows.coef, risk_rows.group_value.map(expected))
                    self.assertTrue(np.isfinite(risk_rows[['se', 'pval']]).all().all())
                    self.assertEqual(set(risk_rows.group_label), set(module.GATE_GROUPS['source_risk'][2].values()))
                    self.assertTrue(result.sample_n.eq(n - 1).all())
                    result['source_ecoli_range'] = '0--100'
                    with tempfile.TemporaryDirectory() as directory:
                        paths = module.create_heterogeneity_comparison_tables(result, directory, 'test_gate', (result.specification.iloc[0],), False)
                        table = Path(paths[0]).read_text()
                        for label in ('Decile 1', 'No Risk Source', 'Some Risk Source', 'Very High Risk Source'):
                            self.assertIn(label, table)


if __name__ == '__main__':
    unittest.main()

