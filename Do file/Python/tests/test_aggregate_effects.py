"""Aggregate inference must follow the project rule, not average SEs."""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import norm

from _ate_impl import cluster_robust_framework_inference
from _compare_ate_att_atu import aggregate_repetitions


class AggregateEffectsTests(unittest.TestCase):
    def frame(self, coefficients, errors, estimand='ATE'):
        return pd.DataFrame([
            dict(dataset='HH', outcome='y', method='IRM', contrast='d',
                 estimand=estimand, repetition=i+1, coef=c, se=s,
                 pval=2*norm.sf(abs(c/s)), ci_lower=c-norm.ppf(.975)*s,
                 ci_upper=c+norm.ppf(.975)*s, p=.4, n=100)
            for i,(c,s) in enumerate(zip(coefficients,errors))])

    def test_matches_existing_inference_helper(self):
        coefficients = np.array([1.,5.,3.])
        errors = np.array([2.,.1,.2])
        scores = np.zeros((2,1,3))
        scores[0,0,:] = 2*errors
        reference = cluster_robust_framework_inference(
            SimpleNamespace(scaled_psi=scores, all_thetas=coefficients[None,:]), [0,1])
        result = aggregate_repetitions(self.frame(coefficients,errors)).iloc[0]
        for field in ('coef','se','pval','ci_lower','ci_upper'):
            self.assertAlmostEqual(result[field],reference[field][0])
        self.assertNotAlmostEqual(result.se,np.median(errors))
        self.assertTrue(pd.isna(result.identity_residual))

    def test_contrast_is_aggregated_after_derivation(self):
        att, atu = np.array([0.,10.,11.]), np.array([0.,1.,10.])
        results = pd.concat([self.frame(att,[1]*3,'ATT'),
                             self.frame(atu,[1]*3,'ATU_derived'),
                             self.frame(att-atu,[1]*3,'ATT_minus_ATU')])
        result = aggregate_repetitions(results).set_index('estimand')
        self.assertEqual(result.loc['ATT_minus_ATU','coef'],1)
        self.assertEqual(result.loc['ATT','coef']-result.loc['ATU_derived','coef'],9)
        with self.assertRaises(ValueError):
            aggregate_repetitions(results.iloc[1:])


if __name__ == '__main__':
    unittest.main()
