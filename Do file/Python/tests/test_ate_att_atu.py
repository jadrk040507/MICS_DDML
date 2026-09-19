"""Checks for the estimated-share delta method and joint contrasts."""
import unittest
import numpy as np
from compare_ate_att_atu import joint_effects, linear_combination


class JointEffectsTests(unittest.TestCase):
    def test_delta_method_and_cluster_covariance(self):
        d = np.array([0, 1, 0, 1, 0, 1.])
        ia = np.array([1, -1, 2, -2, 3, -3.])
        it = np.array([2, -2, 1, -1, 4, -4.])
        groups = np.array([0, 0, 0, 1, 1, 2])
        theta, cov = joint_effects(2., 3., ia, it, d, groups)
        np.testing.assert_allclose(theta, [2, 3, 1, .5])
        # Numerically differentiate ATU with respect to all estimated inputs.
        def f(v):
            a, t, p = v
            return (a-p*t)/(1-p)
        base = np.array([2., 3., .5])
        eps = 1e-6
        grad = np.array([(f(base+eps*e)-f(base-eps*e))/(2*eps)
                         for e in np.eye(3)])
        iu = np.column_stack([ia, it, d-.5]) @ grad
        expected = sum(iu[groups == g].sum()**2 for g in np.unique(groups))/len(d)**2
        self.assertAlmostEqual(cov[2,2], expected)
        # Identity's full derivative includes the estimated share p.
        identity_gradient = [1, -.5, -.5, -(3-1)]
        self.assertLess(abs(np.asarray(identity_gradient) @ cov @ identity_gradient), 1e-12)
        coef, se = linear_combination(theta, cov, [0,1,-1,0])
        self.assertEqual(coef, 2)
        self.assertAlmostEqual(se**2, cov[1,1]+cov[2,2]-2*cov[1,2])

    def test_population_and_alignment_validation(self):
        for d in ([0,0,0], [1,1,1], [0,2,1]):
            with self.assertRaises(ValueError):
                joint_effects(1,2,np.zeros(3),np.zeros(3),d,[0,1,2])
        with self.assertRaises(ValueError):
            joint_effects(1,2,np.zeros(2),np.zeros(3),[0,1,0],[0,1,2])


if __name__ == '__main__':
    unittest.main()
