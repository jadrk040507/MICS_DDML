import math
import unittest

from _sensitivity_scale import diagonal_equivalent


class DiagonalEquivalentTest(unittest.TestCase):
    def test_equal_pair_with_adversarial_rho_stays_equal(self):
        self.assertAlmostEqual(diagonal_equivalent(.12, .12), .12)

    def test_unequal_pair_matches_the_doubleml_point_bias(self):
        r = diagonal_equivalent(.04, .25, -.6)
        self.assertAlmostEqual(r * r / (1 - r), .6**2 * .04 * .25 / .75)

    def test_empirical_rho_never_exceeds_adversarial_strength(self):
        self.assertLess(diagonal_equivalent(.04, .25, .6), diagonal_equivalent(.04, .25, 1))

    def test_cf_d_boundary_has_its_defined_limit(self):
        self.assertEqual(diagonal_equivalent(.1, 1, 1), 1.0)
        self.assertEqual(diagonal_equivalent(0, 1, 1), 0.0)
        self.assertEqual(diagonal_equivalent(.1, 1, 0), 0.0)

    def test_invalid_benchmark_rejected(self):
        for values in [(-.1, .2, 1), (.1, .2, 1.1), (math.nan, .2, 1)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                diagonal_equivalent(*values)
