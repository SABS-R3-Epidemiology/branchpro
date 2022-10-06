#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import unittest

import numpy as np
import numpy.testing as npt

import branchpro as bp


class TestNegBinBranchProModelClass(unittest.TestCase):
    """
    Test the 'NegBinBranchProModel' class.
    """
    def test__init__(self):
        with self.assertRaises(ValueError):
            bp.NegBinBranchProModel(0, [0], 0.05)

        with self.assertRaises(TypeError):
            bp.NegBinBranchProModel('0', [1], 0.05)

        with self.assertRaises(ValueError):
            bp.NegBinBranchProModel(0, 1, 0.05)

        with self.assertRaises(TypeError):
            bp.NegBinBranchProModel(0, [1], '0.05')

        with self.assertRaises(ValueError):
            bp.NegBinBranchProModel(0, [1], 0)

    def test_get_serial_intervals(self):
        nbbr_model = bp.NegBinBranchProModel(0, [1, 2], 0.05)
        npt.assert_array_equal(
            nbbr_model.get_serial_intervals(), np.array([1, 2]))

    def test_get_r_profile(self):
        nbbr_model1 = bp.NegBinBranchProModel(0, [1, 2], 0.05)
        nbbr_model1.set_r_profile([1], [2])
        npt.assert_array_equal(nbbr_model1.get_r_profile(), np.array([0, 1]))

    def test_set_r_profile(self):
        nbbr_model1 = bp.NegBinBranchProModel(0, [1, 2], 0.05)
        nbbr_model1.set_r_profile([1], [2])
        npt.assert_array_equal(nbbr_model1.get_r_profile(), np.array([0, 1]))

        nbbr_model2 = bp.NegBinBranchProModel(0, [1, 2], 0.05)
        nbbr_model2.set_r_profile([3, 1], [1, 2], 3)
        npt.assert_array_equal(
            nbbr_model2.get_r_profile(), np.array([3, 1, 1]))

        with self.assertRaises(ValueError):
            nbbr_model1.set_r_profile(1, [1])

        with self.assertRaises(ValueError):
            nbbr_model1.set_r_profile([1], 1)

        with self.assertRaises(ValueError):
            nbbr_model1.set_r_profile([0.5, 1], [1])

        with self.assertRaises(ValueError):
            nbbr_model1.set_r_profile([1], [-1])

        with self.assertRaises(ValueError):
            nbbr_model1.set_r_profile([1, 2], [2, 1])

    def test_set_serial_intervals(self):
        nbbr_model = bp.NegBinBranchProModel(0, [1, 2], 0.05)
        nbbr_model.set_serial_intervals([1, 3, 2])
        npt.assert_array_equal(
                                nbbr_model.get_serial_intervals(),
                                np.array([1, 3, 2])
                                )

        with self.assertRaises(ValueError):
            nbbr_model.set_serial_intervals((1))

    def test_get_overdispersion(self):
        nbbr_model = bp.NegBinBranchProModel(0, [1, 2], 0.05)
        self.assertEqual(nbbr_model.get_overdispersion(), 0.05)

    def test_set_overdispersion(self):
        nbbr_model = bp.NegBinBranchProModel(0, [1, 2], 0.05)
        nbbr_model.set_overdispersion(3)
        self.assertEqual(nbbr_model.get_overdispersion(), 3)

        with self.assertRaises(TypeError):
            nbbr_model.set_overdispersion([1])

        with self.assertRaises(ValueError):
            nbbr_model.set_overdispersion(-1)

    def test_simulate(self):
        nbbr_model_1 = bp.NegBinBranchProModel(
            2, np.array([1, 2, 3, 2, 1]), 0.05)
        simulated_sample_model_1 = nbbr_model_1.simulate(10, np.array([2, 4]))
        new_simulated_sample_model_1 = nbbr_model_1.simulate(10, [0, 2, 4])
        self.assertEqual(simulated_sample_model_1.shape, (2,))
        self.assertEqual(new_simulated_sample_model_1.shape, (3,))

        nbbr_model_2 = bp.NegBinBranchProModel(2, [1, 2, 3, 2, 1], 0.05)
        simulated_sample_model_2 = nbbr_model_2.simulate(10, [2, 4, 7])
        self.assertEqual(simulated_sample_model_2.shape, (3,))

        nbbr_model3 = bp.NegBinBranchProModel(0, [1, 2], 0.05)
        nbbr_model3.set_r_profile([0, 0], [1, 2], 3)
        simulated_sample_model_3 = nbbr_model3.simulate(10, [2, 4, 7])
        self.assertEqual(simulated_sample_model_3.shape, (3,))


class TestLocImpNegBinBranchProModelClass(unittest.TestCase):
    """
    Test the 'LocImpNegBinBranchProModel' class.
    """
    def test__init__(self):
        with self.assertRaises(TypeError):
            bp.LocImpNegBinBranchProModel(0, [1], '0', 0.05)

        with self.assertRaises(ValueError):
            bp.LocImpNegBinBranchProModel(0, [1], -13, 0.05)

        with self.assertRaises(TypeError):
            bp.LocImpNegBinBranchProModel(0, [1], 0, '0.05')

        with self.assertRaises(ValueError):
            bp.LocImpNegBinBranchProModel(0, [1], 0, 0)

    def test_get_overdispersion(self):
        linbbr_model = bp.LocImpNegBinBranchProModel(0, [1, 2], 0, 0.05)
        self.assertEqual(linbbr_model.get_overdispersion(), 0.05)

    def test_set_overdispersion(self):
        linbbr_model = bp.LocImpNegBinBranchProModel(0, [1, 2], 0, 0.05)
        linbbr_model.set_overdispersion(3)
        self.assertEqual(linbbr_model.get_overdispersion(), 3)

        with self.assertRaises(TypeError):
            linbbr_model.set_overdispersion([1])

        with self.assertRaises(ValueError):
            linbbr_model.set_overdispersion(-1)

    def test_simulate(self):
        linbbr_model_1 = bp.LocImpNegBinBranchProModel(
            2, np.array([1, 2, 3, 2, 1]), 0, 0.05)
        linbbr_model_1.set_imported_cases([1, 2.0, 4, 8], [5, 10, 9, 2])
        simulated_sample_model_1 = linbbr_model_1.simulate(1, np.array([2, 4]))
        new_simulated_sample_model_1 = linbbr_model_1.simulate(10, [0, 2, 4])
        self.assertEqual(simulated_sample_model_1.shape, (2,))
        self.assertEqual(new_simulated_sample_model_1.shape, (3,))

        linbbr_model_2 = bp.LocImpNegBinBranchProModel(
            2, [1, 2, 3, 2, 1], 0, 0.05)
        linbbr_model_2.set_imported_cases([1, 2, 4, 8], [5, 10, 9, 2])
        simulated_sample_model_2 = linbbr_model_2.simulate(10, [2, 4, 7])
        self.assertEqual(simulated_sample_model_2.shape, (3,))

        linbbr_model_3 = bp.LocImpNegBinBranchProModel(0, [1, 2], 0, 0.05)
        linbbr_model_3.set_r_profile([3, 0], [1, 2], 3)
        linbbr_model_3.set_imported_cases([1, 2, 4, 8], [5, 10, 9, 2])
        simulated_sample_model_3 = linbbr_model_3.simulate(10, [2, 4, 7])
        self.assertEqual(simulated_sample_model_3.shape, (3,))
