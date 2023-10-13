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


class TestPoiBinBranchProModelClass(unittest.TestCase):
    """
    Test the 'PoiBinBranchProModel' class.
    """
    def test__init__(self):
        with self.assertRaises(ValueError):
            bp.PoiBinBranchProModel(0, [-1])

        with self.assertRaises(ValueError):
            bp.PoiBinBranchProModel(0, [0, 0.2, 1.2])

        with self.assertRaises(TypeError):
            bp.PoiBinBranchProModel('0', [1])

        with self.assertRaises(ValueError):
            bp.PoiBinBranchProModel(0, 1)

    def test_get_next_gen(self):
        pbbr_model = bp.PoiBinBranchProModel(0, [0.1, 0.2])
        npt.assert_array_equal(
            pbbr_model.get_next_gen(), np.array([0.1, 0.2]))

    def test_get_mean_contact(self):
        pbbr_model1 = bp.PoiBinBranchProModel(0, [0.1, 0.2])
        pbbr_model1.set_mean_contact([1], [2])
        npt.assert_array_equal(
            pbbr_model1.get_mean_contact(), np.array([0, 1]))

    def test_set_mean_contact(self):
        pbbr_model1 = bp.PoiBinBranchProModel(0, [1, 0.2])
        pbbr_model1.set_mean_contact([1], [2])
        npt.assert_array_equal(
            pbbr_model1.get_mean_contact(), np.array([0, 1]))

        pbbr_model2 = bp.PoiBinBranchProModel(0, [0.1, 0.2])
        pbbr_model2.set_mean_contact([3, 1], [1, 2], 3)
        npt.assert_array_equal(
            pbbr_model2.get_mean_contact(), np.array([3, 1, 1]))

        with self.assertRaises(ValueError):
            pbbr_model1.set_mean_contact(1, [1])

        with self.assertRaises(ValueError):
            pbbr_model1.set_mean_contact([1], 1)

        with self.assertRaises(ValueError):
            pbbr_model1.set_mean_contact([0.5, 1], [1])

        with self.assertRaises(ValueError):
            pbbr_model1.set_mean_contact([1], [-1])

        with self.assertRaises(ValueError):
            pbbr_model1.set_mean_contact([1, 2], [2, 1])

    def test_set_next_gen(self):
        pbbr_model = bp.PoiBinBranchProModel(0, [0.1, 0.2])
        pbbr_model.set_next_gen([0.1, 0.3, 0.2])
        npt.assert_array_equal(
                                pbbr_model.get_next_gen(),
                                np.array([0.1, 0.3, 0.2])
                                )

        with self.assertRaises(ValueError):
            pbbr_model.set_next_gen((0.1))

    def test_simulate(self):
        branch_model_1 = bp.PoiBinBranchProModel(
            2, np.array([0.1, 0.2, 0.3, 0.2, 0.1]))
        simulated_sample_model_1 = branch_model_1.simulate(1, np.array([2, 4]))
        new_simulated_sample_model_1 = branch_model_1.simulate(1, [0, 2, 4])
        self.assertEqual(simulated_sample_model_1.shape, (2,))
        self.assertEqual(new_simulated_sample_model_1.shape, (3,))

        branch_model_2 = bp.PoiBinBranchProModel(
            2, [0.1, 0.2, 0.3, 0.2, 0.1])
        simulated_sample_model_2 = branch_model_2.simulate(1, [2, 4, 7])
        self.assertEqual(simulated_sample_model_2.shape, (3,))

        pbbr_model3 = bp.PoiBinBranchProModel(0, [0.1, 0.2])
        pbbr_model3.set_mean_contact([3, 1], [1, 2], 3)
        simulated_sample_model_3 = pbbr_model3.simulate(1, [2, 4, 7])
        self.assertEqual(simulated_sample_model_3.shape, (3,))
