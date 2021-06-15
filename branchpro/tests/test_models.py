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


class TestForwardModelClass(unittest.TestCase):
    """
    Test the 'ForwardModel' class.
    """
    def test__init__(self):
        bp.ForwardModel()

    def test_simulate(self):
        forward_model = bp.ForwardModel()
        with self.assertRaises(NotImplementedError):
            forward_model.simulate(0, 1)


class TestBrachProModelClass(unittest.TestCase):
    """
    Test the 'BranchProModel' class.
    """
    def test__init__(self):
        with self.assertRaises(ValueError):
            bp.BranchProModel(0, [0])

        with self.assertRaises(TypeError):
            bp.BranchProModel('0', [1])

        with self.assertRaises(ValueError):
            bp.BranchProModel(0, 1)

    def test_get_serial_intervals(self):
        br_model = bp.BranchProModel(0, [1, 2])
        npt.assert_array_equal(
            br_model.get_serial_intervals(), np.array([1, 2]))

    def test_get_r_profile(self):
        br_model1 = bp.BranchProModel(0, [1, 2])
        br_model1.set_r_profile([1], [2])
        npt.assert_array_equal(br_model1.get_r_profile(), np.array([0, 1]))

    def test_set_r_profile(self):
        br_model1 = bp.BranchProModel(0, [1, 2])
        br_model1.set_r_profile([1], [2])
        npt.assert_array_equal(br_model1.get_r_profile(), np.array([0, 1]))

        br_model2 = bp.BranchProModel(0, [1, 2])
        br_model2.set_r_profile([3, 1], [1, 2], 3)
        npt.assert_array_equal(br_model2.get_r_profile(), np.array([3, 1, 1]))

        with self.assertRaises(ValueError):
            br_model1.set_r_profile(1, [1])

        with self.assertRaises(ValueError):
            br_model1.set_r_profile([1], 1)

        with self.assertRaises(ValueError):
            br_model1.set_r_profile([0.5, 1], [1])

        with self.assertRaises(ValueError):
            br_model1.set_r_profile([1], [-1])

        with self.assertRaises(ValueError):
            br_model1.set_r_profile([1, 2], [2, 1])

    def test_set_serial_intervals(self):
        br_model = bp.BranchProModel(0, [1, 2])
        br_model.set_serial_intervals([1, 3, 2])
        npt.assert_array_equal(
                                br_model.get_serial_intervals(),
                                np.array([1, 3, 2])
                                )

        with self.assertRaises(ValueError):
            br_model.set_serial_intervals((1))

    def test_simulate(self):
        branch_model_1 = bp.BranchProModel(2, np.array([1, 2, 3, 2, 1]))
        simulated_sample_model_1 = branch_model_1.simulate(1, np.array([2, 4]))
        new_simulated_sample_model_1 = branch_model_1.simulate(1, [0, 2, 4])
        self.assertEqual(simulated_sample_model_1.shape, (2,))
        self.assertEqual(new_simulated_sample_model_1.shape, (3,))

        branch_model_2 = bp.BranchProModel(2, [1, 2, 3, 2, 1])
        simulated_sample_model_2 = branch_model_2.simulate(1, [2, 4, 7])
        self.assertEqual(simulated_sample_model_2.shape, (3,))

        br_model3 = bp.BranchProModel(0, [1, 2])
        br_model3.set_r_profile([3, 1], [1, 2], 3)
        simulated_sample_model_3 = br_model3.simulate(1, [2, 4, 7])
        self.assertEqual(simulated_sample_model_3.shape, (3,))


class TestLocImpBranchProModelClass(unittest.TestCase):
    """
    Test the 'LocImpBranchProModel' class.
    """
    def test__init__(self):
        with self.assertRaises(TypeError):
            bp.LocImpBranchProModel(0, [1], '0')

        with self.assertRaises(ValueError):
            bp.LocImpBranchProModel(0, [1], -13)

    def test_set_epsilon(self):
        libr_model1 = bp.LocImpBranchProModel(0, [1, 2], 0)
        libr_model1.set_epsilon(1.5)
        self.assertEqual(libr_model1.epsilon, 1.5)

    def test_set_imported_cases(self):
        libr_model = bp.LocImpBranchProModel(0, [1, 2], 0)

        with self.assertRaises(ValueError):
            libr_model.set_imported_cases([1, 2], np.array([[5], [10]]))

        with self.assertRaises(ValueError):
            libr_model.set_imported_cases(np.array([[1], [2]]), [5, 10])

        with self.assertRaises(ValueError):
            libr_model.set_imported_cases([1, 2, 4], [5, 10])

    def test_simulate(self):
        libr_model_1 = bp.LocImpBranchProModel(2, np.array([1, 2, 3, 2, 1]), 0)
        libr_model_1.set_imported_cases([1, 2.0, 4, 8], [5, 10, 9, 2])
        simulated_sample_model_1 = libr_model_1.simulate(1, np.array([2, 4]))
        new_simulated_sample_model_1 = libr_model_1.simulate(1, [0, 2, 4])
        self.assertEqual(simulated_sample_model_1.shape, (2,))
        self.assertEqual(new_simulated_sample_model_1.shape, (3,))

        libr_model_2 = bp.LocImpBranchProModel(2, [1, 2, 3, 2, 1], 0)
        libr_model_2.set_imported_cases([1, 2, 4, 8], [5, 10, 9, 2])
        simulated_sample_model_2 = libr_model_2.simulate(1, [2, 4, 7])
        self.assertEqual(simulated_sample_model_2.shape, (3,))

        libr_model_3 = bp.LocImpBranchProModel(0, [1, 2], 0)
        libr_model_3.set_r_profile([3, 1], [1, 2], 3)
        libr_model_3.set_imported_cases([1, 2, 4, 8], [5, 10, 9, 2])
        simulated_sample_model_3 = libr_model_3.simulate(1, [2, 4, 7])
        self.assertEqual(simulated_sample_model_3.shape, (3,))
