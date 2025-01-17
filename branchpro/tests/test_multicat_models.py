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


class TestMultiCatPoissonBranchProModelClass(unittest.TestCase):
    """
    Test the 'MultiCatPoissonBranchProModel' class.
    """
    def test__init__(self):
        with self.assertRaises(TypeError):
            bp.MultiCatPoissonBranchProModel(
                '3', [0, 0], 2, [[1., 1.], [1., 1.]], [1, 1])

        with self.assertRaises(TypeError):
            bp.MultiCatPoissonBranchProModel(
                3, [0, 0], '2', [[1., 1.], [1., 1.]], [1, 1])

        with self.assertRaises(TypeError):
            bp.MultiCatPoissonBranchProModel(
                3, [0, 0], .2, [[1., 1.], [1., 1.]], [1, 1])

        with self.assertRaises(ValueError):
            bp.MultiCatPoissonBranchProModel(
                3, [0, 0], -2, [[1., 1.], [1., 1.]], [1, 1])

        with self.assertRaises(ValueError):
            bp.MultiCatPoissonBranchProModel(
                3, [0, 0.2], 2, [1., 1.], [1, 1])

        with self.assertRaises(ValueError):
            bp.MultiCatPoissonBranchProModel(
                3, [0, 0.2], 2, [[1., 1., 1.], [1., 1., 1.]], [1, 1])

        with self.assertRaises(ValueError):
            bp.MultiCatPoissonBranchProModel(
                3, [0, 0.2], 2, [[1., 1.], [1., 1.], [1., 1.]], [1, 1])

        with self.assertRaises(ValueError):
            bp.MultiCatPoissonBranchProModel(
                3, [0, 0.2], 2, [[1., 1.], [1., -1.]], [1, 1])

        with self.assertRaises(TypeError):
            bp.MultiCatPoissonBranchProModel(
                3, [0, 0.2], 2, [[1., 1.], [1., '1']], [1, 1])

        with self.assertRaises(ValueError):
            bp.MultiCatPoissonBranchProModel(
                3, [0, 0.2], 2, [[1., 1.], [1., 1.]], '1')

        with self.assertRaises(ValueError):
            bp.MultiCatPoissonBranchProModel(
                3, [0, 0.2], 2, [[1., 1.], [1., 1.]], [1])

        with self.assertRaises(ValueError):
            bp.MultiCatPoissonBranchProModel(
                3, [0, 0.2], 2, [[1., 1.], [1., 1.]], [1, -1])

        with self.assertRaises(TypeError):
            bp.MultiCatPoissonBranchProModel(
                3, [0, 0.2], 2, [[1., 1.], [1., 1.]], [1, '1'])

        with self.assertRaises(ValueError):
            bp.MultiCatPoissonBranchProModel(
                3, [[0, 0.2]], 2, [[1., 1.], [1., 1.]], [1, 1])

        with self.assertRaises(ValueError):
            bp.MultiCatPoissonBranchProModel(
                3, [0, -0.2], 2, [[1., 1.], [1., 1.]], [1, 1])

        with self.assertRaises(ValueError):
            bp.MultiCatPoissonBranchProModel(
                3, [0, 0.2], 2, [[1., 1.], [1., 1.]], [1, 1], True)

        with self.assertRaises(ValueError):
            bp.MultiCatPoissonBranchProModel(
                3, [[0, 0.2], [0, 0.5], [1, 0]], 2,
                [[1., 1.], [1., 1.]], [1, 1], True)

        with self.assertRaises(ValueError):
            bp.MultiCatPoissonBranchProModel(
                3, [[0, 0.2], [0.2, -0.5]], 2,
                [[1., 1.], [1., 1.]], [1, 1], True)

    def test_get_transmissibility(self):
        multicat_model = bp.MultiCatPoissonBranchProModel(
            3, [1, 2], 2, [[1., 1.], [1., 1.]], [1, 1])
        npt.assert_array_equal(
            multicat_model.get_transmissibility(), np.array([1, 1]))

    def test_get_contact_matrix(self):
        multicat_model = bp.MultiCatPoissonBranchProModel(
            3, [1, 2], 2, [[1., 1.], [1., 1.]], [1, 1])
        npt.assert_array_equal(
            multicat_model.get_contact_matrix(),
            np.array([[1., 1.], [1., 1.]]))

    def test_set_transmissibility(self):
        multicat_model = bp.MultiCatPoissonBranchProModel(
            3, [1, 2], 2, [[1., 1.], [1., 1.]], [1, 1])
        multicat_model.set_transmissibility([1, 0.8])
        npt.assert_array_equal(
            multicat_model.get_transmissibility(), [1, 0.8]
        )

        with self.assertRaises(ValueError):
            multicat_model.set_transmissibility([[1, 0.8]])

        with self.assertRaises(ValueError):
            multicat_model.set_transmissibility([1, 0.8, 0.9])

        with self.assertRaises(ValueError):
            multicat_model.set_transmissibility([1, -0.8])

        with self.assertRaises(TypeError):
            multicat_model.set_transmissibility(['1', 0.8])

    def test_set_contact_matrix(self):
        multicat_model = bp.MultiCatPoissonBranchProModel(
            3, [1, 2], 2, [[1., 1.], [1., 1.]], [1, 1])
        multicat_model.set_contact_matrix([[1., 2.], [.1, 1.]])
        npt.assert_array_equal(
            multicat_model.get_contact_matrix(),
            [[1., 2.], [.1, 1.]]
        )

        with self.assertRaises(ValueError):
            multicat_model.set_contact_matrix([1., 2.])

        with self.assertRaises(ValueError):
            multicat_model.set_contact_matrix([[1., 2.]])

        with self.assertRaises(ValueError):
            multicat_model.set_contact_matrix([[1., 2., .1], [1., 0., 2.]])

        with self.assertRaises(ValueError):
            multicat_model.set_contact_matrix([[1., 2.], [-.1, 1.]])

        with self.assertRaises(TypeError):
            multicat_model.set_contact_matrix([[1., 2.], ['.1', 1.]])

    def test_get_serial_intervals(self):
        multicat_model = bp.MultiCatPoissonBranchProModel(
            3, [1, 2], 2, [[1., 1.], [1., 1.]], [1, 1])
        npt.assert_array_equal(
            multicat_model.get_serial_intervals(),
            np.array([[1, 2], [1, 2]]))

    def test_get_r_profile(self):
        multicat_model = bp.MultiCatPoissonBranchProModel(
            3, [1, 2], 2, [[1., 1.], [1., 1.]], [1, 1])
        self.assertEqual(multicat_model.get_r_profile(), 3)

    def test_set_r_profile(self):
        multicat_model1 = bp.MultiCatPoissonBranchProModel(
            3, [1, 2], 2, [[1., 1.], [1., 1.]], [1, 1])
        multicat_model1.set_r_profile([2, 0.5], [2, 3])
        npt.assert_array_equal(
            multicat_model1.get_r_profile(), np.array([3, 2, 0.5]))

        multicat_model2 = bp.MultiCatPoissonBranchProModel(
            0, [1, 2], 2, [[1., 1.], [1., 1.]], [1, 1])
        multicat_model2.set_r_profile([3, 1], [1, 2], 3)
        npt.assert_array_equal(
            multicat_model2.get_r_profile(),
            np.array([3, 1, 1]))

        with self.assertRaises(ValueError):
            multicat_model1.set_r_profile(1, [1])

        with self.assertRaises(ValueError):
            multicat_model1.set_r_profile([1], 1)

        with self.assertRaises(ValueError):
            multicat_model1.set_r_profile([0.5, 1], [1])

        with self.assertRaises(ValueError):
            multicat_model1.set_r_profile([1], [-1])

        with self.assertRaises(ValueError):
            multicat_model1.set_r_profile([1, 2], [2, 1])

    def test_set_serial_intervals(self):
        multicat_model = bp.MultiCatPoissonBranchProModel(
            3, [1, 2], 2, [[1., 1.], [1., 1.]], [1, 1])
        multicat_model.set_serial_intervals([1, 3, 2])
        npt.assert_array_equal(
            multicat_model.get_serial_intervals(),
            np.array([[1, 3, 2], [1, 3, 2]])
            )

        multicat_model2 = bp.MultiCatPoissonBranchProModel(
            3, [[1, 2], [0, 1]], 2, [[1., 1.], [1., 1.]], [1, 1], True)
        multicat_model2.set_serial_intervals([[1, 3, 2], [2, 1, 0]], True)
        npt.assert_array_equal(
            multicat_model2.get_serial_intervals(),
            np.array([[1, 3, 2], [2, 1, 0]])
            )

        with self.assertRaises(ValueError):
            multicat_model.set_serial_intervals((1))

        with self.assertRaises(ValueError):
            multicat_model.set_serial_intervals([1, -1.2])

        with self.assertRaises(ValueError):
            multicat_model2.set_serial_intervals([2, 1, 0], True)

        with self.assertRaises(ValueError):
            multicat_model2.set_serial_intervals([[1, 3, 2]], True)

        with self.assertRaises(ValueError):
            multicat_model2.set_serial_intervals([[1, 3, 2], [-2, 1, 0]], True)

    def test_simulate(self):
        multicat_model_1 = bp.MultiCatPoissonBranchProModel(
            1, np.array([1, 2, 3, 2, 1]), 2, [[1., 1.], [1., 1.]], [1, 1])
        simulated_sample_model_1 = multicat_model_1.simulate(
            [10, 10], np.array([2, 4]))
        new_simulated_sample_model_1 = multicat_model_1.simulate(
            [10, 10], [0, 2, 4])
        self.assertEqual(simulated_sample_model_1.shape, (2, 2))
        self.assertEqual(new_simulated_sample_model_1.shape, (3, 2))

        multicat_model_2 = bp.MultiCatPoissonBranchProModel(
            1, [1, 2, 3, 2, 1], 2, [[1., 1.], [1., 1.]], [1, 1])
        simulated_sample_model_2 = multicat_model_2.simulate(
            [10, 10], [2, 4, 7], var_contacts=True)
        self.assertEqual(simulated_sample_model_2.shape, (3, 2))

        multicat_model3 = bp.MultiCatPoissonBranchProModel(
            0, [1, 2, 3, 2, 1], 2, [[1., 1.], [1., 1.]], [1, 1])
        multicat_model3.set_r_profile([0, 0], [1, 2], 3)
        simulated_sample_model_3 = multicat_model3.simulate(
            [10, 10], [2, 4, 7], neg_binom=True)
        self.assertEqual(simulated_sample_model_3.shape, (3, 2))

        multicat_model_4 = bp.MultiCatPoissonBranchProModel(
            1, np.array([1, 2, 3, 2, 1]), 2, [[1., 1.], [1., 1.]], [1, 1])
        simulated_sample_model_4 = multicat_model_4.simulate(
            [10, 10], np.array([2, 4]))
        new_simulated_sample_model_4 = multicat_model_4.simulate(
            [10, 10], [0, 2, 4], neg_binom=True, niu=0.2)
        self.assertEqual(simulated_sample_model_4.shape, (2, 2))
        self.assertEqual(new_simulated_sample_model_4.shape, (3, 2))

        multicat_model_5 = bp.MultiCatPoissonBranchProModel(
            1, np.array([1, 2, 3, 2, 1]), 2, [[1., 1.], [1., 1.]], [1, 1])
        simulated_sample_model_5 = multicat_model_5.simulate(
            [10, 10], np.array([2, 4]))
        new_simulated_sample_model_5 = multicat_model_5.simulate(
            [10, 10], [0, 2, 4], var_contacts=True, neg_binom=True, niu=0.2)
        self.assertEqual(simulated_sample_model_5.shape, (2, 2))
        self.assertEqual(new_simulated_sample_model_5.shape, (3, 2))


class TestLocImpMultiCatPoissonBranchProModelClass(unittest.TestCase):
    """
    Test the 'LocImpMultiCatPoissonBranchProModel' class.
    """
    def test__init__(self):
        with self.assertRaises(TypeError):
            bp.LocImpMultiCatPoissonBranchProModel(
                0, [1], '0', 2, [[1., 1.], [1., 1.]], [1., 1.])

        with self.assertRaises(ValueError):
            bp.LocImpMultiCatPoissonBranchProModel(
                0, [1], -13, 2, [[1., 1.], [1., 1.]], [1, 1])

        with self.assertRaises(TypeError):
            bp.LocImpMultiCatPoissonBranchProModel(
                0, [[1], [1]], '0', 2, [[1., 1.], [1., 1.]], [1., 1.], True)

        with self.assertRaises(ValueError):
            bp.LocImpMultiCatPoissonBranchProModel(
                0, [[1], [1]], -13, 2, [[1., 1.], [1., 1.]], [1, 1], True)

    def test_simulate(self):
        limulticat_model_1 = bp.LocImpMultiCatPoissonBranchProModel(
            1, np.array([1, 2, 3, 2, 1]), 0, 2, [[1., 1.], [1., 1.]], [1, 1])
        limulticat_model_1.set_imported_cases(
            [1, 2.0, 4, 8],
            [[5, 2], [10, 8], [9, 1], [2, 10]])

        simulated_sample_model_1 = limulticat_model_1.simulate(
            [1, 1], np.array([2, 4]))
        new_simulated_sample_model_1 = limulticat_model_1.simulate(
            [10, 10], [0, 2, 4])
        self.assertEqual(simulated_sample_model_1.shape, (2, 2))
        self.assertEqual(new_simulated_sample_model_1.shape, (3, 2))

        limulticat_model_2 = bp.LocImpMultiCatPoissonBranchProModel(
            1, [1, 2, 3, 2, 1], 0, 2, [[1., 1.], [1., 1.]], [1, 1])
        limulticat_model_2.set_imported_cases(
            [1, 2, 4, 8],
            [[5, 2], [10, 8], [9, 1], [2, 10]])
        simulated_sample_model_2 = limulticat_model_2.simulate(
            [10, 10], [2, 4, 7], var_contacts=True)
        self.assertEqual(simulated_sample_model_2.shape, (3, 2))

        limulticat_model_3 = bp.LocImpMultiCatPoissonBranchProModel(
            0, [1, 2], 0, 2, [[1., 1.], [1., 1.]], [1, 1])
        limulticat_model_3.set_r_profile(
            [2, 0], [1, 2], 3)
        limulticat_model_3.set_imported_cases(
            [1, 2, 4, 8],
            [[5, 2], [10, 8], [9, 1], [2, 10]])
        simulated_sample_model_3 = limulticat_model_3.simulate(
            10, [2, 4, 7], neg_binom=True)
        self.assertEqual(simulated_sample_model_3.shape, (3, 2))

        limulticat_model_4 = bp.LocImpMultiCatPoissonBranchProModel(
            0, [1, 2], 0, 2, [[1., 1.], [1., 1.]], [1, 1])
        limulticat_model_4.set_r_profile(
            [2, 0], [1, 2], 3)
        limulticat_model_4.set_imported_cases(
            [1, 2, 4, 8],
            [[5, 2], [10, 8], [9, 1], [2, 10]])
        simulated_sample_model_4 = limulticat_model_4.simulate(
            10, [2, 4, 7], neg_binom=True, niu=0.2)
        self.assertEqual(simulated_sample_model_4.shape, (3, 2))

        limulticat_model_5 = bp.LocImpMultiCatPoissonBranchProModel(
            0, [1, 2], 0, 2, [[1., 1.], [1., 1.]], [1, 1])
        limulticat_model_5.set_r_profile(
            [2, 0], [1, 2], 3)
        limulticat_model_5.set_imported_cases(
            [1, 2, 4, 8],
            [[5, 2], [10, 8], [9, 1], [2, 10]])
        simulated_sample_model_5 = limulticat_model_5.simulate(
            10, [2, 4, 7], var_contacts=True, neg_binom=True, niu=0.2)
        self.assertEqual(simulated_sample_model_5.shape, (3, 2))
