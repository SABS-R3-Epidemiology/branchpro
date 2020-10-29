#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import unittest
import branchpro as bp
import numpy as np
import numpy.testing as npt


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

    def test_get_serial_intevals(self):
        br_model = bp.BranchProModel(0, [1, 2])
        npt.assert_array_equal(br_model.get_serial_intevals, np.array([1, 2]))

    def test_update_serial_intevals(self):
        br_model = bp.BranchProModel(0, [1, 2])
        br_model.update_serial_intevals([1, 3, 2])
        npt.assert_array_equal(
                                br_model.get_serial_intevals,
                                np.array([1, 3, 2])
                                )

        with self.assertRaises(ValueError):
            br_model.update_serial_intevals((1))

    def test_simulate(self):
        branch_model_1 = bp.BranchProModel(2, np.array([1, 2, 3, 2, 1]))
        simulated_sample_model_1 = branch_model_1.simulate(1, np.array([2, 4]))
        new_simulated_sample_model_1 = branch_model_1.simulate(1, [0, 2, 4])
        self.assertEqual(simulated_sample_model_1.shape, (2,))
        self.assertEqual(new_simulated_sample_model_1.shape, (3,))

        branch_model_2 = bp.BranchProModel(2, [1, 2, 3, 2, 1])
        simulated_sample_model_2 = branch_model_2.simulate(1, [2, 4, 7])
        self.assertEqual(simulated_sample_model_2.shape, (3,))

        with self.assertRaises(ValueError):
            branch_model_1.simulate(2, (1))
