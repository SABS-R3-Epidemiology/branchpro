#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import unittest

import numpy as np

import branchpro as bp


class TestSimulationControllerClass(unittest.TestCase):
    """
    Test the 'SimulationController' class.
    """

    def test_init(self):
        with self.assertRaises(TypeError):
            bp.SimulationController('1', 2, 5)

    def test_switch_resolution(self):
        br_pro_model = bp.BranchProModel(2, np.array([1, 2, 3, 2, 1]))
        simulationController = bp.SimulationController(br_pro_model, 2, 5)
        simulationController.switch_resolution(3)
        one_run_of_simulator = simulationController.run(1)
        self.assertEqual(one_run_of_simulator.shape, (3,))

    def test_run(self):
        br_pro_model = bp.BranchProModel(2, np.array([1, 2, 3, 2, 1]))
        simulationController = bp.SimulationController(br_pro_model, 2, 7)
        one_run_of_simulator = simulationController.run(1)
        self.assertEqual(one_run_of_simulator.shape, (6,))