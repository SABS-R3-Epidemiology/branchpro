#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import unittest
import branchpro as bp


class TestSimulationControllerClass(unittest.TestCase):
    """
    Test the 'SimulationController' class.
    """
    def test_output(self):
        simulationController = bp.SimulationController()
        self.assertEqual(simulationController.value, 2)
