import unittest
import branchpro as bp


class TestSimulationControllerClass(unittest.TestCase):
    """
    Test the 'SimulationController' class.
    """
    def test_output(self):
        simulationController = bp.SimulationController()
        self.assertEqual(simulationController.value, 2)
