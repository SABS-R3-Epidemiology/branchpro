import unittest
import branchpro as bp


class TestSimulationControllerClass(unittest.TestCase):
    """
    Test the 'SimulationController' class.
    """
    def test_output(self):
        self.assertEqual(bp.SimulationController, 2)
