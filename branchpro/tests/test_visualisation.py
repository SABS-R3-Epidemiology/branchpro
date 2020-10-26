import unittest
import branchpro as bp


class TestVisualisationClass(unittest.TestCase):
    """
    Test the 'Visualisation' class.
    """
    def test_output(self):
        visualisation = bp.Visualisation()
        self.assertEqual(visualisation.value, 3)
