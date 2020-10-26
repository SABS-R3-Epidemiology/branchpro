import unittest
import branchpro as bp


class TestVisualisationClass(unittest.TestCase):
    """
    Test the 'Visualisation' class.
    """
    def test_output(self):
        self.assertEqual(bp.Visualisation, 3)
