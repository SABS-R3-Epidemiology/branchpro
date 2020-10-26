import unittest
import branchpro as bp


class TestForwardModelClass(unittest.TestCase):
    """
    Test the 'ForwardModel' class.
    """
    def test_output(self):
        self.assertEqual(bp.ForwardModel, 1)


class TestBranchProModelClass(unittest.TestCase):
    """
    Test the 'BranchProModel' class.
    """
    def test_output(self):
        self.assertEqual(bp.BranchProModel, 0)
