import unittest
import branchpro as bp


class TestForwardModelClass(unittest.TestCase):
    """
    Test the 'ForwardModel' class.
    """
    def test_output(self):
        forwardModel = bp.ForwardModel()
        self.assertEqual(forwardModel.value, 1)


class TestBranchProModelClass(unittest.TestCase):
    """
    Test the 'BranchProModel' class.
    """
    def test_output(self):
        branchProModel = bp.BranchProModel()
        self.assertEqual(branchProModel.value, 0)
