import unittest
import branchpro as bp


class TestForwardModelClass(unittest.TestCase):
    """
    Test the 'ForwardModel' class.
    """
    def test__init__(self):
        bp.ForwardModel()

    def test_simulate(self):
        forwardModel = bp.ForwardModel()
        with self.assertRaises(NotImplementedError):
            forwardModel.simulate()


class TestBranchProModelClass(unittest.TestCase):
    """
    Test the 'BranchProModel' class.
    """
    def test_output(self):
        branchProModel = bp.BranchProModel()
        self.assertEqual(branchProModel.value, 5)
