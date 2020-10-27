import unittest
from branchpro.dummy import Dummy


class TestDummyClass(unittest.TestCase):
    """
    Test the 'Dummy' class.
    """
    def test_output(self):
        dummy = Dummy()
        self.assertEqual(dummy.value, 1)
