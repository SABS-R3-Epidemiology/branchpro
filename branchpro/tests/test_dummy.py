#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import unittest
from branchpro.dummy import Dummy


class TestDummyClass(unittest.TestCase):
    """
    Test the 'Dummy' class.
    """
    def test_output(self):
        dummy = Dummy()
        self.assertEqual(dummy.value, 1)
