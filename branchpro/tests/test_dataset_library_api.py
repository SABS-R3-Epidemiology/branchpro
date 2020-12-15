#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import unittest

import branchpro as bp


class TestDatasetLibraryAPIClass(unittest.TestCase):
    """
    Test the 'DatasetLibraryAPI' class.
    """
    def test__init__(self):
        bp.DatasetLibraryAPI()

    def test_french_flu(self):
        dataframe = bp.DatasetLibraryAPI().french_flu()
        column_names = dataframe.head()
        some_names = [
            'time_index', 'year', 'week', 'day', 'inc', 'inc_low',
            'inc_up', 'inc100', 'inc100_low', 'inc100_up']
        self.assertTrue(set(column_names) == set(some_names))
