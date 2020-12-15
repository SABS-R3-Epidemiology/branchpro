#
# DatasetLibraryAPI Class
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import os
import pandas as pd


class DatasetLibrary:
    """DatasetLibrary Class:
    Class for reading the files in the data library. These files are in the
    .csv format.
    """
    def __init__(self):
        self._directory = os.path.join(
            os.path.dirname(__file__), 'data_library')

    def french_flu(self):
        """
        Reads the datafile on the weekly number of flue incidences registered
        in France between 1984 (week 44) and 2020 (week 42).
        """
        filepath = os.path.join(self._directory, 'french_flu_data.csv')
        return pd.read_csv(filepath)
