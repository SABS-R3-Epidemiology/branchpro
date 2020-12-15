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


class DatasetLibrary(object):
    """DatasetLibrary Class:
    A data library class which contains a number of real world epidemiology
    datasets.

    Each method returns a unique dataset in form of :class:`pandas.DataFrame`.
    """
    def __init__(self):
        self._directory = os.path.join(
            os.path.dirname(__file__), 'data_library')

    def french_flu(self):
        """
        Returns a dataset on the weekly incidence numbers of the annual flue
        outbreak in France between the years 1984 and 2020.

        The data is returned as a :class:`pandas.DataFrame` with columns
        `time_index`: index of the time point represented in the data file;
        `year`: year of that particular incidence value;
        `week`: week of that particular incidence value;
        `day`: day of that particular incidence value;
        `inc`: number of flu cases at that time point;
        `inc_low`: lower bound of the 95% confidence interval of the number
            of flu cases at that time point;
        `inc_up`: upper bound of the 95% confidence interval of the number
            of flu cases at that time point;
        `inc100`: number of flu cases at that time point reported to
            100,000 people;
        `inc100_low`: lower bound of the 95% confidence interval of the number
            of flu cases at that time point reported to 100,000 people;
        `inc100_up`: upper bound of the 95% confidence interval of the number
            of flu cases at that time point reported to 100,000 people.
        """
        filepath = os.path.join(self._directory, 'french_flu_data.csv')
        return pd.read_csv(filepath)
