#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import unittest
from unittest.mock import patch

import pandas as pd

import branchpro as bp


class TestIncidenceNumberPlotClass(unittest.TestCase):
    """
    Test the 'IncidenceNumberPlot' class.
    """
    def test__init__(self):
        bp.IncidenceNumberPlot()

    def test_add_data(self):
        df = pd.DataFrame({
            "Time": [1, 2, 3, 5, 6],
            "Incidence Number": [10, 3, 4, 6, 9]
        })
        my_plot = bp.IncidenceNumberPlot()
        my_plot.add_data(df)

    def test_add_simulation(self):
        df = pd.DataFrame({
            "Time": [1, 2, 3, 5, 6],
            "Incidence Number": [10, 3, 4, 6, 9]
        })
        my_plot = bp.IncidenceNumberPlot()
        my_plot.add_data(df)

        dfs = pd.DataFrame({
            "Time": [1, 2, 4, 5, 6],
            "Incidence Number": [2, 3, 8, 10, 5]
        })

        my_plot.add_simulation(dfs)

    @patch("%s.bp.plt" % __name__)
    def test_show_figure(self, mock_plt):
        df = pd.DataFrame({
            "Time": [1, 2, 3, 5, 6],
            "Incidence Number": [10, 3, 4, 6, 9]
        })
        my_plot = bp.IncidenceNumberPlot()
        my_plot.add_data(df)
        my_plot.show_figure()

        # Assert show_figure is called once
        mock_plt.figure.assert_called_once()
