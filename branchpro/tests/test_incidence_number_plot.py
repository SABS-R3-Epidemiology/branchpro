#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import unittest
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
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
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        my_plot = bp.IncidenceNumberPlot()
        my_plot.add_data(df)

        npt.assert_array_equal(
            np.array(
                [
                    my_plot.figure['data'][0]['x'],
                    my_plot.figure['data'][0]['y']
                ]
                ),
            np.array(
                [np.array([1, 2, 3, 5, 6]), np.array([10,  3,  4,  6,  9])]
                )
        )

        with self.assertRaises(TypeError):
            bp.IncidenceNumberPlot().add_data(0)

    def test_add_simulation(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        my_plot = bp.IncidenceNumberPlot()
        my_plot.add_data(df)

        dfs = pd.DataFrame({
            'Time': [1, 2, 4, 5, 6],
            'Incidence Number': [2, 3, 8, 10, 5]
        })

        my_plot.add_simulation(dfs)

        npt.assert_array_equal(
            np.array(
                [my_plot.figure['data'][1]['x']]
                ),
            np.array(
                [np.array([1, 2, 4, 5, 6])]
                )
        )

        npt.assert_array_equal(
            np.array(
                [my_plot.figure['data'][1]['y']]
                ),
            np.array(
                [np.array([2, 3, 8, 10, 5])]
                )
        )

        with self.assertRaises(TypeError):
            bp.IncidenceNumberPlot().add_simulation(0)

    def test_show_figure(self):
        with patch('plotly.graph_objs.Figure.show') as show_patch:
            df = pd.DataFrame({
                'Time': [1, 2, 3, 5, 6],
                'Incidence Number': [10, 3, 4, 6, 9]
            })
            my_plot = bp.IncidenceNumberPlot()
            my_plot.add_data(df)
            my_plot.show_figure()

        # Assert show_figure is called once
        assert show_patch.called
