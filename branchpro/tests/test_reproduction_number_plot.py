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


class TestReproductionNumberPlotClass(unittest.TestCase):
    """
    Test the 'ReproductionNumberPlot' class.
    """
    def test__init__(self):
        bp.ReproductionNumberPlot()

    def test_add_data(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 4, 5, 6],
            'Incidence Number': [3, 3, 0.5, 0.5, 0.5]
        })
        my_plot = bp.ReproductionNumberPlot()
        my_plot.add_data(df)

        npt.assert_array_equal(
            np.array(
                [my_plot.figure['data'][0]['x']]
                ),
            np.array(
                [np.array([1, 2, 3, 5, 6])]
                )
        )

        npt.assert_array_equal(
            np.array(
                [my_plot.figure['data'][0]['y']]
                ),
            np.array(
                [np.array([10,  3,  4,  6,  9])]
                )
        )

        with self.assertRaises(TypeError):
            bp.IncidenceNumberPlot().add_data(0)

        with self.assertWarns(UserWarning):
            df = pd.DataFrame({
                't': [1, 2, 3, 5, 6],
                'Incidence Number': [10, 3, 4, 6, 9]
                })
            my_plot.add_data(df, time_key='t')

        with self.assertWarns(UserWarning):
            df = pd.DataFrame({
                'Time': [1, 2, 4, 5, 6],
                'i': [2, 3, 8, 10, 5]
                })
            my_plot.add_data(df, inc_key='i')

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

        with self.assertWarns(UserWarning):
            df = pd.DataFrame({
                'Time': [1, 2, 3, 5, 6],
                'Incidence Number': [10, 3, 4, 6, 9]
                })
            my_plot = bp.IncidenceNumberPlot()
            my_plot.add_data(df)

            dfs1 = pd.DataFrame({
                't': [1, 2, 4, 5, 6],
                'Incidence Number': [2, 3, 8, 10, 5]
                })
            my_plot.add_simulation(dfs1, time_key='t')

        with self.assertWarns(UserWarning):
            df = pd.DataFrame({
                'Time': [1, 2, 3, 5, 6],
                'Incidence Number': [10, 3, 4, 6, 9]
                })
            my_plot = bp.IncidenceNumberPlot()
            my_plot.add_data(df)

            dfs2 = pd.DataFrame({
                'Time': [1, 2, 4, 5, 6],
                'i': [2, 3, 8, 10, 5]
                })
            my_plot.add_simulation(dfs2, inc_key='i')

    def test_update_labels(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        my_plot = bp.IncidenceNumberPlot()
        my_plot.add_data(df)

        new_time_label = 'Week'
        new_inc_label = 'Inc'

        my_plot.update_labels(time_label=new_time_label)
        self.assertEqual(
            my_plot.figure['layout']['xaxis']['title']['text'], 'Week')

        my_plot.update_labels(inc_label=new_inc_label)
        self.assertEqual(
            my_plot.figure['layout']['yaxis']['title']['text'], 'Inc')

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
