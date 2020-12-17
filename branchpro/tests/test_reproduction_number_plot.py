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

    def test_add_ground_truth_rt(self):
        df = pd.DataFrame({
            'Time Points': [1, 2, 3, 4, 5, 6],
            'R_t': [3, 3, 0.5, 0.5, 0.5, 0.5]
        })
        my_plot = bp.ReproductionNumberPlot()
        my_plot.add_ground_truth_rt(df)

        npt.assert_array_equal(
            np.array(
                [my_plot.figure['data'][0]['x']]
                ),
            np.array(
                [np.array([1, 2, 3, 4, 5, 6])]
                )
        )

        npt.assert_array_equal(
            np.array(
                [my_plot.figure['data'][0]['y']]
                ),
            np.array(
                [np.array([3, 3, 0.5, 0.5, 0.5, 0.5])]
                )
        )

        with self.assertRaises(TypeError):
            bp.ReproductionNumberPlot().add_ground_truth_rt(0)

        with self.assertWarns(UserWarning):
            df = pd.DataFrame({
                't': [1, 2, 3, 4, 5, 6],
                'R_t': [3, 3, 0.5, 0.5, 0.5, 0.5]
                })
            my_plot.add_ground_truth_rt(df, time_key='t')

        with self.assertWarns(UserWarning):
            df = pd.DataFrame({
                'Time Points': [1, 2, 3, 4, 5, 6],
                'r': [3, 3, 0.5, 0.5, 0.5, 0.5]
                })
            my_plot.add_ground_truth_rt(df, r_key='r')

    def test_add_interval_rt(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        inference = bp.BranchProPosterior(df, ser_int, 1, 0.2)
        inference.run_inference(tau=2)
        intervals_df = inference.get_intervals(.95)

        my_plot = bp.ReproductionNumberPlot()
        my_plot.add_interval_rt(intervals_df)

        npt.assert_array_equal(
            np.array(
                [my_plot.figure['data'][0]['x']]
                ),
            np.array(
                [np.array([3, 4, 5, 6])]
                )
        )

        npt.assert_array_equal(
            np.array(
                [my_plot.figure['data'][0]['y']]
                ),
            np.array(
                [np.array([5.0] * 4)]
                )
        )
        npt.assert_array_equal(
            np.array(
                [my_plot.figure['data'][1]['x']]
                ),
            np.array(
                [np.array([3, 4, 5, 6])]
                )
        )

        npt.assert_array_almost_equal(
            np.array(
                [my_plot.figure['data'][1]['y']]
                ),
            np.array(
                [np.array([0.126589] * 4)]
                )
        )

        npt.assert_array_equal(
            np.array(
                [my_plot.figure['data'][2]['x']]
                ),
            np.array(
                [np.array([3, 4, 5, 6])]
                )
        )

        npt.assert_array_almost_equal(
            np.array(
                [my_plot.figure['data'][2]['y']]
                ),
            np.array(
                [np.array([18.444397] * 4)]
                )
        )

        with self.assertRaises(TypeError):
            bp.ReproductionNumberPlot().add_interval_rt(0)

        with self.assertWarns(UserWarning):
            df = pd.DataFrame({
                'Time Points': [1, 2, 3, 4, 5, 6],
                'R_t': [3, 3, 0.5, 0.5, 0.5, 0.5]
                })
            my_plot = bp.ReproductionNumberPlot()
            my_plot.add_ground_truth_rt(df)

            dfs1 = pd.DataFrame({
                't': [3, 4, 5, 6],
                'Mean': [5.0] * 4,
                'Lower bound CI': [5.0] * 4,
                'Upper bound CI': [5.0] * 4
                })
            my_plot.add_interval_rt(dfs1, time_key='t')

        with self.assertWarns(UserWarning):
            df = pd.DataFrame({
                'Time Points': [1, 2, 3, 4, 5, 6],
                'R_t': [3, 3, 0.5, 0.5, 0.5, 0.5]
                })
            my_plot = bp.ReproductionNumberPlot()
            my_plot.add_ground_truth_rt(df)

            dfs2 = pd.DataFrame({
                'Time Points': [3, 4, 5, 6],
                'r': [5.0] * 4,
                'Lower bound CI': [5.0] * 4,
                'Upper bound CI': [5.0] * 4
                })
            my_plot.add_interval_rt(dfs2, r_key='r')

    def test_update_labels(self):
        df = pd.DataFrame({
            'Time Points': [1, 2, 3, 4, 5, 6],
            'R_t': [3, 3, 0.5, 0.5, 0.5, 0.5]
            })
        my_plot = bp.ReproductionNumberPlot()
        my_plot.add_ground_truth_rt(df)

        new_time_label = 'Time'
        new_r_label = 'R Value'

        my_plot.update_labels(time_label=new_time_label)
        self.assertEqual(
            my_plot.figure['layout']['xaxis']['title']['text'], 'Time')

        my_plot.update_labels(r_label=new_r_label)
        self.assertEqual(
            my_plot.figure['layout']['yaxis']['title']['text'], 'R Value')

    def test_show_figure(self):
        with patch('plotly.graph_objs.Figure.show') as show_patch:
            df = pd.DataFrame({
                'Time Points': [1, 2, 3, 4, 5, 6],
                'R_t': [3, 3, 0.5, 0.5, 0.5, 0.5]
            })
            my_plot = bp.ReproductionNumberPlot()
            my_plot.add_ground_truth_rt(df)
            my_plot.show_figure()

        # Assert show_figure is called once
        assert show_patch.called
