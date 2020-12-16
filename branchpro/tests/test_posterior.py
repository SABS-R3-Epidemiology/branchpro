#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import unittest

import pandas as pd

import branchpro as bp


class TestBranchProPosteriorClass(unittest.TestCase):
    """
    Test the 'BranchProPosterior' class.
    """
    def test__init__(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        bp.BranchProPosterior(df, ser_int, 1, 0.2)

        with self.assertRaises(TypeError):
            bp.BranchProPosterior('0', ser_int, 1, 0.2)

        with self.assertRaises(TypeError):
            bp.BranchProPosterior(df, 0, 1, 0.2)

        with self.assertRaises(ValueError):
            bp.BranchProPosterior(df, ser_int, 1, 0.2, time_key='t')

        with self.assertRaises(ValueError):
            bp.BranchProPosterior(df, ser_int, 1, 0.2, inc_key='i')

    def test_run_inference(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2, 1, 0, 0, 0]

        inference = bp.BranchProPosterior(df, ser_int, 1, 0.2)
        inference.run_inference(tau=2)

        self.assertEqual(len(inference.inference_estimates), 4)
        self.assertEqual(len(inference.inference_times), 4)
        self.assertEqual(len(inference.inference_posterior.mean()), 4)

    def test_get_intervals(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2, 1, 0, 0, 0]

        inference = bp.BranchProPosterior(df, ser_int, 1, 0.2)
        inference.run_inference(tau=2)
        intervals_df = inference.get_intervals(.95)

        self.assertEqual(len(intervals_df['Estimates']), 4)
        self.assertEqual(len(intervals_df['Lower bound CI']), 4)
        self.assertEqual(len(intervals_df['Upper bound CI']), 4)
