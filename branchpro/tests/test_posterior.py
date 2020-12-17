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

        with self.assertRaises(TypeError) as test_excep:
            bp.BranchProPosterior('0', ser_int, 1, 0.2)
        self.assertTrue('Incidence data has to' in str(test_excep.exception))

        with self.assertRaises(TypeError) as test_excep:
            bp.BranchProPosterior(df, 0, 1, 0.2)
        self.assertTrue('must be iterable' in str(test_excep.exception))

        with self.assertRaises(TypeError) as test_excep:
            bp.BranchProPosterior(df, ['zero'], 1, 0.2)
        self.assertTrue(
            'distribution must contain' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.BranchProPosterior(df, ser_int, 1, 0.2, time_key='t')
        self.assertTrue('No time column' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.BranchProPosterior(df, ser_int, 1, 0.2, inc_key='i')
        self.assertTrue('No incidence column' in str(test_excep.exception))

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

        self.assertEqual(len(intervals_df['Time Points']), 4)
        self.assertEqual(len(intervals_df['Estimates of Mean']), 4)
        self.assertEqual(len(intervals_df['Lower bound CI']), 4)
        self.assertEqual(len(intervals_df['Upper bound CI']), 4)
