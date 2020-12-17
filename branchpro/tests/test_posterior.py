#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import unittest

import pandas as pd
import numpy as np
import numpy.testing as npt

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

    def test_get_serial_intervals(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        inference = bp.BranchProPosterior(df, ser_int, 1, 0.2)
        npt.assert_array_equal(
            inference.get_serial_intervals(), np.array([1, 2]))

    def test_run_inference(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int1 = [1, 2, 1, 0, 0, 0]
        ser_int2 = [1, 2]

        inference1 = bp.BranchProPosterior(df, ser_int1, 1, 0.2)
        inference1.run_inference(tau=2)

        inference2 = bp.BranchProPosterior(df, ser_int2, 1, 0.2)
        inference2.run_inference(tau=2)

        self.assertEqual(len(inference1.inference_estimates), 4)
        self.assertEqual(len(inference1.inference_times), 4)
        self.assertEqual(len(inference1.inference_posterior.mean()), 4)

        self.assertEqual(len(inference2.inference_estimates), 4)
        self.assertEqual(len(inference2.inference_times), 4)
        self.assertEqual(len(inference2.inference_posterior.mean()), 4)

    def test_get_intervals(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        inference = bp.BranchProPosterior(df, ser_int, 1, 0.2)
        inference.run_inference(tau=2)
        intervals_df = inference.get_intervals(.95)

        self.assertEqual(len(intervals_df['Time Points']), 4)
        self.assertEqual(len(intervals_df['Mean']), 4)
        self.assertEqual(len(intervals_df['Lower bound CI']), 4)
        self.assertEqual(len(intervals_df['Upper bound CI']), 4)

        self.assertListEqual(
            intervals_df['Mean'].to_list(), [5.0] * 4)
        self.assertEqual(
            intervals_df['Central Probability'].to_list(), [.95] * 4)
