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

    def test_set_serial_intervals(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]
        new_ser_int = [1, 2, 1]
        wrong_ser_int = (1)

        inference = bp.BranchProPosterior(df, ser_int, 1, 0.2)
        inference.set_serial_intervals(new_ser_int)

        npt.assert_array_equal(
            inference.get_serial_intervals(), np.array([1, 2, 1]))

        with self.assertRaises(ValueError):
            inference.set_serial_intervals(wrong_ser_int)

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

        self.assertEqual(len(inference1.inference_estimates), 3)
        self.assertEqual(len(inference1.inference_times), 3)
        self.assertEqual(len(inference1.inference_posterior.mean()), 3)

        self.assertEqual(len(inference2.inference_estimates), 3)
        self.assertEqual(len(inference2.inference_times), 3)
        self.assertEqual(len(inference2.inference_posterior.mean()), 3)

    def test_get_intervals(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        inference = bp.BranchProPosterior(df, ser_int, 1, 0.2)
        inference.run_inference(tau=2)
        intervals_df = inference.get_intervals(.95)

        self.assertEqual(len(intervals_df['Time Points']), 3)
        self.assertEqual(len(intervals_df['Mean']), 3)
        self.assertEqual(len(intervals_df['Lower bound CI']), 3)
        self.assertEqual(len(intervals_df['Upper bound CI']), 3)

        self.assertListEqual(
            intervals_df['Mean'].to_list(), [5.0] * 3)
        self.assertEqual(
            intervals_df['Central Probability'].to_list(), [.95] * 3)


#
# TestBranchProPosteriorMultSI Class
#


class TestBranchProPosteriorMultSIClass(unittest.TestCase):
    """
    Test the 'BranchProPosteriorMultSI' class.
    """
    def test__init__(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_ints = [[1, 2], [0, 1]]

        bp.BranchProPosteriorMultSI(df, ser_ints, 1, 0.2)

        with self.assertRaises(TypeError) as test_excep:
            bp.BranchProPosteriorMultSI(df, [[0], 0], 1, 0.2)
        self.assertTrue('must be iterable' in str(test_excep.exception))

        with self.assertRaises(TypeError) as test_excep:
            bp.BranchProPosteriorMultSI(df, [[1], ['zero']], 1, 0.2)
        self.assertTrue(
            'distribution must contain' in str(test_excep.exception))

    def test_get_serial_intervals(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_ints = [[1, 2], [0, 1]]

        inference = bp.BranchProPosteriorMultSI(df, ser_ints, 1, 0.2)
        npt.assert_array_equal(
            inference.get_serial_intervals(), np.array([[1, 2], [0, 1]]))

    def test_set_serial_intervals(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_ints = [[1, 2], [0, 1]]
        new_ser_ints = [[3, 2, 0], [1, 2, 1], [4, 0, 1]]
        wrong_ser_ints = [(1), [2]]

        inference = bp.BranchProPosteriorMultSI(df, ser_ints, 1, 0.2)
        inference.set_serial_intervals(new_ser_ints)

        npt.assert_array_equal(
            inference.get_serial_intervals(),
            np.array([[3, 2, 0], [1, 2, 1], [4, 0, 1]]))

        with self.assertRaises(ValueError):
            inference.set_serial_intervals(wrong_ser_ints)

    def test_run_inference(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int1 = [[1, 2, 1, 0, 0, 0]]
        ser_int2 = [[1, 2], [3, 4]]

        inference1 = bp.BranchProPosteriorMultSI(df, ser_int1, 1, 0.2)
        inference1.run_inference(tau=2)

        inference2 = bp.BranchProPosteriorMultSI(df, ser_int2, 1, 0.2)
        inference2.run_inference(tau=2)

        self.assertEqual(len(inference1.inference_estimates), 3)
        self.assertEqual(len(inference1.inference_times), 3)
        self.assertEqual(len(inference1.inference_posterior.mean()), 3)

        self.assertEqual(len(inference2.inference_estimates), 3)
        self.assertEqual(len(inference2.inference_times), 3)
        self.assertEqual(len(inference2.inference_posterior.mean()), 3)

    def test_get_intervals(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_ints = [[1, 2], [0, 1]]

        inference = bp.BranchProPosteriorMultSI(df, ser_ints, 1, 0.2)
        inference.run_inference(tau=2)
        intervals_df = inference.get_intervals(.95)

        self.assertEqual(len(intervals_df['Time Points']), 3)
        self.assertEqual(len(intervals_df['Mean']), 3)
        self.assertEqual(len(intervals_df['Lower bound CI']), 3)
        self.assertEqual(len(intervals_df['Upper bound CI']), 3)

        npt.assert_allclose(
            intervals_df['Mean'].to_numpy(), np.array([5.0] * 3), atol=0.5)
        self.assertEqual(
            intervals_df['Central Probability'].to_list(), [.95] * 3)


#
# TestLocImpBranchProPosterior Class
#


class TestLocImpBranchProPosteriorClass(unittest.TestCase):
    """
    Test the 'LocImpBranchProPosterior' class.
    """
    def test__init__(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        local_df1 = pd.DataFrame({
            't': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        local_df2 = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'i': [10, 3, 4, 6, 9]
        })

        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })

        ser_int = [1, 2]
        epsilon = 0.3

        bp.LocImpBranchProPosterior(local_df, imp_df, epsilon, ser_int, 1, 0.2)

        with self.assertRaises(TypeError) as test_excep:
            bp.LocImpBranchProPosterior(local_df, imp_df, '0', ser_int, 1, 0.2)
        self.assertTrue('epsilon must be' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.LocImpBranchProPosterior(local_df, imp_df, -3, ser_int, 1, 0.2)
        self.assertTrue('greater or equal to -1' in str(test_excep.exception))

        with self.assertRaises(TypeError) as test_excep:
            bp.LocImpBranchProPosterior(local_df, '0', 0, ser_int, 1, 0.2)
        self.assertTrue(
            'Imported incidence data has to be' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.LocImpBranchProPosterior(
                local_df1, imp_df, 0, ser_int, 1, 0.2, time_key='t')
        self.assertTrue('No time column' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.LocImpBranchProPosterior(
                local_df2, imp_df, 0, ser_int, 1, 0.2, inc_key='i')
        self.assertTrue(
            'No imported incidence column' in str(test_excep.exception))

    def test_set_epsilon(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        inference = bp.LocImpBranchProPosterior(
            local_df, imp_df, 0.3, ser_int, 1, 0.2)
        inference.set_epsilon(1)

        self.assertEqual(inference.epsilon, 1)

    def test_run_inference(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int1 = [1, 2, 1, 0, 0, 0]
        ser_int2 = [1, 2]

        inference1 = bp.LocImpBranchProPosterior(
            local_df, imp_df, 0.3, ser_int1, 1, 0.2)
        inference1.run_inference(tau=2)

        inference2 = bp.LocImpBranchProPosterior(
            local_df, imp_df, 0.3, ser_int2, 1, 0.2)
        inference2.run_inference(tau=2)

        self.assertEqual(len(inference1.inference_estimates), 3)
        self.assertEqual(len(inference1.inference_times), 3)
        self.assertEqual(len(inference1.inference_posterior.mean()), 3)

        self.assertEqual(len(inference2.inference_estimates), 3)
        self.assertEqual(len(inference2.inference_times), 3)
        self.assertEqual(len(inference2.inference_posterior.mean()), 3)


#
# TestLocImpBranchProPosteriorMultSI Class
#


class TestLocImpBranchProPosteriorMultSIClass(unittest.TestCase):
    """
    Test the 'LocImpBranchProPosteriorMultSI' class.
    """
    def test__init__(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })

        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })

        ser_ints = [[1, 2], [0, 1]]
        epsilon = 0.3

        bp.LocImpBranchProPosteriorMultSI(
            local_df, imp_df, epsilon, ser_ints, 1, 0.2)

        with self.assertRaises(TypeError) as test_excep:
            bp.LocImpBranchProPosteriorMultSI(
                local_df, imp_df, epsilon, [[0], 0], 1, 0.2)
        self.assertTrue('must be iterable' in str(test_excep.exception))

        with self.assertRaises(TypeError) as test_excep:
            bp.LocImpBranchProPosteriorMultSI(
                local_df, imp_df, epsilon, [[1], ['zero']], 1, 0.2)
        self.assertTrue(
            'distribution must contain' in str(test_excep.exception))

    def test_run_inference(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int1 = [[1, 2, 1, 0, 0, 0]]
        ser_int2 = [[1, 2], [3, 4]]

        inference1 = bp.LocImpBranchProPosteriorMultSI(
            local_df, imp_df, 0.3, ser_int1, 1, 0.2)
        inference1.run_inference(tau=2)

        inference2 = bp.LocImpBranchProPosteriorMultSI(
            local_df, imp_df, 0.3, ser_int2, 1, 0.2)
        inference2.run_inference(tau=2)

        self.assertEqual(len(inference1.inference_estimates), 3)
        self.assertEqual(len(inference1.inference_times), 3)
        self.assertEqual(len(inference1.inference_posterior.mean()), 3)

        self.assertEqual(len(inference2.inference_estimates), 3)
        self.assertEqual(len(inference2.inference_times), 3)
        self.assertEqual(len(inference2.inference_posterior.mean()), 3)
