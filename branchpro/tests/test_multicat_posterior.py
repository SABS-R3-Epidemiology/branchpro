#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import unittest

import numpy as np
import pandas as pd
import numpy.testing as npt

import branchpro as bp

#
# Test MultiCatPoissonBranchProLogLik Class
#


class TestMultiCatPoissonBranchProLogLik(unittest.TestCase):
    """
    Test the 'MultiCatPoissonBranchProLogLik' class.
    """
    def test__init__(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        ser_int = [1, 2]

        contact_matrix = 0.5 * np.ones((2, 2))
        transm = [1, 1]

        bp.MultiCatPoissonBranchProLogLik(
            df, ser_int, 2, contact_matrix, transm, 6)

        with self.assertRaises(TypeError) as test_excep:
            bp.MultiCatPoissonBranchProLogLik(
                df, ser_int, '2', contact_matrix, transm, 2)
        self.assertTrue(
            'Number of population categories' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.MultiCatPoissonBranchProLogLik(
                df, ser_int, -2, contact_matrix, transm, 2)
        self.assertTrue(
            'Number of population categories' in str(test_excep.exception))

        with self.assertRaises(TypeError) as test_excep:
            bp.MultiCatPoissonBranchProLogLik(
                '0', ser_int, 2, contact_matrix, transm, 2)
        self.assertTrue('Incidence data has to' in str(test_excep.exception))

        with self.assertRaises(TypeError) as test_excep:
            bp.MultiCatPoissonBranchProLogLik(
                df, 0, 2, contact_matrix, transm, 2)
        self.assertTrue('must be iterable' in str(test_excep.exception))

        with self.assertRaises(TypeError) as test_excep:
            bp.MultiCatPoissonBranchProLogLik(
                df, ['zero'], 2, contact_matrix, transm, 2)
        self.assertTrue(
            'distribution must contain' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.MultiCatPoissonBranchProLogLik(
                df, ser_int, 2, contact_matrix, transm, 2, time_key='t')
        self.assertTrue('No time column' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.MultiCatPoissonBranchProLogLik(
                df, ser_int, 2, contact_matrix, transm, 2, inc_key='i')
        self.assertTrue('No incidence column' in str(test_excep.exception))

        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        local_df1 = pd.DataFrame({
            't': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        local_df2 = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'i Cat 1': [10, 3, 4, 6, 9],
            'i Cat 2': [5, 6, 4, 7, 7]
        })

        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })

        ser_int = [1, 2]
        epsilon = 0.3

        bp.MultiCatPoissonBranchProLogLik(
            local_df, ser_int, 2, contact_matrix, transm, 6, imp_df, epsilon)

        with self.assertRaises(TypeError) as test_excep:
            bp.MultiCatPoissonBranchProLogLik(
                local_df, ser_int, 2, contact_matrix, transm, 2, imp_df, '0',)
        self.assertTrue('epsilon must be' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.MultiCatPoissonBranchProLogLik(
                local_df, ser_int, 2, contact_matrix, transm, 2, imp_df, -3,)
        self.assertTrue('greater or equal to 0' in str(test_excep.exception))

        with self.assertRaises(TypeError) as test_excep:
            bp.MultiCatPoissonBranchProLogLik(
                local_df, ser_int, 2, contact_matrix, transm, 2, '0', 0)
        self.assertTrue(
            'Imported incidence data has to be' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.MultiCatPoissonBranchProLogLik(
                local_df1, ser_int, 2, contact_matrix, transm, 2, imp_df, 0,
                time_key='t')
        self.assertTrue('No time column' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.MultiCatPoissonBranchProLogLik(
                local_df2, ser_int, 2, contact_matrix, transm, 2, imp_df, 0,
                inc_key='i')
        self.assertTrue(
            'No imported incidence column' in str(test_excep.exception))

    def test_set_epsilon(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        ser_int = [1, 2]
        contact_matrix = 0.5 * np.ones((2, 2))
        transm = [1, 1]

        log_lik = bp.MultiCatPoissonBranchProLogLik(
            local_df, ser_int, 2, contact_matrix, transm, 2, imp_df, 0.3)
        log_lik.set_epsilon(1)

        self.assertEqual(log_lik.epsilon, 1)

    def test_n_parameters(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        ser_int = [1, 2]
        contact_matrix = 0.5 * np.ones((2, 2))
        transm = [1, 1]

        log_lik = bp.MultiCatPoissonBranchProLogLik(
            df, ser_int, 2, contact_matrix, transm, 3)

        self.assertEqual(log_lik.n_parameters(), 2)

        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        ser_int = [1, 2]

        log_lik = bp.MultiCatPoissonBranchProLogLik(
            local_df, ser_int, 2, contact_matrix, transm, 3, imp_df, 0.3)

        self.assertEqual(log_lik.n_parameters(), 2)

    def test_get_serial_intervals(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        ser_int = [1, 2]
        contact_matrix = 0.5 * np.ones((2, 2))
        transm = [1, 1]

        log_lik = bp.MultiCatPoissonBranchProLogLik(
            df, ser_int, 2, contact_matrix, transm, 3)

        print(log_lik._serial_interval)
        npt.assert_array_equal(
            log_lik.get_serial_intervals(),
            np.array([[1, 2], [1, 2]]))

        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        ser_int = [1, 2]

        log_lik = bp.MultiCatPoissonBranchProLogLik(
            local_df, ser_int, 2, contact_matrix, transm, 3, imp_df, 0.3)
        npt.assert_array_equal(
            log_lik.get_serial_intervals(),
            np.array([[1, 2], [1, 2]]))

    def test_set_serial_intervals(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        ser_int = [1, 2]
        new_ser_int = [1, 2, 1]
        wrong_ser_int = (1)

        contact_matrix = 0.5 * np.ones((2, 2))
        transm = [1, 1]

        log_lik = bp.MultiCatPoissonBranchProLogLik(
            df, ser_int, 2, contact_matrix, transm, 3)
        log_lik.set_serial_intervals(new_ser_int)

        npt.assert_array_equal(
            log_lik.get_serial_intervals(),
            np.array([[1, 2, 1], [1, 2, 1]]))

        with self.assertRaises(ValueError):
            log_lik.set_serial_intervals(wrong_ser_int)

        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        ser_int = [1, 2]
        new_ser_int = [1, 2, 1]
        wrong_ser_int = (1)

        log_lik = bp.MultiCatPoissonBranchProLogLik(
            local_df, ser_int, 2, contact_matrix, transm, 3,
            imp_df, 0.3)
        log_lik.set_serial_intervals(new_ser_int)

        npt.assert_array_equal(
            log_lik.get_serial_intervals(),
            np.array([[1, 2, 1], [1, 2, 1]]))

        with self.assertRaises(ValueError):
            log_lik.set_serial_intervals(wrong_ser_int)

    def test__call__(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [0, 0, 0, 0, 0],
            'Incidence Number Cat 2': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]
        contact_matrix = 0.5 * np.ones((2, 2))
        transm = [1, 1]

        log_lik = bp.MultiCatPoissonBranchProLogLik(
            df, ser_int, 2, contact_matrix, transm, 3)

        self.assertAlmostEqual(log_lik([1, 1]), 0)

        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [0, 0, 0, 0, 0],
            'Incidence Number Cat 2': [0, 0, 0, 0, 0]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [0, 0, 0, 0, 0],
            'Incidence Number Cat 2': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        log_lik = bp.MultiCatPoissonBranchProLogLik(
            local_df, ser_int, 2, contact_matrix, transm, 3,
            imp_df, 0.3)

        self.assertAlmostEqual(log_lik([1, 1]), 0)

    def test_evaluateS1(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [0, 0, 0, 0, 0],
            'Incidence Number Cat 2': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]
        contact_matrix = 0.5 * np.ones((2, 2))
        transm = [1, 1]

        log_lik = bp.MultiCatPoissonBranchProLogLik(
            df, ser_int, 2, contact_matrix, transm, 3)

        self.assertAlmostEqual(log_lik.evaluateS1([1, 1])[0], 0)
        npt.assert_almost_equal(
            np.array(log_lik.evaluateS1([1, 1])[1]),
            np.array([0, 0]))

        dLl = []
        old_r_profile = [1, 1]

        total_time = log_lik.cases_times.max() - log_lik.cases_times.min() + 1
        time_init_inf_r = log_lik._tau + 1

        for t, _ in enumerate(range(time_init_inf_r+1, total_time+1)):
            new_r_profile = old_r_profile.copy()
            new_r_profile[t] = old_r_profile[t] + 10**(-5)
            dLl.append(
                (log_lik(new_r_profile)-log_lik(old_r_profile))/10**(-5))

        npt.assert_almost_equal(
            np.array(dLl),
            log_lik._compute_derivative_log_likelihood(old_r_profile),
            decimal=0)

        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [0, 0, 0, 0, 0],
            'Incidence Number Cat 2': [0, 0, 0, 0, 0]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [0, 0, 0, 0, 0],
            'Incidence Number Cat 2': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        log_lik = bp.MultiCatPoissonBranchProLogLik(
            local_df, ser_int, 2, contact_matrix, transm, 3,
            imp_df, 0.3)

        self.assertAlmostEqual(log_lik.evaluateS1([1, 1])[0], 0)
        npt.assert_almost_equal(
            np.array(log_lik.evaluateS1([1, 1])[1]),
            np.array([0, 0]))

        dLl = []
        old_r_profile = [1, 1]

        total_time = log_lik.cases_times.max() - log_lik.cases_times.min() + 1
        time_init_inf_r = log_lik._tau + 1

        for t, _ in enumerate(range(time_init_inf_r+1, total_time+1)):
            new_r_profile = old_r_profile.copy()
            new_r_profile[t] = old_r_profile[t] + 10**(-5)
            dLl.append(
                (log_lik(new_r_profile)-log_lik(old_r_profile))/10**(-5))

        npt.assert_almost_equal(
            np.array(dLl),
            log_lik._compute_derivative_log_likelihood(old_r_profile),
            decimal=0)

#
# Test MultiCatPoissonBranchProLogPosterior Class
#


class TestMultiCatPoissonBranchProLogPosteriorClass(unittest.TestCase):
    """
    Test the 'MultiCatPoissonBranchProLogPosterior' class.
    """
    def test__init__(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        ser_int = [1, 2]
        contact_matrix = 0.5 * np.ones((2, 2))
        transm = [1, 1]

        bp.MultiCatPoissonBranchProLogPosterior(
            df, ser_int, 2, contact_matrix, transm, 2, 1, 0.2)

        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [0, 0, 0, 0, 0],
            'Incidence Number Cat 2': [0, 0, 0, 0, 0]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [0, 0, 0, 0, 0],
            'Incidence Number Cat 2': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        bp.MultiCatPoissonBranchProLogPosterior(
            local_df, ser_int, 2, contact_matrix, transm, 2, 1, 0.2,
            imp_df, 0.3)

    def test_return_loglikelihood(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        ser_int = [1, 2]
        contact_matrix = 0.5 * np.ones((2, 2))
        transm = [1, 1]

        inference = bp.MultiCatPoissonBranchProLogPosterior(
            df, ser_int, 2, contact_matrix, transm, 2, 1, 0.2)

        self.assertEqual(
            inference.return_loglikelihood([1, 1, 1]),
            inference.ll([1, 1, 1]))

        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [0, 0, 0, 0, 0],
            'Incidence Number Cat 2': [0, 0, 0, 0, 0]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [0, 0, 0, 0, 0],
            'Incidence Number Cat 2': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        inference = bp.MultiCatPoissonBranchProLogPosterior(
            local_df, ser_int, 2, contact_matrix, transm, 2, 1, 0.2,
            imp_df, 0.3)

        self.assertEqual(
            inference.return_loglikelihood([1, 1, 1]),
            inference.ll([1, 1, 1]))

    def test_return_logprior(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        ser_int = [1, 2]
        contact_matrix = 0.5 * np.ones((2, 2))
        transm = [1, 1]

        inference = bp.MultiCatPoissonBranchProLogPosterior(
            df, ser_int, 2, contact_matrix, transm, 2, 1, 0.2)

        self.assertEqual(
            inference.return_logprior([1, 1, 1]),
            inference.lprior([1, 1, 1]))

        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [0, 0, 0, 0, 0],
            'Incidence Number Cat 2': [0, 0, 0, 0, 0]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [0, 0, 0, 0, 0],
            'Incidence Number Cat 2': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        inference = bp.MultiCatPoissonBranchProLogPosterior(
            local_df, ser_int, 2, contact_matrix, transm, 2, 1, 0.2,
            imp_df, 0.3)

        self.assertEqual(
            inference.return_logprior([1, 1, 1]),
            inference.lprior([1, 1, 1]))

    def test_return_logposterior(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        ser_int = [1, 2]
        contact_matrix = 0.5 * np.ones((2, 2))
        transm = [1, 1]

        inference = bp.MultiCatPoissonBranchProLogPosterior(
            df, ser_int, 2, contact_matrix, transm, 2, 1, 0.2)

        self.assertEqual(
            inference.return_logposterior([1, 1, 1]),
            inference._log_posterior([1, 1, 1]))

        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [0, 0, 0, 0, 0],
            'Incidence Number Cat 2': [0, 0, 0, 0, 0]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [0, 0, 0, 0, 0],
            'Incidence Number Cat 2': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        inference = bp.MultiCatPoissonBranchProLogPosterior(
            local_df, ser_int, 2, contact_matrix, transm, 2, 1, 0.2,
            imp_df, 0.3)

        self.assertEqual(
            inference.return_logposterior([1, 1, 1]),
            inference._log_posterior([1, 1, 1]))

    def test_run_inference(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        ser_int1 = [1, 2, 1, 0, 0, 0]
        ser_int2 = [1, 2]

        contact_matrix = 0.5 * np.ones((2, 2))
        transm = [1, 1]

        inference1 = bp.MultiCatPoissonBranchProLogPosterior(
            df, ser_int1, 2, contact_matrix, transm, 2, 1, 0.2)

        samples = inference1.run_inference(num_iter=100)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (100, 3))

        inference2 = bp.MultiCatPoissonBranchProLogPosterior(
            df, ser_int2, 2, contact_matrix, transm, 2, 1, 0.2)

        samples = inference2.run_inference(num_iter=100)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (100, 3))

        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        ser_int1 = [1, 2, 1, 0, 0, 0]
        ser_int2 = [1, 2]

        inference1 = bp.MultiCatPoissonBranchProLogPosterior(
            local_df, ser_int1, 2, contact_matrix, transm, 2, 1, 0.2,
            imp_df, 0.3)

        samples = inference1.run_inference(num_iter=100)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (100, 3))

        inference2 = bp.MultiCatPoissonBranchProLogPosterior(
            local_df, ser_int2, 2, contact_matrix, transm, 2, 1, 0.2,
            imp_df, 0.3)

        samples = inference2.run_inference(num_iter=100)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (100, 3))

    def test_run_optimisation(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        ser_int1 = [1, 2, 1, 0, 0, 0]
        ser_int2 = [1, 2]

        contact_matrix = 0.5 * np.ones((2, 2))
        transm = [1, 1]

        optimisation1 = bp.MultiCatPoissonBranchProLogPosterior(
            df, ser_int1, 2, contact_matrix, transm, 2, 1, 0.2)

        found, val = optimisation1.run_optimisation()

        self.assertTrue(isinstance(val, (int, float)))
        self.assertEqual(len(found), 3)

        optimisation2 = bp.MultiCatPoissonBranchProLogPosterior(
            df, ser_int2, 2, contact_matrix, transm, 2, 1, 0.2)

        found, val = optimisation2.run_optimisation()

        self.assertTrue(isinstance(val, (int, float)))
        self.assertEqual(len(found), 3)

        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [10, 3, 4, 6, 9],
            'Incidence Number Cat 2': [5, 6, 4, 7, 7]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number Cat 1': [0, 0, 0, 0, 0],
            'Incidence Number Cat 2': [0, 0, 0, 0, 0]
        })
        ser_int1 = [1, 2, 1, 0, 0, 0]
        ser_int2 = [1, 2]

        optimisation1 = bp.MultiCatPoissonBranchProLogPosterior(
            local_df, ser_int1, 2, contact_matrix, transm, 2, 1, 0.2,
            imp_df, 0.3)

        found, val = optimisation1.run_optimisation()

        self.assertTrue(isinstance(val, (int, float)))
        self.assertEqual(len(found), 3)

        optimisation2 = bp.MultiCatPoissonBranchProLogPosterior(
            local_df, ser_int2, 2, contact_matrix, transm, 2, 1, 0.2,
            imp_df, 0.3)

        found, val = optimisation2.run_optimisation()

        self.assertTrue(isinstance(val, (int, float)))
        self.assertEqual(len(found), 3)
