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

#
# Test PoissonBranchProLogLik Class
#


class TestPoissonBranchProLogLik(unittest.TestCase):
    """
    Test the 'PoissonBranchProLogLik' class.
    """
    def test__init__(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        bp.PoissonBranchProLogLik(df, ser_int, 6)

        with self.assertRaises(TypeError) as test_excep:
            bp.PoissonBranchProLogLik('0', ser_int, 2)
        self.assertTrue('Incidence data has to' in str(test_excep.exception))

        with self.assertRaises(TypeError) as test_excep:
            bp.PoissonBranchProLogLik(df, 0, 2)
        self.assertTrue('must be iterable' in str(test_excep.exception))

        with self.assertRaises(TypeError) as test_excep:
            bp.PoissonBranchProLogLik(df, ['zero'], 2)
        self.assertTrue(
            'distribution must contain' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.PoissonBranchProLogLik(df, ser_int, 2, time_key='t')
        self.assertTrue('No time column' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.PoissonBranchProLogLik(df, ser_int, 2, inc_key='i')
        self.assertTrue('No incidence column' in str(test_excep.exception))

    def test_n_parameters(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        log_lik = bp.PoissonBranchProLogLik(df, ser_int, 3)

        self.assertEqual(log_lik.n_parameters(), 2)

    def test_get_serial_intervals(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        log_lik = bp.PoissonBranchProLogLik(df, ser_int, 3)
        npt.assert_array_equal(
            log_lik.get_serial_intervals(), np.array([1, 2]))

    def test_set_serial_intervals(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]
        new_ser_int = [1, 2, 1]
        wrong_ser_int = (1)

        log_lik = bp.PoissonBranchProLogLik(df, ser_int, 3)
        log_lik.set_serial_intervals(new_ser_int)

        npt.assert_array_equal(
            log_lik.get_serial_intervals(), np.array([1, 2, 1]))

        with self.assertRaises(ValueError):
            log_lik.set_serial_intervals(wrong_ser_int)

    def test__call__(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        log_lik = bp.PoissonBranchProLogLik(df, ser_int, 3)

        self.assertEqual(log_lik([1, 1]), 0)

    def test_evaluateS1(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        log_lik = bp.PoissonBranchProLogLik(df, ser_int, 3)

        self.assertEqual(log_lik.evaluateS1([1, 1]), (0, [0, 0]))

#
# Test PoissonBranchProLogPosterior Class
#


class TestPoissonBranchProLogPosteriorClass(unittest.TestCase):
    """
    Test the 'PoissonBranchProLogPosterior' class.
    """
    def test__init__(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        bp.PoissonBranchProLogPosterior(df, ser_int, 2, 1, 0.2)

    def test_return_loglikelihood(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        inference = bp.PoissonBranchProLogPosterior(df, ser_int, 2, 1, 0.2)

        self.assertEqual(
            inference.return_loglikelihood([1, 1, 1]),
            inference.ll([1, 1, 1]))

    def test_return_logprior(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        inference = bp.PoissonBranchProLogPosterior(df, ser_int, 2, 1, 0.2)

        self.assertEqual(
            inference.return_logprior([1, 1, 1]),
            inference.lprior([1, 1, 1]))

    def test_return_logposterior(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        inference = bp.PoissonBranchProLogPosterior(df, ser_int, 2, 1, 0.2)

        self.assertEqual(
            inference.return_logposterior([1, 1, 1]),
            inference._log_posterior([1, 1, 1]))

    def test_run_inference(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int1 = [1, 2, 1, 0, 0, 0]
        ser_int2 = [1, 2]

        inference1 = bp.PoissonBranchProLogPosterior(df, ser_int1, 2, 1, 0.2)

        samples = inference1.run_inference(num_iter=100)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (100, 3))

        inference2 = bp.PoissonBranchProLogPosterior(df, ser_int2, 2, 1, 0.2)

        samples = inference2.run_inference(num_iter=100)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (100, 3))

    def test_run_optimisation(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int1 = [1, 2, 1, 0, 0, 0]
        ser_int2 = [1, 2]

        optimisation1 = bp.PoissonBranchProLogPosterior(
            df, ser_int1, 2, 1, 0.2)

        found, val = optimisation1.run_optimisation()

        self.assertTrue(isinstance(val, (int, float)))
        self.assertEqual(len(found), 3)

        optimisation2 = bp.PoissonBranchProLogPosterior(
            df, ser_int2, 2, 1, 0.2)

        found, val = optimisation2.run_optimisation()

        self.assertTrue(isinstance(val, (int, float)))
        self.assertEqual(len(found), 3)

#
# Test LocImpPoissonBranchProLogLik Class
#


class TestLocImpPoissonBranchProLogLik(unittest.TestCase):
    """
    Test the 'LocImpPoissonBranchProLogLik' class.
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

        bp.LocImpPoissonBranchProLogLik(local_df, imp_df, epsilon, ser_int, 6)

        with self.assertRaises(TypeError) as test_excep:
            bp.LocImpPoissonBranchProLogLik(local_df, imp_df, '0', ser_int, 2)
        self.assertTrue('epsilon must be' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.LocImpPoissonBranchProLogLik(local_df, imp_df, -3, ser_int, 2)
        self.assertTrue('greater or equal to 0' in str(test_excep.exception))

        with self.assertRaises(TypeError) as test_excep:
            bp.LocImpPoissonBranchProLogLik(local_df, '0', 0, ser_int, 2)
        self.assertTrue(
            'Imported incidence data has to be' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.LocImpPoissonBranchProLogLik(
                local_df1, imp_df, 0, ser_int, 2, time_key='t')
        self.assertTrue('No time column' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.LocImpPoissonBranchProLogLik(
                local_df2, imp_df, 0, ser_int, 2, inc_key='i')
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

        log_lik = bp.LocImpPoissonBranchProLogLik(
            local_df, imp_df, 0.3, ser_int, 2)
        log_lik.set_epsilon(1)

        self.assertEqual(log_lik.epsilon, 1)

    def test_n_parameters(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        log_lik = bp.LocImpPoissonBranchProLogLik(
            local_df, imp_df, 0.3, ser_int, 3)

        self.assertEqual(log_lik.n_parameters(), 2)

    def test_get_serial_intervals(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        log_lik = bp.LocImpPoissonBranchProLogLik(
            local_df, imp_df, 0.3, ser_int, 3)
        npt.assert_array_equal(
            log_lik.get_serial_intervals(), np.array([1, 2]))

    def test_set_serial_intervals(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]
        new_ser_int = [1, 2, 1]
        wrong_ser_int = (1)

        log_lik = bp.LocImpPoissonBranchProLogLik(
            local_df, imp_df, 0.3, ser_int, 3)
        log_lik.set_serial_intervals(new_ser_int)

        npt.assert_array_equal(
            log_lik.get_serial_intervals(), np.array([1, 2, 1]))

        with self.assertRaises(ValueError):
            log_lik.set_serial_intervals(wrong_ser_int)

    def test__call__(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        log_lik = bp.LocImpPoissonBranchProLogLik(
            local_df, imp_df, 0.3, ser_int, 3)

        self.assertEqual(log_lik([1, 1]), 0)

    def test_evaluateS1(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        log_lik = bp.LocImpPoissonBranchProLogLik(
            local_df, imp_df, 0.3, ser_int, 3)

        self.assertEqual(log_lik.evaluateS1([1, 1]), (0, [0, 0]))

#
# Test LocImpPoissonBranchProLogPosterior Class
#


class TestLocImpPoissonBranchProLogPosteriorClass(unittest.TestCase):
    """
    Test the 'LocImpPoissonBranchProLogPosterior' class.
    """
    def test__init__(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        bp.LocImpPoissonBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int, 2, 1, 0.2)

    def test_return_loglikelihood(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        inference = bp.LocImpPoissonBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int, 2, 1, 0.2)

        self.assertEqual(
            inference.return_loglikelihood([1, 1, 1]),
            inference.ll([1, 1, 1]))

    def test_return_logprior(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        inference = bp.LocImpPoissonBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int, 2, 1, 0.2)

        self.assertEqual(
            inference.return_logprior([1, 1, 1]),
            inference.lprior([1, 1, 1]))

    def test_return_logposterior(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        inference = bp.LocImpPoissonBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int, 2, 1, 0.2)

        self.assertEqual(
            inference.return_logposterior([1, 1, 1]),
            inference._log_posterior([1, 1, 1]))

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

        inference1 = bp.LocImpPoissonBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int1, 2, 1, 0.2)

        samples = inference1.run_inference(num_iter=100)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (100, 3))

        inference2 = bp.LocImpPoissonBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int2, 2, 1, 0.2)

        samples = inference2.run_inference(num_iter=100)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (100, 3))

    def test_run_optimisation(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int1 = [1, 2, 1, 0, 0, 0]
        ser_int2 = [1, 2]

        optimisation1 = bp.LocImpPoissonBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int1, 2, 1, 0.2)

        found, val = optimisation1.run_optimisation()

        self.assertTrue(isinstance(val, (int, float)))
        self.assertEqual(len(found), 3)

        optimisation2 = bp.LocImpPoissonBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int2, 2, 1, 0.2)

        found, val = optimisation2.run_optimisation()

        self.assertTrue(isinstance(val, (int, float)))
        self.assertEqual(len(found), 3)

#
# Test NegBinBranchProLogLik Class
#


class TestNegBinBranchProLogLik(unittest.TestCase):
    """
    Test the 'NegBinBranchProLogLik' class.
    """
    def test__init__(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        bp.NegBinBranchProLogLik(df, ser_int, 6, 0.5)
        log_lik = bp.NegBinBranchProLogLik(df, ser_int, 6, 0.5, False)

        self.assertEqual(log_lik._infer_phi, False)

        with self.assertRaises(TypeError) as test_excep:
            bp.NegBinBranchProLogLik('0', ser_int, 2, 0.5)
        self.assertTrue('Incidence data has to' in str(test_excep.exception))

        with self.assertRaises(TypeError) as test_excep:
            bp.NegBinBranchProLogLik(df, 0, 2, 0.5)
        self.assertTrue('must be iterable' in str(test_excep.exception))

        with self.assertRaises(TypeError) as test_excep:
            bp.NegBinBranchProLogLik(df, ['zero'], 2, 0.5)
        self.assertTrue(
            'distribution must contain' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.NegBinBranchProLogLik(df, ser_int, 2, 0.5, time_key='t')
        self.assertTrue('No time column' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.NegBinBranchProLogLik(df, ser_int, 2, 0.5, inc_key='i')
        self.assertTrue('No incidence column' in str(test_excep.exception))

    def test_set_overdispersion(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        log_lik = bp.NegBinBranchProLogLik(df, ser_int, 6, 0.5)
        log_lik.set_overdispersion(1)

        self.assertEqual(log_lik._overdispersion, 1)

        with self.assertRaises(TypeError) as test_excep:
            log_lik.set_overdispersion('1')
        self.assertTrue(
            'overdispersion must be integer' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            log_lik.set_overdispersion(0)

    def test_get_overdispersion(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        log_lik = bp.NegBinBranchProLogLik(df, ser_int, 6, 0.5)

        self.assertEqual(log_lik.get_overdispersion(), 0.5)

    def test_n_parameters(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        log_lik = bp.NegBinBranchProLogLik(df, ser_int, 3, 0.5)
        log_lik1 = bp.NegBinBranchProLogLik(df, ser_int, 3, 0.5, False)

        self.assertEqual(log_lik.n_parameters(), 3)
        self.assertEqual(log_lik1.n_parameters(), 2)

    def test_get_serial_intervals(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        log_lik = bp.NegBinBranchProLogLik(df, ser_int, 3, 0.5)
        npt.assert_array_equal(
            log_lik.get_serial_intervals(), np.array([1, 2]))

    def test_set_serial_intervals(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]
        new_ser_int = [1, 2, 1]
        wrong_ser_int = (1)

        log_lik = bp.NegBinBranchProLogLik(df, ser_int, 3, 0.5)
        log_lik.set_serial_intervals(new_ser_int)

        npt.assert_array_equal(
            log_lik.get_serial_intervals(), np.array([1, 2, 1]))

        with self.assertRaises(ValueError):
            log_lik.set_serial_intervals(wrong_ser_int)

    def test__call__(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        log_lik = bp.NegBinBranchProLogLik(df, ser_int, 3, 0.5)
        log_lik1 = bp.NegBinBranchProLogLik(df, ser_int, 3, 0.5, False)

        self.assertEqual(log_lik([1, 1, 0.5]), 0)
        self.assertEqual(log_lik1([1, 1]), 0)

    def test_evaluateS1(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        log_lik = bp.NegBinBranchProLogLik(df, ser_int, 3, 0.5)
        log_lik1 = bp.NegBinBranchProLogLik(df, ser_int, 3, 0.5, False)

        self.assertEqual(log_lik.evaluateS1([1, 1, 0.5]), (0, [0, 0, 0]))
        self.assertEqual(log_lik1.evaluateS1([1, 1]), (0, [0, 0]))

#
# Test NegBinBranchProLogPosterior Class
#


class TestNegBinBranchProLogPosteriorClass(unittest.TestCase):
    """
    Test the 'NegBinBranchProLogPosterior' class.
    """
    def test__init__(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        bp.NegBinBranchProLogPosterior(df, ser_int, 2, 0.5, 1, 0.2)
        bp.NegBinBranchProLogPosterior(df, ser_int, 2, 0.5, 1, 0.2, 1, False)

    def test_return_loglikelihood(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        inference = bp.NegBinBranchProLogPosterior(df, ser_int, 2, 0.5, 1, 0.2)
        inference1 = bp.NegBinBranchProLogPosterior(
            df, ser_int, 2, 0.5, 1, 0.2, 1, False)

        self.assertEqual(
            inference.return_loglikelihood([1, 1, 1, 0.5]),
            inference.ll([1, 1, 1, 0.5]))

        self.assertEqual(
            inference1.return_loglikelihood([1, 1, 1]),
            inference1.ll([1, 1, 1]))

    def test_return_logprior(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        inference = bp.NegBinBranchProLogPosterior(df, ser_int, 2, 0.5, 1, 0.2)
        inference1 = bp.NegBinBranchProLogPosterior(
            df, ser_int, 2, 0.5, 1, 0.2, 1, False)

        self.assertEqual(
            inference.return_logprior([1, 1, 1, 0.5]),
            inference.lprior([1, 1, 1, 0.5]))

        self.assertEqual(
            inference1.return_logprior([1, 1, 1]),
            inference1.lprior([1, 1, 1]))

    def test_return_logposterior(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        inference = bp.NegBinBranchProLogPosterior(df, ser_int, 2, 0.5, 1, 0.2)
        inference1 = bp.NegBinBranchProLogPosterior(
            df, ser_int, 2, 0.5, 1, 0.2, 1, False)

        self.assertEqual(
            inference.return_logposterior([1, 1, 1, 0.5]),
            inference._log_posterior([1, 1, 1, 0.5]))

        self.assertEqual(
            inference1.return_logposterior([1, 1, 1]),
            inference1._log_posterior([1, 1, 1]))

    def test_run_inference(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int1 = [1, 2, 1, 0, 0, 0]
        ser_int2 = [1, 2]

        inference1 = bp.NegBinBranchProLogPosterior(
            df, ser_int1, 2, 0.5, 1, 0.2)

        samples = inference1.run_inference(num_iter=100)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (100, 4))

        inference2 = bp.NegBinBranchProLogPosterior(
            df, ser_int2, 2, 0.5, 1, 0.2)

        samples = inference2.run_inference(num_iter=100)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (100, 4))

        inference3 = bp.NegBinBranchProLogPosterior(
            df, ser_int1, 2, 0.5, 1, 0.2, 1, False)

        samples = inference3.run_inference(num_iter=100)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (100, 3))

        inference4 = bp.NegBinBranchProLogPosterior(
            df, ser_int2, 2, 0.5, 1, 0.2, 1, False)

        samples = inference4.run_inference(num_iter=100)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (100, 3))

    def test_run_optimisation(self):
        df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int1 = [1, 2, 1, 0, 0, 0]
        ser_int2 = [1, 2]

        optimisation1 = bp.NegBinBranchProLogPosterior(
            df, ser_int1, 2, 0.5, 1, 0.2, 2)

        found, val = optimisation1.run_optimisation()

        self.assertTrue(isinstance(val, (int, float)))
        self.assertEqual(len(found), 4)

        optimisation2 = bp.NegBinBranchProLogPosterior(
            df, ser_int2, 2, 0.5, 1, 0.2)

        found, val = optimisation2.run_optimisation()

        self.assertTrue(isinstance(val, (int, float)))
        self.assertEqual(len(found), 4)

        optimisation3 = bp.NegBinBranchProLogPosterior(
            df, ser_int1, 2, 0.5, 1, 0.2, 1, False)

        found, val = optimisation3.run_optimisation()

        self.assertTrue(isinstance(val, (int, float)))
        self.assertEqual(len(found), 3)

        optimisation4 = bp.NegBinBranchProLogPosterior(
            df, ser_int2, 2, 0.5, 1, 0.2, 1, False)

        found, val = optimisation4.run_optimisation()

        self.assertTrue(isinstance(val, (int, float)))
        self.assertEqual(len(found), 3)

#
# Test LocImpNegBinBranchProLogLik Class
#


class TestLocImpNegBinBranchProLogLik(unittest.TestCase):
    """
    Test the 'LocImpNegBinBranchProLogLik' class.
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

        bp.LocImpNegBinBranchProLogLik(
            local_df, imp_df, epsilon, ser_int, 6, 0.5)

        log_lik = bp.LocImpNegBinBranchProLogLik(
            local_df, imp_df, epsilon, ser_int, 6, 0.5, False)

        self.assertEqual(log_lik._infer_phi, False)

        with self.assertRaises(TypeError) as test_excep:
            bp.LocImpNegBinBranchProLogLik(
                local_df, imp_df, '0', ser_int, 2, 0.5)
        self.assertTrue('epsilon must be' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.LocImpNegBinBranchProLogLik(
                local_df, imp_df, -3, ser_int, 2, 0.5)
        self.assertTrue('greater or equal to 0' in str(test_excep.exception))

        with self.assertRaises(TypeError) as test_excep:
            bp.LocImpNegBinBranchProLogLik(
                local_df, '0', 0, ser_int, 2, 0.5)
        self.assertTrue(
            'Imported incidence data has to be' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.LocImpNegBinBranchProLogLik(
                local_df1, imp_df, 0, ser_int, 2, 0.5, time_key='t')
        self.assertTrue('No time column' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            bp.LocImpNegBinBranchProLogLik(
                local_df2, imp_df, 0, ser_int, 2, 0.5, inc_key='i')
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

        log_lik = bp.LocImpNegBinBranchProLogLik(
            local_df, imp_df, 0.3, ser_int, 2, 0.5)
        log_lik.set_epsilon(1)

        self.assertEqual(log_lik.epsilon, 1)

    def test_set_overdispersion(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        log_lik = bp.LocImpNegBinBranchProLogLik(
            local_df, imp_df, 0.3, ser_int, 2, 0.5)
        log_lik.set_overdispersion(1)

        self.assertEqual(log_lik._overdispersion, 1)

        with self.assertRaises(TypeError) as test_excep:
            log_lik.set_overdispersion('1')
        self.assertTrue(
            'overdispersion must be integer' in str(test_excep.exception))

        with self.assertRaises(ValueError) as test_excep:
            log_lik.set_overdispersion(-1)
        self.assertTrue('overdispersion must be > 0' in str(
            test_excep.exception))

    def test_get_overdispersion(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        log_lik = bp.LocImpNegBinBranchProLogLik(
            local_df, imp_df, 0.3, ser_int, 2, 0.5)

        self.assertEqual(log_lik.get_overdispersion(), 0.5)

    def test_n_parameters(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        log_lik = bp.LocImpNegBinBranchProLogLik(
            local_df, imp_df, 0.3, ser_int, 3, 0.5)
        log_lik1 = bp.LocImpNegBinBranchProLogLik(
            local_df, imp_df, 0.3, ser_int, 3, 0.5, False)

        self.assertEqual(log_lik.n_parameters(), 3)
        self.assertEqual(log_lik1.n_parameters(), 2)

    def test_get_serial_intervals(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]

        log_lik = bp.LocImpNegBinBranchProLogLik(
            local_df, imp_df, 0.3, ser_int, 3, 0.5)
        npt.assert_array_equal(
            log_lik.get_serial_intervals(), np.array([1, 2]))

    def test_set_serial_intervals(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        ser_int = [1, 2]
        new_ser_int = [1, 2, 1]
        wrong_ser_int = (1)

        log_lik = bp.LocImpNegBinBranchProLogLik(
            local_df, imp_df, 0.3, ser_int, 3, 0.5)
        log_lik.set_serial_intervals(new_ser_int)

        npt.assert_array_equal(
            log_lik.get_serial_intervals(), np.array([1, 2, 1]))

        with self.assertRaises(ValueError):
            log_lik.set_serial_intervals(wrong_ser_int)

    def test__call__(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        log_lik = bp.LocImpNegBinBranchProLogLik(
            local_df, imp_df, 0.3, ser_int, 3, 0.5)
        log_lik1 = bp.LocImpNegBinBranchProLogLik(
            local_df, imp_df, 0.3, ser_int, 3, 0.5, False)

        self.assertEqual(log_lik([1, 1, 0.5]), 0)
        self.assertEqual(log_lik1([1, 1]), 0)

    def test_evaluateS1(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        log_lik = bp.LocImpNegBinBranchProLogLik(
            local_df, imp_df, 0.3, ser_int, 3, 0.5)
        log_lik1 = bp.LocImpNegBinBranchProLogLik(
            local_df, imp_df, 0.3, ser_int, 3, 0.5, False)

        self.assertEqual(log_lik.evaluateS1([1, 1, 0.5]), (0, [0, 0, 0]))
        self.assertEqual(log_lik1.evaluateS1([1, 1]), (0, [0, 0]))

#
# Test LocImpNegBinBranchProLogPosterior Class
#


class TestLocImpNegBinBranchProLogPosteriorClass(unittest.TestCase):
    """
    Test the 'LocImpNegBinBranchProLogPosterior' class.
    """
    def test__init__(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        bp.LocImpNegBinBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int, 2, 0.5, 1, 0.2)
        bp.LocImpNegBinBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int, 2, 0.5, 1, 0.2, 1, False)

    def test_return_loglikelihood(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        inference = bp.LocImpNegBinBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int, 2, 0.5, 1, 0.2)
        inference1 = bp.LocImpNegBinBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int, 2, 0.5, 1, 0.2, 1, False)

        self.assertEqual(
            inference.return_loglikelihood([1, 1, 1, 0.5]),
            inference.ll([1, 1, 1, 0.5]))
        self.assertEqual(
            inference1.return_loglikelihood([1, 1, 1]),
            inference1.ll([1, 1, 1]))

    def test_return_logprior(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        inference = bp.LocImpNegBinBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int, 2, 0.5, 1, 0.2)
        inference1 = bp.LocImpNegBinBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int, 2, 0.5, 1, 0.2, 1, False)

        self.assertEqual(
            inference.return_logprior([1, 1, 1, 0.5]),
            inference.lprior([1, 1, 1, 0.5]))
        self.assertEqual(
            inference1.return_logprior([1, 1, 1]),
            inference1.lprior([1, 1, 1]))

    def test_return_logposterior(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int = [1, 2]

        inference = bp.LocImpNegBinBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int, 2, 0.5, 1, 0.2)
        inference1 = bp.LocImpNegBinBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int, 2, 0.5, 1, 0.2, 1, False)

        self.assertEqual(
            inference.return_logposterior([1, 1, 1, 0.5]),
            inference._log_posterior([1, 1, 1, 0.5]))
        self.assertEqual(
            inference1.return_logposterior([1, 1, 1]),
            inference1._log_posterior([1, 1, 1]))

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

        inference1 = bp.LocImpNegBinBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int1, 2, 0.5, 1, 0.2)

        samples = inference1.run_inference(num_iter=100)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (100, 4))

        inference2 = bp.LocImpNegBinBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int2, 2, 0.5, 1, 0.2)

        samples = inference2.run_inference(num_iter=100)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (100, 4))

        inference3 = bp.LocImpNegBinBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int1, 2, 0.5, 1, 0.2, 1, False)

        samples = inference3.run_inference(num_iter=100)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (100, 3))

        inference4 = bp.LocImpNegBinBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int2, 2, 0.5, 1, 0.2, 1, False)

        samples = inference4.run_inference(num_iter=100)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (100, 3))

    def test_run_optimisation(self):
        local_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })
        imp_df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [0, 0, 0, 0, 0]
        })
        ser_int1 = [1, 2, 1, 0, 0, 0]
        ser_int2 = [1, 2]

        optimisation1 = bp.LocImpNegBinBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int1, 2, 0.5, 1, 0.2)

        found, val = optimisation1.run_optimisation()

        self.assertTrue(isinstance(val, (int, float)))
        self.assertEqual(len(found), 4)

        optimisation2 = bp.LocImpNegBinBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int2, 2, 0.5, 1, 0.2)

        found, val = optimisation2.run_optimisation()

        self.assertTrue(isinstance(val, (int, float)))
        self.assertEqual(len(found), 4)

        optimisation3 = bp.LocImpNegBinBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int1, 2, 0.5, 1, 0.2, 1, False)

        found, val = optimisation3.run_optimisation()

        self.assertTrue(isinstance(val, (int, float)))
        self.assertEqual(len(found), 3)

        optimisation4 = bp.LocImpNegBinBranchProLogPosterior(
            local_df, imp_df, 0.3, ser_int2, 2, 0.5, 1, 0.2, 1, False)

        found, val = optimisation4.run_optimisation()

        self.assertTrue(isinstance(val, (int, float)))
        self.assertEqual(len(found), 3)
