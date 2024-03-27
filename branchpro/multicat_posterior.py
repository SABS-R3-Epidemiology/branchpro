#
# MultiCatPoissonBranchProPosterior Class
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import warnings
import numpy as np
import pandas as pd
from scipy.special import loggamma

import pints
import branchpro as bp


class MultiCatPoissonBranchProLogLik(bp.PoissonBranchProLogLik):
    """MultiCatPoissonBranchProLogLik Class:
    Controller class to construct the log-likelihood needed for optimisation or
    inference in a PINTS framework of Poisson branching process with multiple
    population categories.

    Parameters
    ----------
    inc_data
        (pandas Dataframe) Dataframe of the categorical numbers of new cases
        by time unit (usually days).
        Data stored in columns of with one for time and one for incidence
        number vector per categories, respectively.
    daily_serial_interval
        (list) Unnormalised probability distribution of that the recipient
        first displays symptoms s days after the infector first displays
        symptoms for each category.
    num_cat
        (int) Number of categories in which the population is split.
    contact_matrix
        (array) Matrix of contacts between the different categories in which
        the population is split.
    transm
        (list) List of overall reductions in transmissibility per category.
    tau
        (numeric) size sliding time window over which the reproduction number
        is estimated.
    imported_inc_data
        (pandas Dataframe) contains numbers of categorical imported new cases
        by time unit (usually days).
        Data stored in columns of with one for time and one for incidence
        number vector per categories, respectively.
    epsilon
        (numeric) Proportionality constant of the R number for imported cases
        with respect to its analog for local ones.
    time_key
        label key given to the temporal data in the inc_data dataframe.
    inc_key
        label key given to the incidental data in the inc_data dataframe.
    multipleSI
        (boolean) Different serial intervals used for categories.

    """
    def __init__(self, inc_data, daily_serial_interval, num_cat,
                 contact_matrix, transm, tau,
                 imported_inc_data=None, epsilon=None,
                 time_key='Time', inc_key='Incidence Number',
                 multipleSI=False):

        if not isinstance(num_cat, int):
            raise TypeError('Number of population categories must be integer.')
        if num_cat <= 0:
            raise ValueError('Number of population categories must be > 0.')

        # Local incidence data
        if not issubclass(type(inc_data), pd.DataFrame):
            raise TypeError('Incidence data has to be a dataframe')

        self._check_serial(daily_serial_interval)

        if time_key not in inc_data.columns:
            raise ValueError('No time column with this name in given data')

        for _ in range(num_cat):
            if inc_key+' Cat {}'.format(_+1) not in inc_data.columns:
                raise ValueError(
                    'No incidence column with this name in given data')

        data_times = inc_data[time_key]

        # Pad with zeros the time points where we have no information on
        # the number of incidences
        padded_inc_data = inc_data.set_index(time_key).reindex(
            range(
                min(data_times), max(data_times)+1)
                ).fillna(0).reset_index()

        # Imported cases data
        if imported_inc_data is not None:
            if not issubclass(type(imported_inc_data), pd.DataFrame):
                raise TypeError(
                    'Imported incidence data has to be a dataframe')

            if time_key not in imported_inc_data.columns:
                raise ValueError('No time column with this name in given data')

            for _ in range(num_cat):
                if inc_key+' Cat {}'.format(
                        _+1) not in imported_inc_data.columns:
                    raise ValueError(
                        'No imported incidence column with this name in ' +
                        'given data')

            data_times = inc_data[time_key]

            # Pad with zeros the time points where we have no information on
            # the number of imported incidences
            padded_imp_inc_data = imported_inc_data.set_index(
                time_key).reindex(range(
                    min(data_times), max(data_times)+1)
                    ).fillna(0).reset_index()
        else:
            padded_imp_inc_data = pd.DataFrame(
                0, columns=padded_inc_data.columns,
                index=padded_inc_data.index)

        # Set the prerequisites for the inference wrapper
        # Model and Incidence data
        self._num_cat = num_cat
        self.cases_labels = list(padded_inc_data[
            [time_key] +
            [inc_key+' Cat {}'.format(_+1) for _ in range(num_cat)]].columns)

        self.cases_data = padded_inc_data[
            [inc_key+' Cat {}'.format(_+1) for _ in range(num_cat)]].to_numpy()
        self.cases_times = padded_inc_data[time_key]

        self.imp_cases_labels = list(padded_imp_inc_data[
            [time_key] +
            [inc_key+' Cat {}'.format(_+1) for _ in range(num_cat)]].columns)

        self.imp_cases_data = padded_imp_inc_data[
            [inc_key+' Cat {}'.format(_+1) for _ in range(num_cat)]].to_numpy()
        self.imp_cases_times = padded_imp_inc_data[time_key]

        self._contact_matrix = np.asarray(contact_matrix)
        self._transm = np.asarray(transm)

        if multipleSI is False:
            if np.asarray(daily_serial_interval).ndim != 1:
                raise ValueError(
                    'Serial interval values storage format must be\
                    1-dimensional')
            if np.sum(daily_serial_interval) < 0:
                raise ValueError('Sum of serial interval values must be >= 0.')
            self._serial_interval = np.tile(
                np.asarray(daily_serial_interval)[::-1], (num_cat, 1))
        else:
            if np.asarray(daily_serial_interval).ndim != 2:
                raise ValueError(
                    'Serial interval values storage format must be\
                    2-dimensional')
            if np.asarray(daily_serial_interval).shape[0] != num_cat:
                raise ValueError(
                    'Serial interval values storage format must match\
                    number of categories')
            for _ in range(num_cat):
                if np.sum(daily_serial_interval[_, :]) < 0:
                    raise ValueError(
                        'Sum of serial interval values must be >= 0.')
            self._serial_interval = np.asarray(daily_serial_interval)[:, ::-1]

        self._normalizing_const = np.sum(self._serial_interval, axis=1)

        # Sliding window length
        self._tau = tau

        # Set proportionality constant
        if epsilon is not None:
            self.set_epsilon(epsilon)
        else:
            self.set_epsilon(0)

        # Precompute quantities for the log-likelihood computation and its
        # derivatives
        self._log_lik_precomp()

    def n_parameters(self):
        """
        Returns number of parameters for log-likelihood object.

        Returns
        -------
        int
            Number of parameters for log-likelihood object.

        """
        return np.shape(self.cases_data)[0] - self._tau - 1

    def _infectious_individuals(self, cases_data, t):
        """
        Computes expected number of new cases at time t, using previous
        incidences and serial intervals.

        Parameters
        ----------
        cases_data
            (1D numpy array) contains numbers of cases occuring in each time
            unit (usually days) including zeros.
        t
            evaluation time
        """
        start_date = t - self._serial_interval.shape[1] - 1

        if t > self._serial_interval.shape[1]:
            eff_num = np.divide(np.diag(
                np.matmul(
                    self._serial_interval,
                    cases_data[start_date:(t-1), :])),
                self._normalizing_const)
            return eff_num

        eff_num = np.divide(np.diag(
            np.matmul(
                self._serial_interval[:, -start_date:],
                cases_data[:(t-1), :])),
            self._normalizing_const)

        return eff_num

    def _infectives_in_tau(self, cases_data, start, end):
        """
        Get number of infectives in tau window.

        Parameters
        ----------
        cases_data
            (1D numpy array) contains numbers of cases occuring in each time
            unit (usually days) including zeros.
        start
            start time of the time window in which to calculate effective
            number of infectives.
        end
            end time of the time window in which to calculate effective number
            of infectives.
        """
        num = []
        for time in range(start, end):
            num.append([np.dot(
                            np.matmul(
                                self._contact_matrix,
                                np.diag(self._transm)
                            )[i, :],
                            self._infectious_individuals(cases_data, time)
                        ) for i in range(self._num_cat)])
        return num

    def _compute_log_likelihood(self, r_profile):
        """
        Computes the log-likelihood evaluated a given choice of
        R numbers timeline for the branching process model.

        Parameters
        ----------
        r_profile : list
            Time-dependent R numbers trajectory per category for which
            the log-likelihood is computed for.

        Returns
        -------
        float
            Value of the log-likelihood evaluated at the given choice of
            R numbers timeline.
        """
        total_time = self.cases_times.max() - self.cases_times.min() + 1
        time_init_inf_r = self._tau + 1

        Ll = 0

        for _, time in enumerate(range(time_init_inf_r+1, total_time+1)):
            Ll += np.log(r_profile[_]) * np.sum(self.slice_cases[_])
            Ll += np.sum(
                np.multiply(self.slice_cases[_], self.log_tau_window[_]))
            Ll += - r_profile[_] * self.sum_tau_window[_]
            Ll += - np.sum(self.ll_normalizing[_])

        return Ll

    def _log_lik_precomp(self):
        """
        Precompute quantities for the log-likelihood computation and its
        derivatives.
        """
        self.slice_cases = []
        self.ll_normalizing = []
        self.tau_window = []
        self.tau_window_imp = []
        self.log_tau_window = []
        self.sum_tau_window = []

        total_time = self.cases_times.max() - self.cases_times.min() + 1
        time_init_inf_r = self._tau + 1

        for _, time in enumerate(range(time_init_inf_r+1, total_time+1)):
            # get cases in tau window
            start_window = time - self._tau
            end_window = time + 1

            slice_cases = self.cases_data[(start_window-1):(end_window-1), :]
            self.slice_cases.append(slice_cases)

            self.ll_normalizing.append(loggamma(slice_cases + 1))

            try:
                # try to shift the window by 1 time point
                tau_window = (tau_window[1:] +  # noqa
                              [self._infectious_individuals(self.cases_data,
                                                            end_window-1)])
                tau_window_imp = (tau_window_imp[1:] +  # noqa
                                  [self._infectious_individuals(
                                    self.imp_cases_data,
                                    end_window-1)])

            except UnboundLocalError:
                # First iteration, so set up the sliding window
                tau_window = self._infectives_in_tau(
                    self.cases_data, start_window, end_window)
                tau_window_imp = self._infectives_in_tau(
                    self.imp_cases_data, start_window, end_window)

            self.tau_window.append(tau_window)
            self.tau_window_imp.append(tau_window_imp)

            log_tau_window = np.zeros_like(tau_window)
            for tv, tau_val in enumerate(tau_window):
                for _ in range(self._num_cat):
                    if (tau_val[_] == 0) and (tau_window_imp[tv][_] == 0):
                        log_tau_window[tv, _] = 0
                    else:
                        log_tau_window[tv, _] = np.log(
                            tau_window[tv][_] +
                            self.epsilon * tau_window_imp[tv][_])

            self.log_tau_window.append(log_tau_window)

            self.sum_tau_window.append(
                np.sum(tau_window) + self.epsilon * np.sum(tau_window_imp))

    def _compute_derivative_log_likelihood(self, r_profile):
        """
        Computes the R-number-dependent derivatives of the
        model log-likelihood evaluated a given choice of
        R numbers timeline.

        Parameters
        ----------
        r_profile : list
            Time-dependent R numbers trajectory per category for which
            the log-likelihood is computed for.

        Returns
        -------
        list
            List of the  R-number-dependent derivatives the log-likelihood
            evaluated at the given choice of R numbers timeline.
        """
        total_time = self.cases_times.max() - self.cases_times.min() + 1
        time_init_inf_r = self._tau + 1

        dLl = []

        for _, time in enumerate(range(time_init_inf_r+1, total_time+1)):
            dLl.append(
                (1/r_profile[_]) * np.sum(self.slice_cases[_]) -
                self.sum_tau_window[_])

        return dLl

    def evaluateS1(self, x):
        # Compute log-likelihood
        try:
            Ll = self._compute_log_likelihood(x)

            # Compute derivatives of the log-likelihood
            dLl = self._compute_derivative_log_likelihood(x)

            return Ll, dLl

        except ValueError:  # pragma: no cover
            warnings.warn('RuntimeWarning: for x={}, the likelihood \
                returned -infinity.'.format(x))
            return -np.inf, [-np.inf] * self.n_parameters()

    def __call__(self, x):
        """
        Evaluates the log-likelihood in a PINTS framework.

        Parameters
        ----------
        x : list
            List of free parameters used for computing the log-likelihood.

        Returns
        -------
        float
            Value of the log-likelihood at the given point in the free
            parameter space.

        """
        try:
            return self._compute_log_likelihood(x)

        except ValueError:  # pragma: no cover
            warnings.warn('RuntimeWarning: for x={}, the likelihood \
                returned -infinity.'.format(x))
            return -np.inf


class MultiCatPoissonBranchProLogPosterior(object):
    """MultiCatPoissonBranchProLogPosterior Class:
    Controller class for the optimisation or inference of parameters of the
    Poisson Branching process model in a PINTS framework.

    Parameters
    ----------
    inc_data
        (pandas Dataframe) Dataframe of the numbers of new cases by time unit
        (usually days).
        Data stored in columns of with one for time and one for incidence
        number, respectively.
    daily_serial_interval
        (list) Unnormalised probability distribution of that the recipient
        first displays symptoms s days after the infector first displays
        symptoms.
    num_cat
        (int) Number of categories in which the population is split.
    contact_matrix
        (array) Matrix of contacts between the different categories in which
        the population is split.
    transm
        (list) List of overall reductions in transmissibility per category.
    tau
        (numeric) Size sliding time window over which the reproduction number
        is estimated.
    alpha
        the shape parameter of the Gamma distribution of the prior.
    beta
        the rate parameter of the Gamma distribution of the prior.
    imported_inc_data
        (pandas Dataframe) contains numbers of imported new cases by time unit
        (usually days).
        Data stored in columns of with one for time and one for incidence
        number, respectively.
    epsilon
        (numeric) Proportionality constant of the R number for imported cases
        with respect to its analog for local ones.
    time_key
        label key given to the temporal data in the inc_data dataframe.
    inc_key
        label key given to the incidental data in the inc_data dataframe.

    """
    def __init__(self, inc_data, daily_serial_interval, num_cat,
                 contact_matrix, transm, tau, alpha, beta,
                 imported_inc_data=None, epsilon=None,
                 time_key='Time', inc_key='Incidence Number'):
        super(MultiCatPoissonBranchProLogPosterior, self).__init__()

        loglikelihood = MultiCatPoissonBranchProLogLik(
            inc_data, daily_serial_interval, num_cat,
            contact_matrix, transm, tau,
            imported_inc_data, epsilon, time_key, inc_key)

        # Create a prior and compute prior std vector
        logprior = pints.ComposedLogPrior(
            *[pints.GammaLogPrior(alpha, beta) for _ in range(np.shape(
                loglikelihood.cases_data)[0] - loglikelihood._tau - 1)])

        logprior_std = [np.sqrt(alpha) / beta for _ in range(
            np.shape(
                loglikelihood.cases_data)[0] - loglikelihood._tau - 1)]

        self.lprior = logprior
        self.logprior_std = logprior_std
        self.ll = loglikelihood

        # Create a posterior log-likelihood (log(likelihood * prior))
        self._log_posterior = pints.LogPosterior(loglikelihood, logprior)

    def return_loglikelihood(self, x):
        """
        Return the log-likelihood used for the optimisation or inference.

        Parameters
        ----------
        x : list
            List of free parameters used for computing the log-likelihood.

        Returns
        -------
        float
            Value of the log-likelihood at the given point in the free
            parameter space.

        """
        return self.ll(x)

    def return_logprior(self, x):
        """
        Return the log-prior used for the optimisation or inference.

        Parameters
        ----------
        x : list
            List of free parameters used for computing the log-prior.

        Returns
        -------
        float
            Value of the log-prior at the given point in the free
            parameter space.

        """
        return self.lprior(x)

    def return_logposterior(self, x):
        """
        Return the log-posterior used for the optimisation or inference.

        Parameters
        ----------
        x : list
            List of free parameters used for computing the log-posterior.

        Returns
        -------
        float
            Value of the log-posterior at the given point in the free
            parameter space.

        """
        return self._log_posterior(x)

    def run_inference(self, num_iter):
        """
        Runs the parameter inference routine for the Poisson branching process
        model.

        Parameters
        ----------
        num_iter : integer
            Number of iterations the MCMC sampler algorithm is run for.

        Returns
        -------
        numpy.array
            3D-matrix of the proposed parameters for each iteration for
            each of the chains of the MCMC sampler.

        """
        # Starting points arround from prior mean
        x0 = [self.run_optimisation()[0].tolist()]*3
        transformation = pints.RectangularBoundariesTransformation(
            [0] * self.lprior.n_parameters(),
            [200] * self.lprior.n_parameters()
        )

        # Create MCMC routine
        mcmc = pints.MCMCController(
            self._log_posterior, 3, x0, method=pints.NoUTurnMCMC,
            transformation=transformation)
        mcmc.set_max_iterations(num_iter)
        mcmc.set_log_to_screen(True)
        # mcmc.set_parallel(True)

        print('Running...')
        chains = mcmc.run()
        print('Done!')

        param_names = []

        for _ in range(self.lprior.n_parameters()):
            param_names.append('R_t{}'.format(_ + 1 + self.ll._tau))

        # Check convergence and other properties of chains
        results = pints.MCMCSummary(
            chains=chains, time=mcmc.time(),
            parameter_names=param_names)
        print(results)

        return chains

    def run_optimisation(self):
        """
        Runs the initial conditions optimisation routine for the Poisson
        branching process model.

        Returns
        -------
        numpy.array
            Matrix of the optimised parameters at the end of the optimisation
            procedure.
        float
            Value of the log-posterior at the optimised point in the free
            parameter space.

        """
        # Starting points
        x0 = [1.5] * self.lprior.n_parameters()
        transformation = pints.RectangularBoundariesTransformation(
            [0] * self.lprior.n_parameters(),
            [200] * self.lprior.n_parameters()
        )

        # Create optimisation routine
        optimiser = pints.OptimisationController(
            self._log_posterior, x0, sigma0=1,
            method=pints.CMAES,
            transformation=transformation)

        optimiser.set_max_unchanged_iterations(100, 1)

        found_ics, found_posterior_val = optimiser.run()

        print(found_ics, found_posterior_val)

        print("Optimisation phase is finished.")

        return found_ics, found_posterior_val
