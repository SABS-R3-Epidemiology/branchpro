#
# PoissonBranchProPosterior Class
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import numpy as np
import pandas as pd
from scipy.special import loggamma, digamma

import pints


class PoissonBranchProLogLik(pints.LogPDF):
    """PoissonBranchProLogLik Class:
    Controller class to construct the log-likelihood needed for optimisation or
    inference in a PINTS framework of Poisson branching process.

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
    tau
        (numeric) size sliding time window over which the reproduction number
        is estimated.
    time_key
        label key given to the temporal data in the inc_data dataframe.
    inc_key
        label key given to the incidental data in the inc_data dataframe.

    """
    def __init__(self, inc_data, daily_serial_interval, tau,
                 time_key='Time', inc_key='Incidence Number'):

        if not issubclass(type(inc_data), pd.DataFrame):
            raise TypeError('Incidence data has to be a dataframe')

        self._check_serial(daily_serial_interval)

        if time_key not in inc_data.columns:
            raise ValueError('No time column with this name in given data')

        if inc_key not in inc_data.columns:
            raise ValueError(
                'No incidence column with this name in given data')

        data_times = inc_data[time_key]

        # Pad with zeros the time points where we have no information on
        # the number of incidences
        padded_inc_data = inc_data.set_index(time_key).reindex(
            range(
                min(data_times), max(data_times)+1)
                ).fillna(0).reset_index()

        # Set the prerequisites for the inference wrapper
        # Model and Incidence data
        self.cases_labels = list(padded_inc_data[[time_key, inc_key]].columns)
        self.cases_data = padded_inc_data[inc_key].to_numpy()
        self.cases_times = padded_inc_data[time_key]
        self._serial_interval = np.asarray(daily_serial_interval)[::-1]
        self._normalizing_const = np.sum(self._serial_interval)

        # Sliding window kength
        self._tau = tau

    def n_parameters(self):
        """
        Returns number of parameters for log-likelihood object.

        Returns
        -------
        int
            Number of parameters for log-likelihood object.

        """
        return np.shape(self.cases_data)[0] - self._tau - 1

    def _check_serial(self, si):
        """
        Checks serial interval is iterable and only contains numeric values.
        """
        try:
            float(next(iter(si)))
        except (TypeError, StopIteration):
            raise TypeError(
                'Daily Serial Interval distributions must be iterable')
        except ValueError:
            raise TypeError('Daily Serial Interval distribution must contain \
                            numeric values')

    def get_serial_intervals(self):
        """
        Returns serial intervals for the model.

        Returns
        -------
        list
            Serial intervals for the model.

        """
        # Reverse inverting of order of serial intervals
        return self._serial_interval[::-1]

    def set_serial_intervals(self, serial_intervals):
        """
        Updates serial intervals for the model.

        Parameters
        ----------
        serial_intervals
            New unnormalised probability distribution of that the recipient
            first displays symptoms s days after the infector first displays
            symptoms.

        """
        if np.asarray(serial_intervals).ndim != 1:
            raise ValueError(
                'Chosen times storage format must be 1-dimensional')

        # Invert order of serial intervals for ease in _effective_no_infectives
        self._serial_interval = np.asarray(serial_intervals)[::-1]
        self._normalizing_const = np.sum(self._serial_interval)

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
        if t > len(self._serial_interval):
            start_date = t - len(self._serial_interval) - 1
            eff_num = (
                (cases_data[start_date:(t-1)] * self._serial_interval).sum() /
                self._normalizing_const)
            return eff_num

        eff_num = (
            (cases_data[:(t-1)] * self._serial_interval[-(t-1):]).sum() /
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
            num += [self._infectious_individuals(cases_data, time)]
        return num

    def _compute_log_likelihood(self, r_profile):
        """
        """
        total_time = self.cases_times.max() - self.cases_times.min() + 1
        time_init_inf_r = self._tau + 1

        Ll = 0

        for _, time in enumerate(range(time_init_inf_r+1, total_time+1)):
            # get cases in tau window
            start_window = time - self._tau
            end_window = time + 1

            slice_cases = self.cases_data[(start_window-1):(end_window-1)]

            try:
                # try to shift the window by 1 time point
                tau_window = (tau_window[1:] +  # noqa
                              [self._infectious_individuals(self.cases_data,
                                                            end_window-1)])

            except UnboundLocalError:
                # First iteration, so set up the sliding window
                tau_window = self._infectives_in_tau(
                    self.cases_data, start_window, end_window)

            Ll += np.log(r_profile[_]) * np.sum(slice_cases)

            log_tau_window = np.zeros_like(tau_window)
            for tv, tau_val in enumerate(tau_window):
                if tau_val == 0:
                    log_tau_window[tv] = 0
                else:
                    log_tau_window[tv] = np.log(tau_window[tv])

            Ll += np.sum(np.multiply(slice_cases, log_tau_window))
            Ll += - r_profile[_] * np.sum(tau_window)

        return Ll

    def _compute_derivative_log_likelihood(self, r_profile):
        """
        """
        total_time = self.cases_times.max() - self.cases_times.min() + 1
        time_init_inf_r = self._tau + 1

        dLl = []

        for _, time in enumerate(range(time_init_inf_r+1, total_time+1)):
            # get cases in tau window
            start_window = time - self._tau
            end_window = time + 1

            try:
                # try to shift the window by 1 time point
                tau_window = (tau_window[1:] +  # noqa
                              [self._infectious_individuals(self.cases_data,
                                                            end_window-1)])
            except UnboundLocalError:
                # First iteration, so set up the sliding window
                tau_window = self._infectives_in_tau(
                    self.cases_data, start_window, end_window)

            dLl.append((1/r_profile[_]) * np.sum(
                    self.cases_data[(start_window-1):(end_window-1)])
                    - sum(tau_window))

        return dLl

    def evaluateS1(self, x):
        """
        """
        # Compute log-likelihood
        try:
            Ll = self._compute_log_likelihood(x)

            # Compute derivatives of the log-likelihood
            dLl = self._compute_derivative_log_likelihood(x)

            return Ll, dLl

        except ValueError:  # pragma: no cover
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
            return -np.inf


class PoissonBranchProLogPosterior(object):
    """PoissonBranchProLogPosterior Class:
    Controller class for the optimisation or inference of parameters of the
    Poisson Branching process model in a PINTS framework.

    Parameters
    ----------
    inc_data
        (pandas Dataframe) Dataframe of the numbers of new cases by time unit
        (usually days).
        Data stored in columns of with one for time and one for incidence
        number, respectively.
    tau
        (numeric) Size sliding time window over which the reproduction number
        is estimated.
    daily_serial_interval
        (list) Unnormalised probability distribution of that the recipient
        first displays symptoms s days after the infector first displays
        symptoms.
    alpha
        the shape parameter of the Gamma distribution of the prior.
    beta
        the rate parameter of the Gamma distribution of the prior.
    time_key
        label key given to the temporal data in the inc_data dataframe.
    inc_key
        label key given to the incidental data in the inc_data dataframe.

    """
    def __init__(self, inc_data, daily_serial_interval, tau, alpha, beta,
                 time_key='Time', inc_key='Incidence Number'):
        super(PoissonBranchProLogPosterior, self).__init__()

        loglikelihood = PoissonBranchProLogLik(
            inc_data, daily_serial_interval, tau, time_key, inc_key)

        # Create a prior
        logprior = pints.ComposedLogPrior(
            *[pints.GammaLogPrior(alpha, beta) for _ in range(np.shape(
                loglikelihood.cases_data)[0] - loglikelihood._tau - 1)])

        self.lprior = logprior
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
        # Starting points using optimisation object
        x0 = [self.lprior.mean()]*3
        transformation = pints.RectangularBoundariesTransformation(
            [0] * self.lprior.n_parameters(),
            [20] * self.lprior.n_parameters()
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
            [20] * self.lprior.n_parameters()
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

#
# LocImpPoissonBranchProPosterior Class
#


class LocImpPoissonBranchProLogLik(PoissonBranchProLogLik):
    r"""LocImpPoissonBranchProLogLik Class:
    Controller class to construct the log-likelihood needed for optimisation or
    inference in a PINTS framework of Poisson branching process with local and
    imported cases.

    We assume that at all times the R number of the imported cases is
    proportional to the R number of the local incidences:

    .. math::
        R_{t}^{\text(imported)} = \epsilon R_{t}^{\text(local)}

    Parameters
    ----------
    inc_data
        (pandas Dataframe) Dataframe of the numbers of new cases by time unit
        (usually days).
        Data stored in columns of with one for time and one for incidence
        number, respectively.
    imported_inc_data
        (pandas Dataframe) contains numbers of imported new cases by time unit
        (usually days).
        Data stored in columns of with one for time and one for incidence
        number, respectively.
    epsilon
        (numeric) Proportionality constant of the R number for imported cases
        with respect to its analog for local ones.
    daily_serial_interval
        (list) Unnormalised probability distribution of that the recipient
        first displays symptoms s days after the infector first displays
        symptoms.
    tau
        (numeric) Size sliding time window over which the reproduction number
        is estimated.
    time_key
        label key given to the temporal data in the inc_data dataframe.
    inc_key
        label key given to the incidental data in the inc_data dataframe.

    """
    def __init__(self, inc_data, imported_inc_data, epsilon,
                 daily_serial_interval, tau,
                 time_key='Time', inc_key='Incidence Number'):

        super().__init__(
            inc_data, daily_serial_interval, tau, time_key, inc_key)

        if not issubclass(type(imported_inc_data), pd.DataFrame):
            raise TypeError('Imported incidence data has to be a dataframe')

        if time_key not in imported_inc_data.columns:
            raise ValueError('No time column with this name in given data')

        if inc_key not in imported_inc_data.columns:
            raise ValueError(
                'No imported incidence column with this name in given data')

        data_times = inc_data[time_key]

        # Pad with zeros the time points where we have no information on
        # the number of imported incidences
        padded_imp_inc_data = imported_inc_data.set_index(time_key).reindex(
            range(
                min(data_times), max(data_times)+1)
                ).fillna(0).reset_index()

        self.imp_cases_labels = list(
            padded_imp_inc_data[[time_key, inc_key]].columns)
        self.imp_cases_data = padded_imp_inc_data[inc_key].to_numpy()
        self.imp_cases_times = padded_imp_inc_data[time_key]
        self.set_epsilon(epsilon)

    def set_epsilon(self, new_epsilon):
        """
        Updates proportionality constant of the R number for imported cases
        with respect to its analog for local ones.

        Parameters
        ----------
        new_epsilon
            new value of constant of proportionality.

        """
        if not isinstance(new_epsilon, (int, float)):
            raise TypeError('Value of epsilon must be integer or float.')
        if new_epsilon < 0:
            raise ValueError('Epsilon needs to be greater or equal to 0.')

        self.epsilon = new_epsilon

    def n_parameters(self):
        """
        Returns number of parameters for log-likelihood object.

        Returns
        -------
        int
            Number of parameters for log-likelihood object.

        """
        return np.shape(self.cases_data)[0] - self._tau - 1

    def _compute_log_likelihood(self, r_profile):
        """
        """
        total_time = self.cases_times.max() - self.cases_times.min() + 1
        time_init_inf_r = self._tau + 1

        Ll = 0

        for _, time in enumerate(range(time_init_inf_r+1, total_time+1)):
            # get cases in tau window
            start_window = time - self._tau
            end_window = time + 1

            slice_cases = self.cases_data[(start_window-1):(end_window-1)]

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

            Ll += np.log(r_profile[_]) * np.sum(slice_cases)

            log_tau_window = np.zeros_like(tau_window)
            for tv, tau_val in enumerate(tau_window):
                if tau_val == 0:
                    log_tau_window[tv] = 0
                else:
                    log_tau_window[tv] = np.log(
                        tau_window[tv] + self.epsilon * tau_window_imp[tv])

            Ll += np.sum(np.multiply(slice_cases, log_tau_window))
            Ll += - r_profile[_] * (
                np.sum(tau_window) + self.epsilon * np.sum(tau_window_imp))

        return Ll

    def _compute_derivative_log_likelihood(self, r_profile):
        """
        """
        total_time = self.cases_times.max() - self.cases_times.min() + 1
        time_init_inf_r = self._tau + 1

        dLl = []

        for _, time in enumerate(range(time_init_inf_r+1, total_time+1)):
            # get cases in tau window
            start_window = time - self._tau
            end_window = time + 1

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

            dLl.append((1/r_profile[_]) * np.sum(
                    self.cases_data[(start_window-1):(end_window-1)])
                    - sum(tau_window) - self.epsilon * sum(tau_window_imp))

        return dLl


class LocImpPoissonBranchProLogPosterior(PoissonBranchProLogPosterior):
    r"""LocImpPoissonBranchProLogPosterior Class:
    Controller class for the optimisation or inference of parameters of the
    Poisson Branching process model with local and imported cases in a PINTS
    framework.

    We assume that at all times the R number of the imported cases is
    proportional to the R number of the local incidences:

    .. math::
        R_{t}^{\text(imported)} = \epsilon R_{t}^{\text(local)}

    Parameters
    ----------
    inc_data
        (pandas Dataframe) Dataframe of the numbers of new cases by time unit
        (usually days).
        Data stored in columns of with one for time and one for incidence
        number, respectively.
    imported_inc_data
        (pandas Dataframe) contains numbers of imported new cases by time unit
        (usually days).
        Data stored in columns of with one for time and one for incidence
        number, respectively.
    epsilon
        (numeric) Proportionality constant of the R number for imported cases
        with respect to its analog for local ones.
    daily_serial_interval
        (list) Unnormalised probability distribution of that the recipient
        first displays symptoms s days after the infector first displays
        symptoms.
    alpha
        the shape parameter of the Gamma distribution of the prior.
    beta
        the rate parameter of the Gamma distribution of the prior.
    time_key
        label key given to the temporal data in the inc_data dataframe.
    inc_key
        label key given to the incidental data in the inc_data dataframe.

    """
    def __init__(self, inc_data, imported_inc_data, epsilon,
                 daily_serial_interval, tau, alpha, beta,
                 time_key='Time', inc_key='Incidence Number'):

        PoissonBranchProLogPosterior.__init__(
            self, inc_data, daily_serial_interval, tau, alpha, beta,
            time_key, inc_key)

        loglikelihood = LocImpPoissonBranchProLogLik(
            inc_data, imported_inc_data, epsilon, daily_serial_interval,
            tau, time_key, inc_key)

        # Create a prior
        logprior = pints.ComposedLogPrior(
            *[pints.GammaLogPrior(alpha, beta) for _ in range(np.shape(
                loglikelihood.cases_data)[0] - loglikelihood._tau - 1)])

        self.lprior = logprior
        self.ll = loglikelihood

        # Create a posterior log-likelihood (log(likelihood * prior))
        self._log_posterior = pints.LogPosterior(loglikelihood, logprior)

#
# NegBinBranchProPosterior Class
#


class NegBinBranchProLogLik(PoissonBranchProLogLik):
    """NegBinBranchProLogLik Class:
    Controller class to construct the log-likelihood needed for optimisation or
    inference in a PINTS framework of negative binomial branching process.

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
    tau
        (numeric) size sliding time window over which the reproduction number
        is estimated.
    phi
        (numeric) Value of the overdispersion parameter for the negative
        binomial noise distribution.
    time_key
        label key given to the temporal data in the inc_data dataframe.
    inc_key
        label key given to the incidental data in the inc_data dataframe.

    """
    def __init__(self, inc_data, daily_serial_interval, tau, phi,
                 time_key='Time', inc_key='Incidence Number'):

        PoissonBranchProLogLik.__init__(
            self, inc_data, daily_serial_interval, tau, time_key, inc_key)

        self.set_overdispersion(phi)

    def set_overdispersion(self, phi):
        """
        Updates overdispersion noise parameter for the model.

        Parameters
        ----------
        phi
            New value of the overdispersion parameter for the negative
            binomial noise distribution.

        """
        if not isinstance(phi, (int, float)):
            raise TypeError(
                'Value of overdispersion must be integer or float.')
        if phi <= 0:
            raise ValueError(
                'Value of overdispersion must be must be > 0. For \
                overdispesion = 0, please use `LocImpBranchProModel` class \
                type.')

        self._overdispersion = phi

    def get_overdispersion(self):
        """
        Returns overdispersion noise parameter for the model.

        """
        return self._overdispersion

    def n_parameters(self):
        """
        Returns number of parameters for log-likelihood object.

        Returns
        -------
        int
            Number of parameters for log-likelihood object.

        """
        return np.shape(self.cases_data)[0] - self._tau

    def _compute_log_likelihood(self, r_profile):
        """
        """
        total_time = self.cases_times.max() - self.cases_times.min() + 1
        time_init_inf_r = self._tau + 1

        Ll = 0

        for _, time in enumerate(range(time_init_inf_r+1, total_time+1)):
            # get cases in tau window
            start_window = time - self._tau
            end_window = time + 1

            slice_cases = self.cases_data[(start_window-1):(end_window-1)]

            loggamma_slice_cases_phi = loggamma(
                slice_cases + 1 / self._overdispersion) - loggamma(
                1 / self._overdispersion)

            Ll += np.sum(loggamma_slice_cases_phi - np.log(
                self._overdispersion) / self._overdispersion)

            try:
                # try to shift the window by 1 time point
                tau_window = (tau_window[1:] +  # noqa
                              [self._infectious_individuals(self.cases_data,
                                                            end_window-1)])

            except UnboundLocalError:
                # First iteration, so set up the sliding window
                tau_window = self._infectives_in_tau(
                    self.cases_data, start_window, end_window)

            Ll += np.log(r_profile[_]) * np.sum(slice_cases)

            log_tau_window = np.zeros_like(tau_window)
            log_phi_r_tau_window = np.zeros_like(tau_window)
            for tv, tau_val in enumerate(tau_window):
                if tau_val == 0:
                    log_tau_window[tv] = 0
                else:
                    log_tau_window[tv] = np.log(tau_window[tv])

                log_phi_r_tau_window[tv] = np.log(
                    1 / self._overdispersion +
                    r_profile[_] * tau_window[tv])

            Ll += np.sum(np.multiply(slice_cases, log_tau_window))
            Ll += - np.sum(np.multiply(
                slice_cases + 1 / self._overdispersion, log_phi_r_tau_window))

        return Ll

    def _compute_derivative_log_likelihood(self, r_profile):
        """
        """
        total_time = self.cases_times.max() - self.cases_times.min() + 1
        time_init_inf_r = self._tau + 1

        dLl = []
        dLl_phi = 0

        for _, time in enumerate(range(time_init_inf_r+1, total_time+1)):
            # get cases in tau window
            start_window = time - self._tau
            end_window = time + 1

            slice_cases = self.cases_data[(start_window-1):(end_window-1)]

            try:
                # try to shift the window by 1 time point
                tau_window = (tau_window[1:] +  # noqa
                              [self._infectious_individuals(self.cases_data,
                                                            end_window-1)])
            except UnboundLocalError:
                # First iteration, so set up the sliding window
                tau_window = self._infectives_in_tau(
                    self.cases_data, start_window, end_window)

            inv_phi_r_tau_window = np.zeros_like(tau_window)
            inv_phi_r_tau_window2 = np.zeros_like(tau_window)
            log_phi_r_tau_window = np.zeros_like(tau_window)
            for tv, tau_val in enumerate(tau_window):
                inv_phi_r_tau_window[tv] = \
                    tau_window[tv] * np.reciprocal(
                    1 / self._overdispersion +
                    r_profile[_] * tau_window[tv])
                inv_phi_r_tau_window2[tv] = np.reciprocal(
                    1 / self._overdispersion +
                    r_profile[_] * tau_window[tv])
                log_phi_r_tau_window[tv] = np.log(
                    1 / self._overdispersion +
                    r_profile[_] * tau_window[tv])

            dLl.append(
                (1/r_profile[_]) * np.sum(slice_cases) - np.sum(np.multiply(
                    slice_cases + 1 / self._overdispersion,
                    inv_phi_r_tau_window)))

            dLl_phi += np.sum(
                digamma(slice_cases + 1 / self._overdispersion) -
                digamma(1 / self._overdispersion) + np.log(
                    1 / self._overdispersion) + 1)

            dLl_phi -= np.sum(log_phi_r_tau_window) + np.sum(np.multiply(
                slice_cases + 1 / self._overdispersion, inv_phi_r_tau_window2))

        dLl_phi *= - 1 / (self._overdispersion ** 2)

        dLl.append(dLl_phi)

        return dLl


class NegBinBranchProLogPosterior(PoissonBranchProLogPosterior):
    """NegBinBranchProLogPosterior Class:
    Controller class for the optimisation or inference of parameters of the
    negative binomial branching process model in a PINTS framework.

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
    tau
        (numeric) Size sliding time window over which the reproduction number
        is estimated.
    phi
        (numeric) Value of the overdispersion parameter for the negative
        binomial noise distribution.
    alpha
        the shape parameter of the Gamma distribution of the prior.
    beta
        the rate parameter of the Gamma distribution of the prior.
    lam
        the mean parameter of the Exponential distribution of the prior of
        the overdispersion.
    time_key
        label key given to the temporal data in the inc_data dataframe.
    inc_key
        label key given to the incidental data in the inc_data dataframe.

    """
    def __init__(self, inc_data, daily_serial_interval, tau, phi, alpha, beta,
                 lam=1, time_key='Time', inc_key='Incidence Number'):
        PoissonBranchProLogPosterior.__init__(
            self, inc_data, daily_serial_interval, tau, alpha, beta,
            time_key, inc_key)

        loglikelihood = NegBinBranchProLogLik(
            inc_data, daily_serial_interval, tau, phi, time_key, inc_key)

        # Create a prior
        list_priors = [pints.GammaLogPrior(alpha, beta) for _ in range(
            np.shape(
                loglikelihood.cases_data)[0] - loglikelihood._tau - 1)] + [
                    pints.ExponentialLogPrior(lam)
                ]
        logprior = pints.ComposedLogPrior(*list_priors)

        self.lprior = logprior
        self.ll = loglikelihood

        # Create a posterior log-likelihood (log(likelihood * prior))
        self._log_posterior = pints.LogPosterior(loglikelihood, logprior)

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
        # Starting points using optimisation object
        x0 = [self.lprior.mean()]*3
        transformation = pints.RectangularBoundariesTransformation(
            [0] * self.lprior.n_parameters(),
            [20] * self.lprior.n_parameters()
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

        for _ in range(self.lprior.n_parameters() - 1):
            param_names.append('R_t{}'.format(_ + 1 + self.ll._tau))
        param_names.append('Phi')

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
            [20] * self.lprior.n_parameters()
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

#
# LocImpNegBinBranchProPosterior Class
#


class LocImpNegBinBranchProLogLik(LocImpPoissonBranchProLogLik):
    """LocImpNegBinBranchProLogLik Class:
    Controller class to construct the log-likelihood needed for optimisation or
    inference in a PINTS framework of negative binomial branching process.

    Parameters
    ----------
    inc_data
        (pandas Dataframe) Dataframe of the numbers of new cases by time unit
        (usually days).
        Data stored in columns of with one for time and one for incidence
        number, respectively.
    imported_inc_data
        (pandas Dataframe) contains numbers of imported new cases by time unit
        (usually days).
        Data stored in columns of with one for time and one for incidence
        number, respectively.
    epsilon
        (numeric) Proportionality constant of the R number for imported cases
        with respect to its analog for local ones.
    daily_serial_interval
        (list) Unnormalised probability distribution of that the recipient
        first displays symptoms s days after the infector first displays
        symptoms.
    tau
        (numeric) size sliding time window over which the reproduction number
        is estimated.
    phi
        (numeric) Value of the overdispersion parameter for the negative
        binomial noise distribution.
    time_key
        label key given to the temporal data in the inc_data dataframe.
    inc_key
        label key given to the incidental data in the inc_data dataframe.

    """
    def __init__(self, inc_data, imported_inc_data, epsilon,
                 daily_serial_interval, tau, phi,
                 time_key='Time', inc_key='Incidence Number'):

        LocImpPoissonBranchProLogLik.__init__(
            self, inc_data, imported_inc_data, epsilon,
            daily_serial_interval, tau, time_key, inc_key)

        self.set_overdispersion(phi)

    def set_overdispersion(self, phi):
        """
        Updates overdispersion noise parameter for the model.

        Parameters
        ----------
        phi
            New value of the overdispersion parameter for the negative
            binomial noise distribution.

        """
        if not isinstance(phi, (int, float)):
            raise TypeError(
                'Value of overdispersion must be integer or float.')
        if phi <= 0:
            raise ValueError(
                'Value of overdispersion must be > 0. For \
                overdispesion = 0, please use `LocImpBranchProModel` class \
                type.')

        self._overdispersion = phi

    def get_overdispersion(self):
        """
        Returns overdispersion noise parameter for the model.

        """
        return self._overdispersion

    def n_parameters(self):
        """
        Returns number of parameters for log-likelihood object.

        Returns
        -------
        int
            Number of parameters for log-likelihood object.

        """
        return np.shape(self.cases_data)[0] - self._tau - 1

    def _compute_log_likelihood(self, r_profile):
        """
        """
        total_time = self.cases_times.max() - self.cases_times.min() + 1
        time_init_inf_r = self._tau + 1

        Ll = 0

        for _, time in enumerate(range(time_init_inf_r+1, total_time+1)):
            # get cases in tau window
            start_window = time - self._tau
            end_window = time + 1

            slice_cases = self.cases_data[(start_window-1):(end_window-1)]

            loggamma_slice_cases_phi = loggamma(
                slice_cases + 1 / self._overdispersion) - loggamma(
                1 / self._overdispersion)

            Ll += np.sum(loggamma_slice_cases_phi - np.log(
                self._overdispersion) / self._overdispersion)

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

            Ll += np.log(r_profile[_]) * np.sum(slice_cases)

            log_tau_window = np.zeros_like(tau_window)
            log_phi_r_tau_window = np.zeros_like(tau_window)
            for tv, tau_val in enumerate(tau_window):
                if tau_val == 0:
                    log_tau_window[tv] = 0
                else:
                    log_tau_window[tv] = np.log(
                        tau_window[tv] + self.epsilon * tau_window_imp[tv])

                log_phi_r_tau_window[tv] = np.log(
                    1 / self._overdispersion +
                    r_profile[_] * (
                        tau_window[tv] + self.epsilon * tau_window_imp[tv]
                    ))

            Ll += np.sum(np.multiply(slice_cases, log_tau_window))
            Ll += - np.sum(np.multiply(
                slice_cases + 1 / self._overdispersion, log_phi_r_tau_window))

        return Ll

    def _compute_derivative_log_likelihood(self, r_profile):
        """
        """
        total_time = self.cases_times.max() - self.cases_times.min() + 1
        time_init_inf_r = self._tau + 1

        dLl = []
        dLl_phi = 0

        for _, time in enumerate(range(time_init_inf_r+1, total_time+1)):
            # get cases in tau window
            start_window = time - self._tau
            end_window = time + 1

            slice_cases = self.cases_data[(start_window-1):(end_window-1)]

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

            inv_phi_r_tau_window = np.zeros_like(tau_window)
            inv_phi_r_tau_window2 = np.zeros_like(tau_window)
            log_phi_r_tau_window = np.zeros_like(tau_window)
            for tv, tau_val in enumerate(tau_window):
                inv_phi_r_tau_window[tv] = \
                    (tau_window[tv] + self.epsilon * tau_window_imp[tv]) \
                    * np.reciprocal(
                    1 / self._overdispersion +
                    r_profile[_] * (
                        tau_window[tv] + self.epsilon * tau_window_imp[tv]
                    ))
                inv_phi_r_tau_window2[tv] = np.reciprocal(
                    1 / self._overdispersion +
                    r_profile[_] * (
                        tau_window[tv] + self.epsilon * tau_window_imp[tv]
                    ))
                log_phi_r_tau_window[tv] = np.log(
                    1 / self._overdispersion +
                    r_profile[_] * (
                        tau_window[tv] + self.epsilon * tau_window_imp[tv]
                    ))

            dLl.append(
                (1/r_profile[_]) * np.sum(slice_cases) - np.sum(np.multiply(
                    slice_cases + 1 / self._overdispersion,
                    inv_phi_r_tau_window)))

            dLl_phi += np.sum(
                digamma(slice_cases + 1 / self._overdispersion) -
                digamma(1 / self._overdispersion) + np.log(
                    1 / self._overdispersion) + 1)

            dLl_phi -= np.sum(log_phi_r_tau_window) + np.sum(np.multiply(
                slice_cases + 1 / self._overdispersion, inv_phi_r_tau_window2))

        dLl_phi *= - 1 / (self._overdispersion ** 2)

        # dLl.append(dLl_phi)

        return dLl


class LocImpNegBinBranchProLogPosterior(LocImpPoissonBranchProLogPosterior):
    """LocImpNegBinBranchProLogPosterior Class:
    Controller class for the optimisation or inference of parameters of the
    negative binomial branching process model in a PINTS framework.

    Parameters
    ----------
    inc_data
        (pandas Dataframe) Dataframe of the numbers of new cases by time unit
        (usually days).
        Data stored in columns of with one for time and one for incidence
        number, respectively.
    imported_inc_data
        (pandas Dataframe) contains numbers of imported new cases by time unit
        (usually days).
        Data stored in columns of with one for time and one for incidence
        number, respectively.
    epsilon
        (numeric) Proportionality constant of the R number for imported cases
        with respect to its analog for local ones.
    daily_serial_interval
        (list) Unnormalised probability distribution of that the recipient
        first displays symptoms s days after the infector first displays
        symptoms.
    tau
        (numeric) Size sliding time window over which the reproduction number
        is estimated.
    phi
        (numeric) Value of the overdispersion parameter for the negative
        binomial noise distribution.
    alpha
        the shape parameter of the Gamma distribution of the prior.
    beta
        the rate parameter of the Gamma distribution of the prior.
    lam
        the mean parameter of the Exponential distribution of the prior of
        the overdispersion.
    time_key
        label key given to the temporal data in the inc_data dataframe.
    inc_key
        label key given to the incidental data in the inc_data dataframe.

    """
    def __init__(self, inc_data, imported_inc_data, epsilon,
                 daily_serial_interval, tau, phi, alpha, beta,
                 lam=1, time_key='Time', inc_key='Incidence Number'):
        LocImpPoissonBranchProLogPosterior.__init__(
            self, inc_data, imported_inc_data, epsilon,
            daily_serial_interval, tau, alpha, beta,
            time_key, inc_key)

        loglikelihood = LocImpNegBinBranchProLogLik(
            inc_data, imported_inc_data, epsilon,
            daily_serial_interval, tau, phi, time_key, inc_key)

        # Create a prior
        list_priors = [pints.GammaLogPrior(alpha, beta) for _ in range(
            np.shape(
                loglikelihood.cases_data)[0] - loglikelihood._tau - 1)]  # + [
        #             pints.ExponentialLogPrior(lam)
        #         ]
        logprior = pints.ComposedLogPrior(*list_priors)

        self.lprior = logprior
        self.ll = loglikelihood

        # Create a posterior log-likelihood (log(likelihood * prior))
        self._log_posterior = pints.LogPosterior(loglikelihood, logprior)

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
        # Starting points using optimisation object
        x0 = [self.lprior.mean()]*3
        transformation = pints.RectangularBoundariesTransformation(
            [0] * self.lprior.n_parameters(),
            [20] * self.lprior.n_parameters()
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
        # param_names.append('Phi')

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
            [20] * self.lprior.n_parameters()
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
