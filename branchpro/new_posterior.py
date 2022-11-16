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


class PoissonBranchProLogPrior(pints.LogPrior):
    """PoissonBranchProLogPrior Class:
    Controller class to construct the log-prior needed for optimisation or
    inference in a PINTS framework.

    Parameters
    ----------
    inc_data
        (pandas Dataframe) Dataframe of the numbers of new cases by time unit
        (usually days).
        Data stored in columns of with one for time and one for incidence
        number, respectively.
    alpha
        the shape parameter of the Gamma distribution of the prior.
    beta
        the rate parameter of the Gamma distribution of the prior.
    time_key
        label key given to the temporal data in the inc_data dataframe.
    inc_key
        label key given to the incidental data in the inc_data dataframe.

    """
    def __init__(self, inc_data, tau, alpha, beta,
                 time_key='Time', inc_key='Incidence Number'):

        if not issubclass(type(inc_data), pd.DataFrame):
            raise TypeError('Incidence data has to be a dataframe')

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

        # Sliding window kength
        self._tau = tau

        # Prior parameters
        self.prior_parameters = (alpha, beta)

    def n_parameters(self):
        """
        Returns number of parameters for log-likelihood object.

        Returns
        -------
        int
            Number of parameters for log-likelihood object.

        """
        return np.shape(self.cases_data)[0] - self._tau - 1

    def evaluateS1(self, x):
        """
        """
        alpha, beta = self.prior_parameters

        return pints.GammaLogPrior(alpha, beta).evaluateS1(x)

    def __call__(self, x):
        """
        Evaluates the log-prior in a PINTS framework.

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
        alpha, beta = self.prior_parameters

        log_prior = 0

        # Prior contribution of R_t
        for _ in x:
            log_prior += pints.GammaLogPrior(alpha, beta)([_])

        return log_prior

    def mean(self):
        """ See :meth:`LogPrior.mean()`. """
        alpha, beta = self.prior_parameters

        log_prior_mean = []

        for _ in range(self.n_parameters()):
            log_prior_mean.append(pints.GammaLogPrior(alpha, beta).mean())

        return log_prior_mean


class PoissonBranchProLogPosterior(object):
    """PoissonBranchProLogPosterior Class:
    Controller class for the optimisation or inference of parameters of the
    Roche model in a PINTS framework.

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
        logprior = PoissonBranchProLogPrior(
            inc_data, tau, alpha, beta, time_key, inc_key)

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
        Runs the parameter inference routine for the Roche model.

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
