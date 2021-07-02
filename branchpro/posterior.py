#
# BranchProPosterior Class
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import copy
import numpy as np
import pandas as pd
import scipy.stats
import scipy.integrate


class BranchProPosterior(object):
    r"""BranchProPosterior Class:
    Class for computing the posterior distribution used for the inference of
    the reproduction numbers of an epidemic in the case of a branching process.

    Choice of prior distribution is the conjugate prior for the likelihood
    (Poisson) of observing given incidence data, hence is a Gamma distribution.
    We express it in the shape-rate configuration, so that the PDF takes the
    form:

    .. math::
        f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}

    Hence, the posterior distribution will be also be Gamma-distributed.

    Parameters
    ----------
    inc_data
        (pandas Dataframe) contains numbers of new cases by time unit (usually
        days).
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

    Notes
    -----
    Always apply method run_inference before calling
    :meth:`BranchProPosterior.get_intervals` to get R behaviour dataframe!
    """

    def __init__(
            self, inc_data, daily_serial_interval, alpha, beta,
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

        self.cases_labels = list(padded_inc_data[[time_key, inc_key]].columns)
        self.cases_data = padded_inc_data[inc_key].to_numpy()
        self.cases_times = padded_inc_data[time_key]
        self._serial_interval = np.asarray(daily_serial_interval)[::-1]
        self._normalizing_const = np.sum(self._serial_interval)
        self.prior_parameters = (alpha, beta)

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
                np.sum(cases_data[start_date:(t-1)] * self._serial_interval) /
                self._normalizing_const)
            return eff_num

        eff_num = (
            np.sum(cases_data[:(t-1)] * self._serial_interval[-(t-1):]) /
            self._normalizing_const)
        return eff_num

    def _infectives_in_tau(self, cases_data, start, end):
        """
        Sum total number of infectives in tau window.

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
        return np.sum(num)

    def run_inference(self, tau):
        """
        Runs the inference of the reproduction numbers based on the entirety
        of the incidence data available.

        First inferred R value is given at the immediate time point after which
        the tau-window of the initial incidences ends.

        Parameters
        ----------
        tau
            size sliding time window over which the reproduction number is
            estimated.
        """
        total_time = self.cases_times.max() - self.cases_times.min() + 1
        alpha, beta = self.prior_parameters
        time_init_inf_r = tau + 1

        shape = []
        rate = []
        mean = []

        for time in range(time_init_inf_r+1, total_time+1):
            # get cases in tau window
            start_window = time - tau
            end_window = time + 1

            # compute shape parameter of the posterior over time
            shape.append(
                alpha + np.sum(
                    self.cases_data[(start_window-1):(end_window-1)]))

            # compute rate parameter of the posterior over time

            rate.append(beta + self._infectives_in_tau(
                self.cases_data, start_window, end_window))

        # compute the mean of the Gamma-shaped posterior over time
        mean = np.divide(shape, rate)

        # compute the Gamma-shaped posterior distribution
        post_dist = scipy.stats.gamma(shape, scale=1/np.array(rate))

        self.inference_times = list(range(
            self.cases_times.min()+1+tau, self.cases_times.max()+1))
        self.inference_estimates = mean
        self.inference_posterior = post_dist

    def get_intervals(self, central_prob):
        """
        Returns a dataframe of the reproduction number posterior mean
        with percentiles over time.

        The lower and upper percentiles are computed from the posterior
        distribution, using the specified central probability to form an
        equal-tailed interval.

        The results are returned in a dataframe with the following columns:
        'Time Points', 'Mean', 'Lower bound CI' and 'Upper bound CI'

        Parameters
        ----------
        central_prob
            level of the computed credible interval of the estimated
            R number values. The interval the central probability.
        """
        # compute bounds of credible interval of level central_prob
        post_dist_interval = self.inference_posterior.interval(central_prob)

        intervals_df = pd.DataFrame(
            {
                'Time Points': self.inference_times,
                'Mean': self.inference_estimates,
                'Median': self.inference_posterior.median(),
                'Lower bound CI': post_dist_interval[0],
                'Upper bound CI': post_dist_interval[1],
                'Central Probability': central_prob
            }
        )

        return intervals_df


#
# BranchProPosteriorMultSI Class
#


class BranchProPosteriorMultSI(BranchProPosterior):
    r"""BranchProPosteriorMultiSI Class:
    Class for computing the posterior distribution used for the inference of
    the reproduction numbers of an epidemic in the case of a branching process
    using mutiple serial intevals. Based on the :class:`BranchProPosterior`.

    Parameters
    ----------
    inc_data
        (pandas Dataframe) contains numbers of new cases by time unit (usually
        days).
        Data stored in columns of with one for time and one for incidence
        number, respectively.
    daily_serial_intervals
        (list of lists) List of unnormalised probability distributions of that
        the recipient first displays symptoms s days after the infector first
        displays symptoms.
    alpha
        the shape parameter of the Gamma distribution of the prior.
    beta
        the rate parameter of the Gamma distribution of the prior.
    time_key
        label key given to the temporal data in the inc_data dataframe.
    inc_key
        label key given to the incidental data in the inc_data dataframe.
    """
    def __init__(
            self, inc_data, daily_serial_intervals, alpha, beta,
            time_key='Time', inc_key='Incidence Number'):

        super().__init__(
            inc_data, daily_serial_intervals[0], alpha, beta, time_key,
            inc_key)

        for si in daily_serial_intervals:
            self._check_serial(si)

        self._serial_intervals = np.flip(
            np.asarray(daily_serial_intervals), axis=1)
        self._normalizing_consts = np.sum(self._serial_intervals, axis=1)

    def get_serial_intervals(self):
        """
        Returns serial intervals for the model.

        """
        # Reverse inverting of order of serial intervals
        return np.flip(self._serial_intervals, axis=1)

    def set_serial_intervals(self, serial_intervals):
        """
        Updates serial intervals for the model.

        Parameters
        ----------
        serial_intervals
            New unnormalised probability distributions of that the recipient
            first displays symptoms s days after the infector first displays
            symptoms.

        """
        for si in serial_intervals:
            if np.asarray(si).ndim != 1:
                raise ValueError(
                    'Chosen times storage format must be 2-dimensional')

        # Invert order of serial intervals for ease in _effective_no_infectives
        self._serial_intervals = np.flip(np.asarray(serial_intervals), axis=1)
        self._normalizing_consts = np.sum(self._serial_intervals, axis=1)

    def run_inference(self, tau):
        """
        Runs the inference of the reproduction numbers based on the entirety
        of the incidence data available.

        First inferred R value is given at the immediate time point after which
        the tau-window of the initial incidences ends.

        Parameters
        ----------
        tau
            size sliding time window over which the reproduction number is
            estimated.
        """
        samples = []  # For saving each gamma posterior (scipy.stats object)

        for nc, si in zip(self._normalizing_consts, self._serial_intervals):
            self._serial_interval = si
            self._normalizing_const = nc
            super().run_inference(tau)
            samples.append(copy.deepcopy(self.inference_posterior))

        self._inference_samples = samples

        self._calculate_posterior_mean()
        self._calculate_posterior_percentiles()

    def _calculate_posterior_percentiles(self):
        """Calculate the posterior inverse CDF.

        To be called after self._inference_samples has been populated.
        """
        samples = self._inference_samples
        N = len(samples)

        # Get the 99.9th percentile of R for each posterior
        max_Rs = [dist.ppf(0.999) for dist in samples]

        # Define a grid of R values from 0 to above the highest of the 99.9
        # percentiles of R
        dR = 0.001
        integration_grid = np.arange(0, 1.1 * np.max(max_Rs), dR)

        # Evaluate the posterior pdf on the grid
        pdf_values = np.zeros((len(integration_grid), len(max_Rs[0])))
        for dist in samples:
            pdf_values += dist.pdf(integration_grid[:, np.newaxis])
        pdf_values *= 1 / N

        # Perform a cumulative integration of the posterior density using the
        # trapezoidal rule to get the cumulative distribution
        cdf = scipy.integrate.cumulative_trapezoid(
            pdf_values, x=integration_grid, axis=0)

        # Add zero for the first element of the CDF values
        cdf = np.vstack((np.zeros(cdf.shape[1]), cdf))

        # Do an interpolation to get the inverse CDF
        inv_cdf = [scipy.interpolate.interp1d(cdf[:, t], integration_grid)
                   for t in range(cdf.shape[1])]

        # Define a function to get the percentiles over time
        def posterior_ppf(p):
            return np.array([ppf(p)[()] for ppf in inv_cdf])

        self.posterior_ppf = posterior_ppf

    def _calculate_posterior_mean(self):
        """Calculate posterior mean.

        To be called after self._inference_samples has been populated.
        """
        self.inference_estimates = \
            np.mean(np.array(
                [dist.mean() for dist in self._inference_samples]), axis=0)

    def get_intervals(self, central_prob):
        """
        Returns a dataframe of the reproduction number posterior mean
        with percentiles over time.

        The lower and upper percentiles are computed from the posterior
        distribution, using the specified central probability to form an
        equal-tailed interval.

        The results are returned in a dataframe with the following columns:
        'Time Points', 'Mean', 'Lower bound CI' and 'Upper bound CI'

        Parameters
        ----------
        central_prob
            level of the computed credible interval of the estimated
            R number values. The interval the central probability.
        """
        lb = (1-central_prob)/2
        ub = (1+central_prob)/2

        intervals_df = pd.DataFrame(
            {
                'Time Points': self.inference_times,
                'Mean': self.inference_estimates,
                'Median': self.posterior_ppf(0.5),
                'Lower bound CI': self.posterior_ppf(lb),
                'Upper bound CI': self.posterior_ppf(ub),
                'Central Probability': central_prob
            }
        )

        return intervals_df

#
# LocImpBranchProPosterior Class
#


class LocImpBranchProPosterior(BranchProPosterior):
    r"""LocImpBranchProPosterior Class:
    Class for computing the posterior distribution used for the inference of
    the reproduction numbers of an epidemic in the case of a branching process
    with local and imported cases.

    Choice of prior distribution is the conjugate prior for the likelihood
    (Poisson) of observing given incidence data, hence is a Gamma distribution.
    We express it in the shape-rate configuration, so that the PDF takes the
    form:

    .. math::
        f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}

    Hence, the posterior distribution will be also be Gamma-distributed.

    We assume that at all times the R number of the imported cases is
    proportional to the R number of the local incidences:

    .. math::
        R_{t}^{\text(imported)} = \epsilon R_{t}^{\text(local)}

    Parameters
    ----------
    inc_data
        (pandas Dataframe) contains numbers of local new cases by time unit
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
        label key given to the temporal data in the inc_data and
        imported_inc_data dataframes.
    inc_key
        label key given to the incidental data in the inc_data and
        imported_inc_data dataframes.

    Notes
    -----
    Always apply method run_inference before calling
    :meth:`BranchProPosterior.get_intervals` to get R behaviour dataframe!
    """
    def __init__(
            self, inc_data, imported_inc_data, epsilon,
            daily_serial_interval, alpha, beta,
            time_key='Time', inc_key='Incidence Number'):

        super().__init__(
            inc_data, daily_serial_interval, alpha, beta, time_key, inc_key)

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

    def run_inference(self, tau):
        """
        Runs the inference of the reproduction numbers based on the entirety
        of the local and imported incidence data available.

        First inferred (local) R value is given at the immediate time point
        after which the tau-window of the initial incidences ends.

        Parameters
        ----------
        tau
            size sliding time window over which the reproduction number is
            estimated.
        """
        total_time = self.cases_times.max() - self.cases_times.min() + 1
        alpha, beta = self.prior_parameters
        time_init_inf_r = tau + 1

        shape = []
        rate = []
        mean = []

        for time in range(time_init_inf_r+1, total_time+1):
            # get cases in tau window
            start_window = time - tau
            end_window = time + 1

            # compute shape parameter of the posterior over time
            shape.append(
                alpha + np.sum(
                    self.cases_data[(start_window-1):(end_window-1)]))

            # compute rate parameter of the posterior over time

            rate.append(beta + self._infectives_in_tau(
                self.cases_data, start_window, end_window) +
                self.epsilon * self._infectives_in_tau(
                self.imp_cases_data, start_window, end_window))

        # compute the mean of the Gamma-shaped posterior over time
        mean = np.divide(shape, rate)

        # compute the Gamma-shaped posterior distribution
        post_dist = scipy.stats.gamma(shape, scale=1/np.array(rate))

        self.inference_times = list(range(
            self.cases_times.min()+1+tau, self.cases_times.max()+1))
        self.inference_estimates = mean
        self.inference_posterior = post_dist


#
# LocImpBranchProPosteriorMultSI
#


class LocImpBranchProPosteriorMultSI(
        BranchProPosteriorMultSI, LocImpBranchProPosterior):
    r"""
    """
    def __init__(
            self, inc_data, imported_inc_data, epsilon,
            daily_serial_intervals, alpha, beta,
            time_key='Time', inc_key='Incidence Number'):

        LocImpBranchProPosterior.__init__(
            self, inc_data, imported_inc_data, epsilon,
            daily_serial_intervals[0], alpha, beta, time_key, inc_key)

        for si in daily_serial_intervals:
            self._check_serial(si)

        self._serial_intervals = np.flip(
            np.asarray(daily_serial_intervals), axis=1)
        self._normalizing_consts = np.sum(self._serial_intervals, axis=1)

    def run_inference(self, tau):
        """
        Runs the inference of the reproduction numbers based on the entirety
        of the incidence data available.

        First inferred R value is given at the immediate time point after which
        the tau-window of the initial incidences ends.

        Parameters
        ----------
        tau
            size sliding time window over which the reproduction number is
            estimated.
        """
        samples = []  # For saving each gamma posterior (scipy.stats object)

        for nc, si in zip(self._normalizing_consts, self._serial_intervals):
            self._serial_interval = si
            self._normalizing_const = nc
            LocImpBranchProPosterior.run_inference(self, tau)
            samples.append(copy.deepcopy(self.inference_posterior))

        self._inference_samples = samples

        self._calculate_posterior_mean()
        self._calculate_posterior_percentiles()
