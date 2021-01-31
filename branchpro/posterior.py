#
# BranchProPosterior Class
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import numpy as np
import scipy.stats
import pandas as pd


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

        try:
            float(next(iter(daily_serial_interval)))
        except (TypeError, StopIteration):
            raise TypeError(
                'Daily Serial Interval distribution must be iterable')
        except ValueError:
            raise TypeError('Daily Serial Interval distribution must contain '
                            'numeric values')

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
        self.prior_parameters = (alpha, beta)

    def get_serial_intervals(self):
        """
        Returns serial intervals for the model.

        """
        # Reverse inverting of order of serial intervals
        return self._serial_interval[::-1]

    def _infectious_individuals(self, cases_data, t):
        """
        Computes expected number of new cases at time t, using previous
        incidences and serial intervals.

        Parameters
        ----------
        cases_data
            (pandas Dataframe) contains numbers of cases by time unit (usually
            days).
            Data stored in columns of with one for time and one for incidence
            number, respectively.
        t
            evaluation time
        """
        if t > len(self._serial_interval):
            start_date = t - len(self._serial_interval)
            eff_num = np.sum(
                cases_data[start_date:t] * self._serial_interval)
            return eff_num

        eff_num = np.sum(cases_data[:t] * self._serial_interval[-t:])
        return eff_num

    def _infectives_in_tau(self, cases_data, start, end):
        """
        Sum total number of infectives in tau window.

        Parameters
        ----------
        cases_data
            (pandas Dataframe) contains numbers of cases by time unit (usually
            days).
            Data stored in columns of with one for time and one for incidence
            number, respectively.
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

        for time in range(time_init_inf_r, total_time+1):
            # get cases in tau window
            start_window = time - tau
            end_window = time + 1

            # compute shape parameter of the posterior over time
            shape.append(
                alpha + np.sum(
                    self.cases_data[start_window:end_window]))

            # compute rate parameter of the posterior over time

            rate.append(beta + self._infectives_in_tau(
                self.cases_data, start_window, end_window))

        # compute the mean of the Gamma-shaped posterior over time
        mean = np.divide(shape, rate)

        # compute the Gamma-shaped posterior distribution
        post_dist = scipy.stats.gamma(shape, scale=1/np.array(rate))

        self.inference_times = list(range(time_init_inf_r, total_time+1))
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
                'Lower bound CI': post_dist_interval[0],
                'Upper bound CI': post_dist_interval[1],
                'Central Probability': central_prob
            }
        )

        return intervals_df


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
        R_{t}^{\text(imported)} = (1 + \epsilon)R_{t}^{\text(local)}

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

        if not isinstance(epsilon, (int, float)):
            raise TypeError('Value of epsilon must be integer or float.')
        if epsilon < -1:
            raise ValueError('Epsilon needs to be greater or equal to -1.')

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
        self.epsilon = self.set_epsilon(epsilon)

    def set_epsilon(self, new_epsilon):
        """
        Updates proportionality constant of the R number for imported cases
        with respect to its analog for local ones.

        Parameters
        ----------
        new_epsilon
            new value of constant of proportionality.

        """
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

        for time in range(time_init_inf_r, total_time+1):
            # get cases in tau window
            start_window = time - tau
            end_window = time + 1

            # compute shape parameter of the posterior over time
            shape.append(
                alpha + np.sum(
                    self.cases_data[start_window:end_window]))

            # compute rate parameter of the posterior over time

            rate.append(beta + self._infectives_in_tau(
                self.cases_data, start_window, end_window) +
                (1 + self.epsilon) * self._infectives_in_tau(
                self.imp_cases_data, start_window, end_window))

        # compute the mean of the Gamma-shaped posterior over time
        mean = np.divide(shape, rate)

        # compute the Gamma-shaped posterior distribution
        post_dist = scipy.stats.gamma(shape, scale=1/np.array(rate))

        self.inference_times = list(range(time_init_inf_r, total_time+1))
        self.inference_estimates = mean
        self.inference_posterior = post_dist
