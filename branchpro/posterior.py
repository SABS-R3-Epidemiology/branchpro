#
# BranchProPosterior Class
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#
import numpy as np
import math
import scipy.stats
import pandas as pd


class BranchProPosterior(object):
    """BranchProPosterior Class:
    Class for computing the posterior distribution used for the inference of
    the reproduction numbers of an epidemic in the case of a branching process.

    Choice of prior distribution is the conjugate prior for the likelihood
    (Poisson) of observing given incidence data, hence a Gamma distribution.
    We express it in the shape-rate configuration.

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

        if not issubclass(type(daily_serial_interval), list):
            raise TypeError(
                'Daily Serial Interval distribution has to be a list')

        if time_key not in list(inc_data.columns):
            raise ValueError('No column with this name in given data')

        if inc_key not in list(inc_data.columns):
            raise ValueError('No column with this name in given data')

        data_times = inc_data[time_key]

        # Pad with zeros the time points where we have no information on
        # the number of incidences
        padded_inc_data = inc_data.set_index(time_key).reindex(
            range(
                min(data_times), max(data_times)+1)
                ).fillna(0).reset_index()

        self.cases_data = padded_inc_data[inc_key]
        self.cases_times = padded_inc_data[time_key]
        self.serial_interval = daily_serial_interval
        self.prior_parameters = (alpha, beta)

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
        alpha = self.prior_parameters[0]
        beta = self.prior_parameters[1]
        serial_interval = self.serial_interval
        time_init_inf_r = tau + 1
        cases = self.cases_data.to_list()

        shape = []
        rate = []
        mean = []

        for time in range(time_init_inf_r, total_time+1):
            # compute shape parameter of the posterior over time
            shape.append(
                alpha + math.fsum(cases[(time - tau):(time + 1)]))

            # compute shape parameter of the posterior over time
            incidences = 0

            for subtime in range(time - tau, time + 1):
                # incidences up to subtime taken in reverse order
                sub_incidences = cases[(subtime - 1)::-1]
                # serial interval values up to subtime
                sub_serials = serial_interval[:subtime]

                # compute the total amount of new incidences based on
                # previous days and the serial interval
                incidences += math.fsum(np.multiply(
                    sub_incidences, sub_serials))

            rate.append(1/beta + incidences)

        # compute the mean of the Gamma-shaped posterior over time
        mean = np.divide(shape, rate)

        # compute the Gamma-shaped posterior distribution
        post_dist = scipy.stats.gamma(shape, scale=1/np.array(rate))

        self.inference_times = list(range(time_init_inf_r, total_time+1))
        self.inference_estimates = mean
        self.inference_posterior = post_dist

    def get_intervals(self, central_prob):
        """
        Returns a dataframe of the estimated recombination number values,
        as well as the lower and upper bounds of a credible interval of
        specified level.

        Parameters
        ----------
        central_prob
            level of the computed credible interval of the estimated
            R number values.
        """
        # compute bounds of credible interval of level central_prob
        post_dist_interval = self.inference_posterior.interval(central_prob)

        intervals_df = pd.DataFrame(
            {
                'Estimates': self.inference_estimates,
                'Lower bound CI': post_dist_interval[0],
                'Upper bound CI': post_dist_interval[1]
            }
        )

        return intervals_df
