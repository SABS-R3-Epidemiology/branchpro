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
import numexpr as ne
import pandas as pd
import scipy.stats
import scipy.special
import scipy.integrate


class GammaDist:
    r"""Gamma distribution.

    Smaller version of the scipy.stats class. It uses the scipy methods, but
    only saves the shape and rate parameters in the object. Instantiation
    is much faster than scipy; method calls are similar in speed. It also uses
    less memory than scipy.

    It also has a new density function, :meth:`big_pdf`, which is faster on
    large array inputs.

    We use the shape/rate parametrization, under which the gamma pdf is:

    .. math::
        f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}

    for shape :math:`alpha` and rate :`beta`.
    """
    def __init__(self, shape, rate):
        self.shape = np.asarray(shape)
        self.rate = np.asarray(rate)
        self.scipy_args = {'a': self.shape, 'scale': 1/self.rate}

    def ppf(self, q):
        return scipy.stats.gamma.ppf(q, **self.scipy_args)

    def big_pdf(self, x):
        """Probability density function optimized for large inputs.

        For small arrays x, it will be slower than the regular pdf. However it
        can be much faster if x is a large array.
        """
        r = self.rate  # noqa
        a = self.shape
        logpdf = -scipy.special.gammaln(a)  # noqa
        pdf = ne.evaluate('exp(logpdf + (a-1.0) * log(r * x) - r * x) * r')
        return pdf

    def pdf(self, x):
        return scipy.stats.gamma.pdf(x, **self.scipy_args)

    def mean(self):
        return self.shape / self.rate

    def interval(self, central_prob):
        return scipy.stats.gamma.interval(central_prob, **self.scipy_args)

    def median(self):
        return scipy.stats.gamma.median(**self.scipy_args)


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
                (cases_data[start_date:(t-1)] * self._serial_interval).sum() /
                self._normalizing_const)
            return eff_num

        eff_num = (
            (cases_data[:(t-1)] * self._serial_interval[-(t-1):]).sum() /
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
        post_dist = GammaDist(shape, rate)

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

    def proportion_time_r_more_than_1(self, central_prob=.95, method='Mean'):
        """
        Return the proportion of time the reproduction number posterior
        mean, lower bound and upper bound for a specified central probability
        respectively are bigger than 1.

        Parameters
        ----------
        central_prob
            level of the computed credible interval of the estimated
            R number values. The interval the central probability.
        method
            choice for the average trajcetory of reproduction number
            considered; can be either `Mean` or `Median`.
        """
        self._check_method(method)

        intervals = self.get_intervals(central_prob)

        total_length = len(intervals['Time Points'].tolist())

        # Number of rows of `intervals` meeting condition
        subset_with_method = len(intervals.loc[
            intervals[method] > 1]['Time Points'].tolist())

        subset_with_lower = len(intervals.loc[
            intervals['Lower bound CI'] > 1]['Time Points'].tolist())

        subset_with_upper = len(intervals.loc[
            intervals['Upper bound CI'] > 1]['Time Points'].tolist())

        proportion_time_r_more_than_1 = subset_with_method/total_length
        proportion_time_r_more_than_1_LowerCI = \
            subset_with_lower/total_length
        proportion_time_r_more_than_1_UpperCI = \
            subset_with_upper/total_length

        return(
            proportion_time_r_more_than_1,
            proportion_time_r_more_than_1_LowerCI,
            proportion_time_r_more_than_1_UpperCI)

    def last_time_r_threshold(
            self, type_threshold, central_prob=.95, method='Mean'):
        """
        Return the value of the first time point after the reproduction
        number posterior mean, lower bound and upper bound for a specified
        central probability respectively crosses the imposed threshold for
        the last time during the inference.

        Parameters
        ----------
        type_threshold
            (str) type of threshold imposed; 'more' = last time R > 1 and
            'less' = last time R < 1.
        central_prob
            level of the computed credible interval of the estimated
            R number values. The interval the central probability.
        method
            choice for the average trajcetory of reproduction number
            considered; can be either `Mean` or `Median`.
        """
        intervals = self.get_intervals(central_prob)

        if type_threshold not in ['more', 'less']:
            raise ValueError('Threshold value must be `more` or `less` than'
                             ' 1.')

        self._check_method(method)

        # Subset only rows of `intervals` meeting condition
        if type_threshold == 'more':
            subset_with_method = intervals.loc[
                intervals[method] > 1]['Time Points'].tolist()

            subset_with_lower = intervals.loc[
                intervals['Lower bound CI'] > 1]['Time Points'].tolist()

            subset_with_upper = intervals.loc[
                intervals['Upper bound CI'] > 1]['Time Points'].tolist()

        elif type_threshold == 'less':
            subset_with_method = intervals.loc[
                intervals[method] < 1]['Time Points'].tolist()

            subset_with_lower = intervals.loc[
                intervals['Lower bound CI'] < 1]['Time Points'].tolist()

            subset_with_upper = intervals.loc[
                intervals['Upper bound CI'] < 1]['Time Points'].tolist()

        if len(subset_with_method) == 0:
            last_time_r_threshold = None
        else:
            last_time_r_threshold = subset_with_method[-1] + 1

        if len(subset_with_lower) == 0:
            last_time_r_threshold_LowerCI = None
        else:
            last_time_r_threshold_LowerCI = subset_with_lower[-1] + 1

        if len(subset_with_upper) == 0:
            last_time_r_threshold_UpperCI = None
        else:
            last_time_r_threshold_UpperCI = subset_with_upper[-1] + 1

        return(
            last_time_r_threshold,
            last_time_r_threshold_LowerCI,
            last_time_r_threshold_UpperCI)

    def _check_method(self, method):
        """
        Checks validity of method option.

        Paramaters
        ----------
        method
            choice for the average trajcetory of reproduction number
            considered; can be either `Mean` or `Median`.
        """
        if method not in ['Mean', 'Median']:
            raise ValueError('Method of selecting the R numbers can only be '
                             '`Mean` or `Median`.')

#
# BranchProPosteriorMultSI Class
#


class BranchProPosteriorMultSI(BranchProPosterior):
    r"""BranchProPosteriorMultiSI Class:
    Class for computing the posterior distribution used for the inference of
    the reproduction numbers of an epidemic in the case of a branching process
    using mutiple serial intevals. Based on the :class:`BranchProPosterior`.

    In order to incorporate the uncertainty in the serial interval into the
    posterior of :math:`R_t`, this class employs the approximation

    .. math::
        p(R_t|I) = \int p(R_t|I, w) p(w) dw
        \approx \frac{1}{N} \sum_{i=1}^N p(R_t|I,w^{(i)}); w^{(i)} \sim p(w)

    where :math:`I` indicates the incidence data. At instantiation, the user
    supplies the samples :math:`w^{(i)}` which are assumed to have been drawn
    IID from the distribution of serial intervals.

    Requested posterior percentiles are computed from the above density using
    numerical integration.

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
        # We assume that the serial interval samples were drawn IID. Thus, the
        # marginal posterior pdf can be approximated by an average of the
        # conditional posteriors (those saved in self._inference_samples.) See
        # the class docstring for further details.
        pdf_values = np.zeros((len(integration_grid), len(max_Rs[0])))
        for dist in samples:
            pdf_values += dist.big_pdf(integration_grid[:, np.newaxis])
        pdf_values *= 1 / N

        # Perform a cumulative integration of the posterior density using the
        # trapezoidal rule to get the cumulative distribution
        cdf = scipy.integrate.cumulative_trapezoid(
            pdf_values, x=integration_grid, axis=0)

        # Add zero for the first element of the CDF values
        cdf = np.vstack((np.zeros(cdf.shape[1]), cdf))

        # Do an interpolation to get the inverse CDF
        # Interpolation is performed separately for each time point
        inv_cdf = [scipy.interpolate.interp1d(cdf[:, t], integration_grid)
                   for t in range(cdf.shape[1])]

        # Define a function to get the values of R_t over time in an array
        def posterior_ppf(p):
            # p = probability
            # returns = posterior R_t trajectory
            return np.array([ppf(p)[()] for ppf in inv_cdf])

        self._posterior_ppf = posterior_ppf

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
                'Median': self._posterior_ppf(0.5),
                'Lower bound CI': self._posterior_ppf(lb),
                'Upper bound CI': self._posterior_ppf(ub),
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
        post_dist = GammaDist(shape, rate)

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
