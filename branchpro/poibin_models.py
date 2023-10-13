#
# PoissonBinomialBranchProModel Class
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#
from branchpro import ForwardModel
import numpy as np
from fast_poibin import PoiBin


class PoiBinBranchProModel(ForwardModel):
    r"""PoiBinBranchProModel Class:
    Class for the models following a Branching Processes behaviour with
    Poisson distribution noise and multiple population categories.
    It inherits from the ``BranchProModel`` class.

    In the branching process model, we track the number of cases
    registered for each category and each day, I_{t,i}, also known as the
    "incidence" at time t.

    The incidence at time t is modelled by a random variable distributed
    according to a Poisson distribution with a mean that depends on
    previous number of cases, according to the following formula:

    .. math::
        E(I_{t, i}^{\text(local)}|I_0, I_1, \dots I_{t-1}, w_{s}, R_{t}) =
            \sum_{j}R_{t, j}\sum_{s=1}^{t}I_{t-s,j}w_{s}

    The total reproduction number at time t is computed as follows:

    .. math::
        R_{t} = \sum_{j}R_{t, j}

    Always apply method :meth:`set_r_profile` before calling
    :meth:`NegBinBranchProModel.simulate` for a change of R_t profile!

    Parameters
    ----------
    initial_mu
        (numeric) Value of the mean number of contacts of an individual at the
        beginning of the epidemic.
    next_gen
        (list) Unnormalised probability distribution of that the infector
        infects a contact s days after they first displays symptoms.

    """

    def __init__(self, initial_mu, next_gen):
        if np.asarray(next_gen).ndim != 1:
            raise ValueError(
                'Next generation distribution values storage format must be \
                1-dimensional')
        for ng in next_gen:
            if (ng < 0) or (ng > 1):
                raise ValueError(
                    'Next generation distribution values must be >= 0 and \
                    <= 1.')

        if not isinstance(initial_mu, (int, float)):
            raise TypeError(
                'Value of mean number of contacts must be integer or float.')

        # Invert order of next generation distribution for ease in
        # _normalised_daily_mean
        self._next_gen = np.asarray(next_gen)[::-1]
        self._mu = np.array([initial_mu])

    def set_mean_contact(self, new_mus, start_times, last_time=None):
        """
        Creates a new profile of the time-dependent mean number of contacts of
        an individual for the model.

        Parameters
        ----------
        new_mus
            sequence of new time-dependent values of the mean number of
            contacts of an individual.
        start_times
            sequence of the first time unit when the corresponding
            indexed value of mu_t in new_mus is used. Must be an ordered
            sequence and without duplicates or negative values.
        last_time
            total evaluation time; optional.

        """
        # Raise error if not correct dimensionality of inputs
        if np.asarray(new_mus).ndim != 1:
            raise ValueError(
                'New average contacts numbers storage format must be \
                1-dimensional')
        if np.asarray(start_times).ndim != 1:
            raise ValueError(
                'Starting times values storage format must be 1-dimensional')

        # Raise error if inputs do not have same shape
        if np.asarray(new_mus).shape != np.asarray(start_times).shape:
            raise ValueError('Both inputs should have same number of elements')

        # Raise error if start times are not non-negative and increasing
        if np.any(np.asarray(start_times) < 0):
            raise ValueError('Times can not be negative.')
        if np.any(np.asarray(start_times)[:-1] >= np.asarray(start_times)[1:]):
            raise ValueError('Times must be increasing.')

        # Ceil times to integer numbers a
        times = np.ceil(start_times).astype(int)

        # Create mu profile
        mu = []

        # Fill in initial mu from day 1 up to start time
        initial_mu = self._mu[0]
        mu += [initial_mu] * (times[0] - 1)

        # Fill in later mu's
        time_intervals = times[1:] - times[:-1]
        for time_id, time_interval in enumerate(time_intervals):
            # Append mu for each time unit
            mu += [new_mus[time_id]] * time_interval

        # Add final r
        if last_time:
            final_interval = last_time - start_times[-1] + 1
            mu += [new_mus[-1]] * final_interval
        else:
            mu += [new_mus[-1]]

        self._mu = np.asarray(mu)

    def get_next_gen(self):
        """
        Returns next generation probability distribution for the model.

        """
        # Reverse inverting of order of serial intervals
        return self._next_gen[::-1]

    def get_mean_contact(self):
        """
        Returns the profile of the time-dependent mean number of contacts of
        an individual for the model.

        """
        return self._mu

    def set_next_gen(self, new_next_gen):
        """
        Updates next generation probability distribution for the model.

        Parameters
        ----------
        new_next_gen
            New unnormalised probability distribution of that the infector
            infects a contact s days after they first displays symptoms

        """
        if np.asarray(new_next_gen).ndim != 1:
            raise ValueError(
                'Chosen times storage format must be 1-dimensional')

        # Invert order of serial intervals for ease in _effective_no_infectives
        self._next_gen = np.asarray(new_next_gen)[::-1]

    def _effective_success_prob(self, t, mu, incidences):
        """
        Computes the vector of probabilities of successful infections for all
        individuals at time t, using previous incidences and the next
        generation distribution.

        Parameters
        ----------
        t
            evaluation time
        mu
            value of the average number of contacts
        incidences
            sequence of incidence numbers

        """
        mean = []
        if t > len(self._next_gen):
            start_date = t - len(self._next_gen)
            for s in range(start_date, t):
                # Compute H_{s,I_1+...+I_(s-1)+1,}, ..., H_{s,I_1+...+I_s}
                contacts_of_inf_at_time_s = np.random.poisson(
                    lam=mu, size=int(incidences[s]))
                mean += [self._next_gen[s-t]] * np.sum(
                    contacts_of_inf_at_time_s)
            return mean
        if t == 1:
            # Compute H_{s,I_1+...+I_(s-1)+1,}, ..., H_{s,I_1+...+I_s}
            contacts_of_inf_at_time_s = np.random.poisson(
                lam=mu, size=int(incidences[0]))
            return [self._next_gen[-t]] * np.sum(contacts_of_inf_at_time_s)
        else:
            for s in range(t):
                # Compute H_{s,I_1+...+I_(s-1)+1,}, ..., H_{s,I_1+...+I_s}
                contacts_of_inf_at_time_s = np.random.poisson(
                    lam=mu, size=int(incidences[s]))
                mean += [self._next_gen[s-t]] * np.sum(
                    contacts_of_inf_at_time_s)
            return mean

    def simulate(self, parameters, times):
        """
        Runs a forward simulation with the given ``parameters`` and returns a
        time-series with incidence numbers per population category
        corresponding to the given ``times``.

        Parameters
        ----------
        parameters
            Initial number of cases per population category.
        times
            The times at which to evaluate. Must be an ordered sequence,
            without duplicates, and without negative values.
            All simulations are started at time 0, regardless of whether this
            value appears in ``times``.

        """
        initial_cond = parameters
        last_time_point = np.max(times)

        # Repeat final mu if necessary
        # (mu_1, mu_2, ..., mu_t)
        if self._mu.shape[0] < last_time_point:
            missing_days = last_time_point - self._mu.shape[0]
            last_mu = self._mu[-1]
            repeated_mu = np.full(shape=missing_days, fill_value=last_mu)
            self._mu = np.append(self._mu, repeated_mu)

        incidences = np.empty(shape=last_time_point + 1)
        incidences[0] = initial_cond

        # Construct simulation times in steps of 1 unit time each
        simulation_times = np.arange(start=1, stop=last_time_point+1, step=1)

        # Compute normalised daily means for full timespan
        # and draw samples for the incidences
        for t in simulation_times:
            norm_daily_mean = self._effective_success_prob(t, self._mu[t-1],
                                                           incidences)
            # Use inversion sampling
            incidences[t] = PoiBin(probabilities=norm_daily_mean).quantile(
                p=np.random.uniform(size=1))

        mask = np.in1d(np.append(np.asarray(0), simulation_times), times)
        return incidences[mask]
