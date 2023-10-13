#
# MultiCatPoissonBranchProModel Class
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#
from branchpro import BranchProModel
import numpy as np


class MultiCatPoissonBranchProModel(BranchProModel):
    r"""MultiCatPoissonBranchProModel Class:
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
    initial_r
        (list) List of reproduction numbers per category at the beginning
        of the epidemic
    serial_interval
        (list) Unnormalised probability distribution of that the recipient
        first displays symptoms s days after the infector first displays
        symptoms.
    num_cat
        (int) Number of categories in which the population is split.

    """

    def __init__(self, initial_r, serial_interval, num_cat):
        if np.asarray(serial_interval).ndim != 1:
            raise ValueError(
                'Serial interval values storage format must be 1-dimensional')
        if np.sum(serial_interval) < 0:
            raise ValueError('Sum of serial interval values must be >= 0.')

        if not isinstance(num_cat, int):
            raise TypeError('Number of population categories must be integer.')
        if num_cat <= 0:
            raise ValueError('Number of population categories must be > 0.')

        if np.asarray(initial_r).ndim != 1:
            raise ValueError(
                'R values storage format must be 1-dimensional.')
        if np.asarray(initial_r).shape[0] != num_cat:
            raise ValueError(
                'R values storage format must match the number of population \
                categories.')
        for _ in initial_r:
            if not isinstance(_, (int, float)):
                raise TypeError('R values must be integer or float.')

        # Invert order of serial intervals for ease in _normalised_daily_mean
        self._num_cat = num_cat
        self._serial_interval = np.asarray(serial_interval)[::-1]
        self._r_profile = np.array([initial_r])
        self._normalizing_const = np.sum(self._serial_interval)

    def set_r_profile(self, new_rs, start_times, last_time=None):
        """
        Creates a new R_t per category profile for the model.

        Parameters
        ----------
        new_rs
            sequence of new time-dependent values of the reproduction
            numbers per category (times x number of categories).
        start_times
            sequence of the first time unit when the corresponding
            indexed value of R_t in new_rs is used. Must be an ordered sequence
            and without duplicates or negative values.
        last_time
            total evaluation time; optional.

        """
        # Raise error if not correct dimensionality of inputs
        if np.asarray(new_rs).ndim != 2:
            raise ValueError(
                'New reproduction numbers storage format must be 2-dimensional'
                )
        if np.asarray(new_rs).shape[1] != self._num_cat:
            raise ValueError(
                'New reproduction numbers storage format must match the number\
                 of population categories.'
                )
        if np.asarray(start_times).ndim != 1:
            raise ValueError(
                'Starting times values storage format must be 1-dimensional')

        # Raise error if inputs do not have same shape
        if np.asarray(new_rs).shape[0] != np.asarray(start_times).shape[0]:
            raise ValueError('Both inputs should have same number of elements')

        # Raise error if start times are not non-negative and increasing
        if np.any(np.asarray(start_times) < 0):
            raise ValueError('Times can not be negative.')
        if np.any(np.asarray(start_times)[:-1] >= np.asarray(start_times)[1:]):
            raise ValueError('Times must be increasing.')

        # Ceil times to integer numbers a
        times = np.ceil(start_times).astype(int)

        # Create r profile
        r_profile = []

        # Fill in initial r from day 1 up to start time
        initial_r = self._r_profile[0, :]
        r_profile += [initial_r] * (times[0] - 1)

        # Fill in later r's
        time_intervals = times[1:] - times[:-1]
        for time_id, time_interval in enumerate(time_intervals):
            # Append r for each time unit
            r_profile += [np.asarray(new_rs)[time_id, :]] * time_interval

        # Add final r
        if last_time:
            final_interval = last_time - start_times[-1] + 1
            r_profile += [np.asarray(new_rs)[-1, :]] * final_interval
        else:
            r_profile += [np.asarray(new_rs)[-1, :]]

        self._r_profile = np.asarray(r_profile)

    def _effective_no_infectives(self, t, incidences):
        """
        Computes expected number of new cases at time t, using previous
        incidences and serial intervals at a rate of 1:1 reproduction.

        Parameters
        ----------
        t
            evaluation time
        incidences
            sequence of incidence numbers
        """
        if t > len(self._serial_interval):
            start_date = t - len(self._serial_interval)
            mean = (
                np.matmul(self._serial_interval, incidences[start_date:t, :]) /
                self._normalizing_const)
            return mean

        mean = (
            np.matmul(self._serial_interval[-t:], incidences[:t, :]) /
            self._normalizing_const)
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

        # Repeat final r if necessary
        # (r_1, r_2, ..., r_t)
        if self._r_profile.shape[0] < last_time_point:
            missing_days = last_time_point - len(self._r_profile)
            last_r = self._r_profile[-1, :]
            repeated_r = np.array([list(last_r)] * missing_days)
            self._r_profile = np.concatenate(
                [self._r_profile, repeated_r], axis=0)

        incidences = np.empty(shape=(last_time_point + 1, self._num_cat))
        incidences[0, :] = initial_cond

        # Construct simulation times in steps of 1 unit time each
        simulation_times = np.arange(start=1, stop=last_time_point+1, step=1)

        # Compute normalised daily means for full timespan
        # and draw samples for the incidences
        for t in simulation_times:
            norm_daily_mean = np.matmul(self._r_profile[t-1, :], (
                self._effective_no_infectives(t, incidences)))
            incidences[t, :] = np.random.poisson(
                lam=norm_daily_mean, size=self._num_cat)

        mask = np.in1d(np.append(np.asarray(0), simulation_times), times)
        return incidences[mask]


class LocImpMultiCatPoissonBranchProModel(MultiCatPoissonBranchProModel):
    r"""LocImpMultiCatPoissonBranchProModel Class:
    Class for the models following a Branching Processes behaviour with
    local and imported cases and Poisson distribution noise with multiple
    population categories. It inherits from the ``NegBinBranchProModel`` class.

    In the branching process model, we track the number of cases
    registered for each category and each day, I_{t,i}, also known as the
    "incidence" at time t.

    For the local & imported cases scenario, to the local incidences we add
    migration of cases from an external source. The conditions of this external
    environment may differ from the ones we are currently in through a change
    in the value of the R number.

    To account for this difference, we assume that at all times the R number of
    the imported cases is proportional to the R number of the local incidences:

    .. math::
        R_{t, i}^{\text(imported)} = \epsilon R_{t, i}^{\text(local)}

    The local incidence at time t is modelled by a random variable distributed
    according to a negative binomial distribution with a mean that depends on
    previous number of cases both local and imported, according to the
    following formula:

    .. math::
        E(I_{t, i}^{\text(local)}|I_0, I_1, \dots I_{t-1}, w_{s}, R_{t}) =
            \sum_{j}R_{t, j}^{\text(local)}\sum_{s=1}^{t}I_{t-s,j}^{
            \text(local)}w_{s} + R_{t,j}^{\text(imported)}\sum_{s=1}^{t}
            I_{t-s,j}^{\text(imported)}w_{s}

    Always apply methods :meth:`set_r_profile` and :meth:`set_imported_cases`
    before calling :meth:`LocImpNegBinBranchProModel.simulate` for a change of
    R_t profile and for loading the imported cases data!

    Parameters
    ----------
    initial_r
        (list) List of reproduction numbers per category at the beginning
        of the epidemic.
    serial_interval
        (list) Unnormalised probability distribution of that the recipient
        first displays symptoms s days after the infector first displays
        symptoms.
    epsilon
        (numeric) Proportionality constant of the R number for imported cases
        with respect to its analog for local ones.
    num_cat
        (int) Number of categories in which the population is split.

    """
    def __init__(self, initial_r, serial_interval, epsilon, num_cat):
        super().__init__(initial_r, serial_interval, num_cat)

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

    def set_imported_cases(self, times, cases):
        """
        Sets number of imported cases per population category and when they
        occur.

        Parameters
        ----------
        times
            times at which imported cases occur. Must be an ordered sequence,
            without duplicates, and without negative values.
        cases
            number of imported cases per population category at that specified
            point.

        """
        # Raise error if not correct dimensionality of inputs
        if np.asarray(times).ndim != 1:
            raise ValueError(
                'Times of arising imported cases storage format must \
                    be 1-dimensional'
                )
        if np.asarray(cases).ndim != 2:
            raise ValueError(
                'Number of imported cases storage format must be \
                    2-dimensional')
        if np.asarray(cases).shape[1] != self._num_cat:
            raise ValueError(
                'Number of imported cases storage format must match the number\
                 of population categories.'
                )

        # Raise error if inputs do not have same shape
        if np.asarray(times).shape[0] != np.asarray(cases).shape[0]:
            raise ValueError('Both inputs should have same number of elements')

        self._imported_times = np.asarray(times, dtype=int)
        self._imported_cases = np.asarray(cases)

    def simulate(self, parameters, times):
        """
        Runs a forward simulation with the given ``parameters`` and returns a
        time-series with incidence numbers corresponding to the given ``times``
        .

        Parameters
        ----------
        parameters
            Initial number of cases.
        times
            The times at which to evaluate. Must be an ordered sequence,
            without duplicates, and without negative values.
            All simulations are started at time 0, regardless of whether this
            value appears in ``times``.

        """
        initial_cond = parameters
        last_time_point = np.max(times)

        # Repeat final r if necessary
        # (r_1, r_2, ..., r_t)
        if len(self._r_profile) < last_time_point:
            missing_days = last_time_point - len(self._r_profile)
            last_r = self._r_profile[-1, :]
            repeated_r = np.array([list(last_r)] * missing_days)
            self._r_profile = np.concatenate(
                [self._r_profile, repeated_r], axis=0)

        incidences = np.empty(shape=(last_time_point + 1, self._num_cat))
        incidences[0, :] = initial_cond

        # Create vector of imported cases
        imported_incidences = np.zeros(
            shape=(last_time_point + 1, self._num_cat), dtype=int)

        imports_times = self._imported_times[
            self._imported_times <= last_time_point]
        imports_cases = self._imported_cases[
            self._imported_times <= last_time_point, :]

        np.put(
            imported_incidences, ind=imports_times, v=imports_cases)

        # Construct simulation times in steps of 1 unit time each
        simulation_times = np.arange(start=1, stop=last_time_point+1, step=1)

        # Compute normalised daily means for full timespan
        # and draw samples for the incidences
        for t in simulation_times:
            norm_daily_mean = np.matmul(
                self._r_profile[t-1, :],
                self._effective_no_infectives(
                    t, incidences) + self.epsilon * (
                        self._effective_no_infectives(
                            t, imported_incidences)))
            incidences[t, :] = np.random.poisson(
                lam=norm_daily_mean, size=self._num_cat)

        mask = np.in1d(np.append(np.asarray(0), simulation_times), times)
        return incidences[mask]
