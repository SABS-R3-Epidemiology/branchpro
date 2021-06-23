#
# ForwardModel Class
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#
import numpy as np


class ForwardModel(object):
    """ForwardModel Class:
    Base class for the model classes included in the branchpro package.
    Classes inheriting from ``ForwardModel`` class can implement the methods
    directly in Python.

    Methods
    -------
    simulate: return model output for specified parameters and times.

    """

    def __init__(self):
        super(ForwardModel, self).__init__()

    def simulate(self, parameters, times):
        """
        Runs a forward simulation with the given ``parameters`` and returns a
        time-series with data points corresponding to the given ``times``.

        Returns a sequence of length ``n_times`` (for single output problems)
        or a NumPy array of shape ``(n_times, n_outputs)`` (for multi-output
        problems), representing the values of the model at the given ``times``.

        Parameters
        ----------
        parameters
            An ordered sequence of parameter values.
        times
            The times at which to evaluate. Must be an ordered sequence,
            without duplicates, and without negative values.
            All simulations are started at time 0, regardless of whether this
            value appears in ``times``.

        """
        raise NotImplementedError

#
# BranchProModel Class
#


class BranchProModel(ForwardModel):
    r"""BranchProModel Class:
    Class for the models following a Branching Processes behaviour.
    It inherits from the ``ForwardModel`` class.

    In the branching process model, we track the number of cases
    registered each day, I_t, also known as the "incidence" at time t.

    The incidence at time t is modelled by a random variable distributed
    according to a Poisson distribution with a mean that depends on previous
    number of cases, according to the following formula:

    .. math::
        E(I_{t}^{\text(local)}|I_0, I_1, \dots I_{t-1}, w_{s}, R_{t}) =
            R_{t}\sum_{s=1}^{t}I_{t-s}w_{s}

    Always apply method :meth:`set_r_profile` before calling
    :meth:`BranchProModel.simulate` for a change of R_t profile!

    Parameters
    ----------
    initial_r
        (numeric) Value of the reproduction number at the beginning
        of the epidemic
    serial_interval
        (list) Unnormalised probability distribution of that the recipient
        first displays symptoms s days after the infector first displays
        symptoms.

    """

    def __init__(self, initial_r, serial_interval):
        super(BranchProModel, self).__init__()

        if np.asarray(serial_interval).ndim != 1:
            raise ValueError(
                'Serial interval values storage format must be 1-dimensional')
        if np.sum(serial_interval) <= 0:
            raise ValueError('Sum of serial interval values must be > 0.')
        if not isinstance(initial_r, (int, float)):
            raise TypeError('Value of R must be integer or float.')

        # Invert order of serial intervals for ease in _normalised_daily_mean
        self._serial_interval = np.asarray(serial_interval)[::-1]
        self._r_profile = np.array([initial_r])
        self._normalizing_const = np.sum(self._serial_interval)

    def set_r_profile(self, new_rs, start_times, last_time=None):
        """
        Creates a new R_t profile for the model.

        Parameters
        ----------
        new_rs
            sequence of new time-dependent values of the reproduction
            numbers.
        start_times
            sequence of the first time unit when the corresponding
            indexed value of R_t in new_rs is used. Must be an ordered sequence
            and without duplicates or negative values.
        last_time
            total evaluation time; optional.

        """
        # Raise error if not correct dimensionality of inputs
        if np.asarray(new_rs).ndim != 1:
            raise ValueError(
                'New reproduction numbers storage format must be 1-dimensional'
                )
        if np.asarray(start_times).ndim != 1:
            raise ValueError(
                'Starting times values storage format must be 1-dimensional')

        # Raise error if inputs do not have same shape
        if np.asarray(new_rs).shape != np.asarray(start_times).shape:
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
        initial_r = self._r_profile[0]
        r_profile += [initial_r] * (times[0] - 1)

        # Fill in later r's
        time_intervals = times[1:] - times[:-1]
        for time_id, time_interval in enumerate(time_intervals):
            # Append r for each time unit
            r_profile += [new_rs[time_id]] * time_interval

        # Add final r
        if last_time:
            final_interval = last_time - start_times[-1] + 1
            r_profile += [new_rs[-1]] * final_interval
        else:
            r_profile += [new_rs[-1]]

        self._r_profile = np.asarray(r_profile)

    def get_serial_intervals(self):
        """
        Returns serial intervals for the model.

        """
        # Reverse inverting of order of serial intervals
        return self._serial_interval[::-1]

    def get_r_profile(self):
        """
        Returns R_t profile for the model.

        """
        return self._r_profile

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
                np.sum(incidences[start_date:t] * self._serial_interval) /
                self._normalizing_const)
            return mean

        mean = (
            np.sum(incidences[:t] * self._serial_interval[-t:]) /
            self._normalizing_const)
        return mean

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
            last_r = self._r_profile[-1]
            repeated_r = np.full(shape=missing_days, fill_value=last_r)
            self._r_profile = np.append(self._r_profile, repeated_r)

        incidences = np.empty(shape=last_time_point + 1)
        incidences[0] = initial_cond

        # Construct simulation times in steps of 1 unit time each
        simulation_times = np.arange(start=1, stop=last_time_point+1, step=1)

        # Compute normalised daily means for full timespan
        # and draw samples for the incidences
        for t in simulation_times:
            norm_daily_mean = self._r_profile[t-1] * (
                self._effective_no_infectives(t, incidences))
            incidences[t] = np.random.poisson(lam=norm_daily_mean, size=1)

        mask = np.in1d(np.append(np.asarray(0), simulation_times), times)
        return incidences[mask]


class LocImpBranchProModel(BranchProModel):
    r"""LocImpBranchProModel Class:
    Class for the models following a Branching Processes behaviour with
    local and imported cases. It inherits from the ``BranchProModel`` class.

    In the branching process model, we track the number of cases
    registered each day, I_t, also known as the "incidence" at time t.

    For the local & imported cases scenario, to the local incidences we add
    migration of cases from an external source. The conditions of this external
    environment may differ from the ones we are currently in through a change
    in the value of the R number.

    To account for this difference, we assume that at all times the R number of
    the imported cases is proportional to the R number of the local incidences:

    .. math::
        R_{t}^{\text(imported)} = \epsilon R_{t}^{\text(local)}

    The local incidence at time t is modelled by a random variable distributed
    according to a Poisson distribution with a mean that depends on previous
    number of cases both local and imported, according to the following
    formula:

    .. math::
        E(I_{t}^{\text(local)}|I_0, I_1, \dots I_{t-1}, w_{s}, R_{t}) =
            R_{t}^{\text(local)}\sum_{s=1}^{t}I_{t-s}^{\text(local)}w_{s} +
            R_{t}^{\text(imported)}\sum_{s=1}^{t}I_{t-s}^{\text(imported)}w_{s}

    Always apply methods :meth:`set_r_profile` and :meth:`set_imported_cases`
    before calling :meth:`LocImpBranchProModel.simulate` for a change of R_t
    profile and for loading the imported cases data!

    Parameters
    ----------
    initial_r
        (numeric) Value of the reproduction number at the beginning
        of the epidemic
    serial_interval
        (list) Unnormalised probability distribution of that the recipient
        first displays symptoms s days after the infector first displays
        symptoms
    epsilon
        (numeric) Proportionality constant of the R number for imported cases
        with respect to its analog for local ones.

    """
    def __init__(self, initial_r, serial_interval, epsilon):
        super().__init__(initial_r, serial_interval)

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
        Sets number of imported cases and when they occur.

        Parameters
        ----------
        times
            times at which imported cases occur. Must be an ordered sequence,
            without duplicates, and without negative values.
        cases
            number of imported cases at that specified point.

        """
        # Raise error if not correct dimensionality of inputs
        if np.asarray(times).ndim != 1:
            raise ValueError(
                'Times of arising imported cases storage format must \
                    be 1-dimensional'
                )
        if np.asarray(cases).ndim != 1:
            raise ValueError(
                'Number of imported cases storage format must be \
                    1-dimensional')

        # Raise error if inputs do not have same shape
        if np.asarray(times).shape != np.asarray(cases).shape:
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
            last_r = self._r_profile[-1]
            repeated_r = np.full(shape=missing_days, fill_value=last_r)
            self._r_profile = np.append(self._r_profile, repeated_r)

        incidences = np.empty(shape=last_time_point + 1)
        incidences[0] = initial_cond

        # Create vector of imported cases
        imported_incidences = np.zeros(last_time_point + 1, dtype=int)

        imports_times = self._imported_times[
            self._imported_times <= last_time_point]
        imports_cases = self._imported_cases[
            self._imported_times <= last_time_point]

        np.put(
            imported_incidences, ind=imports_times, v=imports_cases)

        # Construct simulation times in steps of 1 unit time each
        simulation_times = np.arange(start=1, stop=last_time_point+1, step=1)

        # Compute normalised daily means for full timespan
        # and draw samples for the incidences
        for t in simulation_times:
            norm_daily_mean = self._r_profile[t-1] * (
                self._effective_no_infectives(
                    t, incidences) + self.epsilon * (
                        self._effective_no_infectives(
                            t, imported_incidences)))
            incidences[t] = np.random.poisson(lam=norm_daily_mean, size=1)

        mask = np.in1d(np.append(np.asarray(0), simulation_times), times)
        return incidences[mask]
