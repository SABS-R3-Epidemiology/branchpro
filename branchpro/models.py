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
    It inherits from the `ForwardModel`` class.

    In the branching process model, we track the number of cases
    registered each day, I_t, also known as the "incidence" at time t.

    The incidence at time t is modelled by a random variable distributed
    according to a Poisson distribution with a mean that depends on previous
    number of cases, according to the following formula:

    .. math::
        E(I_{t}^{\text(local)}|I_0, I_1, \dots I_{t-1}, w_{s}, R_{t}) =
            R_{t}\sum_{s=1}^{t}I_{t-s}w_{s}

    Parameters
    ----------
    initial_r
        (numeric) Value of the reproduction number at the beginning
        of the epidemic
    serial_interval:
        (list) Unnormalised probability distribution of that the recipient
        first displays symptoms s days after the infector first displays
        symptoms.

    Methods
    -------
    simulate: return model output for specified parameters and times.
    _normalised_daily_mean: (Private) returns the expected number of new cases
        at time t.
    get_serial_intervals: returns serial intevals for the model.
    update_serial_intevals: updates serial intevals for the model.
<<<<<<< HEAD
    set_r_profile: creates a new R_t profile for the model.
    _reproduction_num: (Private) creates extended reproduction numbers profile,
        taking into account multiplicities according to the time profile.

    *Always apply method set_r_profile before simulation for a change of R_t
    profile!*

=======
>>>>>>> main
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
        self._initial_r = initial_r
        self._present_r_profile = np.array([initial_r])
        self._present_t_profile = np.array([1])
        self._normalizing_const = np.sum(self._serial_interval)

    def set_r_profile(self, new_rs, start_times):
        """
        Creates a new R_t profile for the model.

        Parameters
        ----------
        new_rs: sequence of new time-dependent values of the reproduction
            numbers.
        start_times: sequence of the first time unit when the corresponding
            indexed value of R_t in new_rs is used. Must be an ordered sequence
             and without duplicates or negative values.

        """
        # Raise error if not correct dimensionality of inputs
        if np.asarray(new_rs).ndim != 1:
            raise ValueError('New reproduction numbers storage format \
                must be 1-dimensional')
        if np.asarray(start_times).ndim != 1:
            raise ValueError('Starting times values storage format \
                must be 1-dimensional')

        # Raise error if inputs do not have same shape
        if np.asarray(new_rs).shape != np.asarray(start_times).shape:
            raise ValueError('Both inputs should have same number of elements')

        # Raise error if start times are not non-negative and increasing
        if np.any(np.asarray(start_times) < 0):
            raise ValueError('Times can not be negative.')
        if np.any(np.asarray(start_times)[:-1] >= np.asarray(start_times)[1:]):
            raise ValueError('Times must be increasing.')

        # Update the R_t and time profiles with the new values introduced
        self._present_r_profile = np.asarray(new_rs)
        self._initial_r = np.asarray(new_rs)[0]
        self._present_t_profile = np.asarray(start_times)

    def _reproduction_num(self, last_time):
        """
        Creates extended reproduction numbers profile, taking into
        account multiplicities according to the time profile.

        Used to compute the normalized mean in the simulation method.

        Parameters
        ---------
        last_time: Total evaluation and simulation time for the R_t profile.

        """
        # Read current R_t profile and their emerging times
        present_r_profile = np.asarray(self._present_r_profile)
        present_t_profile = np.asarray(self._present_t_profile)

        # Initialise the matrix which we will fill with the R_t per unit time
        reproduction_num = np.empty(shape=last_time+1)
        reproduction_num[0] = self._initial_r

        # Create an array of the reproduction numbers in each unit of time
        # Each element is in the R_t profile is used between
        # Their corresponding limits as expressed in the time profile
        for t in present_t_profile:
            # Compute index where t is in t_profile
            change_in_t_num = np.where(present_t_profile == t)[0][0]

            # Compute the start and finishing date of the time interval
            # beginning at t
            start_time_for_r = t
            end_time_for_r = present_t_profile[change_in_t_num + 1]
            time_spend_at_current_r = end_time_for_r - start_time_for_r

            # Compute value of R_t in this time interval
            r_for_t = present_r_profile[change_in_t_num]

            reproduction_num[start_time_for_r:end_time_for_r] = np.full(
                time_spend_at_current_r, r_for_t)

        # Return the filled matrix or R_t values
        return reproduction_num

    def get_serial_intevals(self):
        """
        Returns serial intevals for the model.

        """
        # Reverse inverting of order of serial intervals
        return self._serial_interval[::-1]

    def set_serial_intevals(self, serial_intevals):
        """
        Updates serial intevals for the model.

        Parameters
        ----------
        serial_intervals
            New unnormalised probability distribution of that the recipient
            first displays symptoms s days after the infector first displays
            symptoms.

        """
        if np.asarray(serial_intevals).ndim != 1:
            raise ValueError(
                'Chosen times storage format must be 1-dimensional')

        # Invert order of serial intervals for ease in _normalised_daily_mean
        self._serial_interval = np.asarray(serial_intevals)[::-1]

    def _normalised_daily_mean(self, t, incidences, last_time):
        """
        Computes expected number of new cases at time t, using previous
        incidences and serial intevals.

        Parameters
        ----------
        t: evaluation time
        incidences: sequence of incidence numbers
        last_time: total evaluation and simulation time for the R_t profile.

        """
        if t > len(self._serial_interval):
            start_date = t - len(self._serial_interval)
            mean = self._reproduction_num(last_time)[t] * (
                np.sum(incidences[start_date:t] * self._serial_interval) /
                self._normalizing_const)
            return mean

        mean = self._reproduction_num(last_time)[t] * (
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

        # Initialise list of number of cases per unit time
        # with initial condition I0
        last_time_point = np.max(times)
        incidences = np.empty(shape=last_time_point + 1)
        incidences[0] = initial_cond

        # Construct simulation times in steps of 1 unit time each
        simulation_times = np.arange(start=1, stop=last_time_point+1, step=1)

        # Compute normalised daily means for full timespan
        # and draw samples for the incidences
        for t in simulation_times:
            norm_daily_mean = self._normalised_daily_mean(
                t, incidences, last_time_point)
            incidences[t] = np.random.poisson(lam=norm_daily_mean, size=1)

        mask = np.in1d(np.append(np.asarray(0), simulation_times), times)
        return incidences[mask]


branch_model_1 = BranchProModel(2, np.array([1, 2, 3, 2, 1]))
print(branch_model_1._reproduction_num(1))
