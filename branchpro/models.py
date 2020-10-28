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
    """BranchProModel Class:

    Parameters
    ----------
    value: numeric, optional
        example of value
    """

    def __init__(self, initial_r, serial_interval):
        super(BranchProModel, self).__init__()

        if np.sum(serial_interval) <= 0:
            raise ValueError('Sum of serial interval values must be > 0')

        self._serial_interval = np.asarray(serial_interval)
        self._present_r_profile = np.asarray(initial_r)
        self._present_t_profile = np.asarray(1)

    def __normalised_daily_mean(self, t, incidences, reproduction_num, serial_interval):  # noqa
        return reproduction_num[t-1] * sum([incidences[t - s - 1] * serial_interval[s] for s in range(t)]) / np.sum(serial_interval)  # noqa

    def add_r_steps(self, new_rs, start_times):
        # Raise error if inputs do not have same dimensions
        if np.asarray(new_rs).ndim != np.asarray(start_times).ndim:
            raise ValueError('Both inputs need to have same dimension')

        # Read most recent R_t and time profile
        present_r_profile = np.asarray(self._present_r_profile)
        present_t_profile = np.asarray(self._present_t_profile)

        # Add new R_t values and the corresponding first time at which
        # this particular R_t had started to be used
        new_r_profile = np.append(present_r_profile, np.asarray(new_rs))
        new_t_profile = np.append(present_t_profile, np.asarray(start_times))

        # Update the R_t profile with the latest value introduced
        self._present_r_profile = new_r_profile
        self._present_t_profile = new_t_profile

    def reproduction_num(self, last_time):
        # Read current R_t profile and their emerging times
        present_r_profile = np.asarray(self._present_r_profile)
        present_t_profile = np.asarray(self._present_t_profile)

        # Initialise the matrix which we will fill with the R_t per unit time
        reproduction_num = np.empty(shape=last_time)

        # Create an array of the reproduction numbers in each unit of time
        # Each element is in the R_t profile is used between
        # Their corresponding limits as expressed in the time profile
        for r in present_r_profile:
            change_in_r_num = np.where(present_r_profile == r)
            start_time_for_r = present_t_profile[change_in_r_num] - 1
            end_time_for_r = present_t_profile[change_in_r_num + 1] - 1
            time_spend_at_current_r = end_time_for_r - start_time_for_r
            reproduction_num[start_time_for_r:end_time_for_r] = np.full(time_spend_at_current_r, r)  # noqa

        # Return the filled matrix or R_t values
        return reproduction_num

    def simulate(self, parameters, times):
        initial_cond = parameters

        # Initialise list of number of cases per unit time
        # with initial condition I0 and vector or R_t
        last_time_point = np.max(times)

        reproduction_num = self.reproduction_num(last_time=last_time_point)
        incidences = np.empty(shape=last_time_point + 1)
        incidences[0] = initial_cond

        # Construct simulation times in steps of 1 unit time each
        simulation_times = np.arange(start=1, stop=last_time_point, step=1)

        # Compute normalised daily means for full timespan
        # and draw samples for the incidences
        for t in simulation_times:
            norm_daily_mean = self.__normalised_daily_mean(t, incidences, reproduction_num, serial_interval)  # noqa
            incidences[t] = np.random.poisson(lam=norm_daily_mean, size=1)

        return incidences[np.in1d(np.append(np.asarray(0), simulation_times), times)]  # noqa
