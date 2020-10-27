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
        self._initial_r = initial_r

    def __normalised_daily_mean__(self, t, incidences, reproduction_num, serial_interval):  # noqa
        return reproduction_num * sum([incidences[t - s] * serial_interval[s - 1] for s in range(t)]) / np.sum(serial_interval)  # noqa

    def simulate(self, parameters, times):
        initial_cond = parameters
        reproduction_num = self._initial_r

        # Initialise list of number of cases per unit time
        # with initial condition I0
        last_time_point = np.max(times)
        incidences = np.empty(shape=last_time_point + 1)
        incidences[0] = initial_cond

        # Construct simulation times in steps of 1 unit time each
        simulation_times = np.arange(start=1, stop=last_time_point, step=1)

        # Compute normalised daily means for full timespan
        # and draw samples for the incidences
        for t in simulation_times:
            norm_daily_mean = self.__normalised_daily_mean__(t, incidences, reproduction_num, serial_interval)  # noqa
            incidences[t] = np.random.poisson(lam=norm_daily_mean, size=1)

        return incidences[np.in1d(np.append(np.asarray(0), simulation_times), times)]  # noqa
