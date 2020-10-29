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
        symptoms

    Methods
    -------
    simulate: return model output for specified parameters and times.
    _normalised_daily_mean: (Private) returns the expected number of new cases
        at time t
    """

    def __init__(self, initial_r, serial_interval):
        super(BranchProModel, self).__init__()

        if np.asarray(serial_interval).ndim != 1:
            raise ValueError(
                'Serial interval values storage format must be 1-dimensional')
        if np.sum(serial_interval) <= 0:
            raise ValueError('Sum of serial interval values must be > 0')
        if not isinstance(initial_r, (int, float)):
            raise TypeError('Value of R must be integer or float')

        self._serial_interval = np.asarray(serial_interval)[::-1]
        self._initial_r = initial_r
        self._normalizing_const = np.sum(self._serial_interval)

    def _normalised_daily_mean(
            self, t, incidences):
        """
        Computes expected number of new cases at time t, using previous
        incidences and serial intevals.

        Parameters
        ----------
        t: evaluation time
        incidences: sequence of incidence numbers
        """
        if t > len(self._serial_interval):
            start_date = t - len(self._serial_interval)
            mean = self._initial_r * (
                np.sum(incidences[start_date:t] * self._serial_interval) /
                self._normalizing_const)
            return mean

        mean = self._initial_r * (
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
            norm_daily_mean = self._normalised_daily_mean(t, incidences)
            incidences[t] = np.random.poisson(lam=norm_daily_mean, size=1)

        mask = np.in1d(np.append(np.asarray(0), simulation_times), times)
        return incidences[mask]
