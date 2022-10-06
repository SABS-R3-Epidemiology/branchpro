#
# NegBinBranchProModel Class
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#
from branchpro import BranchProModel, LocImpBranchProModel
import numpy as np
from scipy.stats import nbinom


class NegBinBranchProModel(BranchProModel):
    r"""NegBinBranchProModel Class:
    Class for the models following a Branching Processes behaviour with
    non-Poisson distribution noise.
    It inherits from the ``BranchProModel`` class.

    In the branching process model, we track the number of cases
    registered each day, I_t, also known as the "incidence" at time t.

    The incidence at time t is modelled by a random variable distributed
    according to a negative binomial distribution with a mean that depends on
    previous number of cases, according to the following formula:

    .. math::
        E(I_{t}^{\text(local)}|I_0, I_1, \dots I_{t-1}, w_{s}, R_{t}) =
            R_{t}\sum_{s=1}^{t}I_{t-s}w_{s}

    The probability mass function of the incidence at time t thus becomes:

    .. math::
        P(I_{t}^{\text(local)}=k|I_0, I_1, \dots I_{t-1}, w_{s}, R_{t}) =
            \frac{\Gamma(k+\frac{1}{\phi})}{k!\Gamma(\frac{1}{\phi})}
            \Big(\frac{1}{1+\mu\phi}\Big)^{\frac{1}{\phi}}
            \Big(\frac{\mu\phi}{1+\mu\phi}\Big)^{k}

    where :math:`\mu` is mean of the distribution defined as above and
    :math:`\phi` is the overdispersion parameter associated with the negative
    binomial noise distribution.

    For the edge case, :math:`\phi = 0`, the incidence at time t becomes
    Poisson distributed, reducing to the simple class:`BranchProModel` class.

    Always apply method :meth:`set_r_profile` before calling
    :meth:`NegBinBranchProModel.simulate` for a change of R_t profile!

    Parameters
    ----------
    initial_r
        (numeric) Value of the reproduction number at the beginning
        of the epidemic
    serial_interval
        (list) Unnormalised probability distribution of that the recipient
        first displays symptoms s days after the infector first displays
        symptoms
    phi
        (numeric) Value of the overdispersion parameter for the negative
        binomial noise distribution.

    """

    def __init__(self, initial_r, serial_interval, phi):
        super(NegBinBranchProModel, self).__init__(initial_r, serial_interval)

        if not isinstance(phi, (int, float)):
            raise TypeError(
                'Value of overdispersion must be integer or float.')
        if phi <= 0:
            raise ValueError(
                'Value of overdispersion must be must be > 0. For \
                overdispesion = 0, please use `BranchProModel` class type.')

        # Invert order of serial intervals for ease in _normalised_daily_mean
        self._serial_interval = np.asarray(serial_interval)[::-1]
        self._r_profile = np.array([initial_r])
        self._overdispersion = phi
        self._normalizing_const = np.sum(self._serial_interval)

    def set_overdispersion(self, phi):
        """
        Updates overdispersion noise parameter for the model.

        Parameters
        ----------
        phi
            New value of the overdispersion parameter for the negative
            binomial noise distribution.

        """
        if not isinstance(phi, (int, float)):
            raise TypeError(
                'Value of overdispersion must be integer or float.')
        if phi <= 0:
            raise ValueError(
                'Value of overdispersion must be must be > 0. For \
                overdispesion = 0, please use `BranchProModel` class type.')

        self._overdispersion = phi

    def get_overdispersion(self):
        """
        Returns overdispersion noise parameter for the model.

        """
        return self._overdispersion

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
            if norm_daily_mean != 0:
                incidences[t] = nbinom.rvs(
                    1/self._overdispersion,
                    float(1/self._overdispersion)/(1/self._overdispersion + norm_daily_mean),  # noqa
                    0,
                    1)
            else:
                incidences[t] = 0

        mask = np.in1d(np.append(np.asarray(0), simulation_times), times)
        return incidences[mask]


class LocImpNegBinBranchProModel(LocImpBranchProModel):
    r"""LocImpNegBinBranchProModel Class:
    Class for the models following a Branching Processes behaviour with
    local and imported cases and non-Poisson distribution noise.
    It inherits from the ``NegBinBranchProModel`` class.

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
    according to a negative binomial distribution with a mean that depends on
    previous number of cases both local and imported, according to the
    following formula:

    .. math::
        E(I_{t}^{\text(local)}|I_0, I_1, \dots I_{t-1}, w_{s}, R_{t}) =
            R_{t}^{\text(local)}\sum_{s=1}^{t}I_{t-s}^{\text(local)}w_{s} +
            R_{t}^{\text(imported)}\sum_{s=1}^{t}I_{t-s}^{\text(imported)}w_{s}

    The probability mass function of the incidence at time t thus becomes:

    .. math::
        P(I_{t}^{\text(local)}=k|I_0, I_1, \dots I_{t-1}, w_{s}, R_{t}) =
            \frac{\Gamma(k+\frac{1}{\phi})}{k!\Gamma(\frac{1}{\phi})}
            \Big(\frac{1}{1+\mu\phi}\Big)^{\frac{1}{\phi}}
            \Big(\frac{\mu\phi}{1+\mu\phi}\Big)^{k}

    where :math:`\mu` is mean of the distribution defined as above and
    :math:`\phi` is the overdispersion parameter associated with the negative
    binomial noise distribution.

    For the edge case, :math:`\phi = 0`, the incidence at time t becomes
    Poisson distributed, reducing to the simple class:`LocImpBranchProModel`
    class.

    Always apply methods :meth:`set_r_profile` and :meth:`set_imported_cases`
    before calling :meth:`LocImpNegBinBranchProModel.simulate` for a change of
    R_t profile and for loading the imported cases data!

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
        with respect to its analog for local ones
    phi
        (numeric) Value of the overdispersion parameter for the negative
        binomial noise distribution.

    """
    def __init__(self, initial_r, serial_interval, epsilon, phi):
        super().__init__(initial_r, serial_interval, epsilon)

        self.set_overdispersion(phi)

    def set_overdispersion(self, phi):
        """
        Updates overdispersion noise parameter for the model.

        Parameters
        ----------
        phi
            New value of the overdispersion parameter for the negative
            binomial noise distribution.

        """
        if not isinstance(phi, (int, float)):
            raise TypeError(
                'Value of overdispersion must be integer or float.')
        if phi <= 0:
            raise ValueError(
                'Value of overdispersion must be must be > 0. For \
                overdispesion = 0, please use `LocImpBranchProModel` class \
                type.')

        self._overdispersion = phi

    def get_overdispersion(self):
        """
        Returns overdispersion noise parameter for the model.

        """
        return self._overdispersion

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
            if norm_daily_mean != 0:
                incidences[t] = nbinom.rvs(
                    1/self._overdispersion,
                    float(1/self._overdispersion)/(1/self._overdispersion + norm_daily_mean),  # noqa
                    0,
                    1)
            else:
                incidences[t] = 0

        mask = np.in1d(np.append(np.asarray(0), simulation_times), times)
        return incidences[mask]
