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
import math


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
            \sum_{j}R_{t}C{i,j}T_{j}\sum_{s=1}^{t}I_{t-s,j}w_{s}

    Always apply method :meth:`set_r_profile` before calling
    :meth:`NegBinBranchProModel.simulate` for a change of R_t profile!

    Parameters
    ----------
    initial_r
        (list) List of reproduction numbers per category at the beginning
        of the epidemic.
    serial_interval
        (list) Unnormalised probability distribution of that the recipient
        first displays symptoms s days after the infector first displays
        symptoms for each category.
    num_cat
        (int) Number of categories in which the population is split.
    contact_matrix
        (array) Matrix of contacts between the different categories in which
        the population is split.
    transm
        (list) List of overall reductions in transmissibility per category.
    multipleSI
        (boolean) Different serial intervals used for categories.

    """

    def __init__(self, initial_r, serial_interval, num_cat, contact_matrix,
                 transm, multipleSI=False):
        if not isinstance(initial_r, (int, float)):
            raise TypeError('Value of R must be integer or float.')

        if not isinstance(num_cat, int):
            raise TypeError('Number of population categories must be integer.')
        if num_cat <= 0:
            raise ValueError('Number of population categories must be > 0.')

        if np.asarray(contact_matrix).ndim != 2:
            raise ValueError(
                'Contact matrix values storage format must be 2-dimensional')
        if np.asarray(contact_matrix).shape[0] != num_cat:
            raise ValueError(
                'Wrong number of rows in contact matrix values storage')
        if np.asarray(contact_matrix).shape[1] != num_cat:
            raise ValueError(
                'Wrong number of columns in contact matrix values storage')
        for c in np.asarray(contact_matrix):
            for _ in c:
                if _ < 0:
                    raise ValueError('Contact matrix values must be >= 0.')
                if not isinstance(_, (int, float)):
                    raise TypeError(
                        'Contact matrix values must be integer or float.')

        if np.asarray(transm).ndim != 1:
            raise ValueError(
                'Transmissiblity storage format must be 1-dimensional')
        if np.asarray(transm).shape[0] != num_cat:
            raise ValueError(
                'Wrong number of categories in transmissibility storage')
        for _ in transm:
            if _ < 0:
                raise ValueError('Transmissiblity values must be >= 0.')
            if not isinstance(_, (int, float)):
                raise TypeError(
                    'Transmissiblity values must be integer or float.')

        # Invert order of serial intervals for ease in _normalised_daily_mean
        self._num_cat = num_cat
        self._contact_matrix = np.asarray(contact_matrix)
        self._transm = np.asarray(transm)

        if multipleSI is False:
            if np.asarray(serial_interval).ndim != 1:
                raise ValueError(
                    'Serial interval values storage format must be\
                    1-dimensional')
            if np.sum(serial_interval) < 0:
                raise ValueError('Sum of serial interval values must be >= 0.')
            self._serial_interval = np.tile(
                np.asarray(serial_interval)[::-1], (num_cat, 1))
        else:
            if np.asarray(serial_interval).ndim != 2:
                raise ValueError(
                    'Serial interval values storage format must be\
                    2-dimensional')
            if np.asarray(serial_interval).shape[0] != num_cat:
                raise ValueError(
                    'Serial interval values storage format must match\
                    number of categories')
            for _ in range(num_cat):
                if np.sum(serial_interval[_, :]) < 0:
                    raise ValueError(
                        'Sum of serial interval values must be >= 0.')
            self._serial_interval = np.asarray(serial_interval)[:, ::-1]

        self._r_profile = np.array([initial_r])
        self._normalizing_const = np.sum(self._serial_interval, axis=1)

    def set_transmissibility(self, contact_matrix):
        """
        Updates contact matrix for the model.

        Parameters
        ----------
        contact_matrix
            New matrix of contacts between the different categories in which
            the population is split.

        """
        if np.asarray(contact_matrix).ndim != 2:
            raise ValueError(
                'Contact matrix values storage format must be 2-dimensional')
        if np.asarray(contact_matrix).shape[0] != self._num_cat:
            raise ValueError(
                'Wrong number of rows in contact matrix values storage')
        if np.asarray(contact_matrix).shape[1] != self._num_cat:
            raise ValueError(
                'Wrong number of columns in contact matrix values storage')
        for c in np.asarray(contact_matrix):
            for _ in c:
                if _ < 0:
                    raise ValueError('Contact matrix values must be >= 0.')
                if not isinstance(_, (int, float)):
                    raise TypeError(
                        'Contact matrix values must be integer or float.')

        self._contact_matrix = np.asarray(contact_matrix)

    def get_transmissibility(self):
        """
        Returns transmissibility vector for the model.

        """
        return self._transm

    def get_contact_matrix(self):
        """
        Returns contact matrix for the model.

        """
        return self._contact_matrix

    def get_serial_intervals(self):
        """
        Returns serial intervals for the model.

        """
        # Reverse inverting of order of serial intervals
        return self._serial_interval[:, ::-1]

    def set_serial_intervals(self, serial_intervals, multipleSI=False):
        """
        Updates serial intervals for the model.

        Parameters
        ----------
        serial_intervals
            New unnormalised probability distribution of that the recipient
            first displays symptoms s days after the infector first displays
            symptoms for each category.
        multipleSI
            (boolean) Different serial intervals used for categories.

        """
        # Invert order of serial intervals for ease in _effective_no_infectives
        if multipleSI is False:
            if np.asarray(serial_intervals).ndim != 1:
                raise ValueError(
                    'Serial interval values storage format must be\
                    1-dimensional')
            if np.sum(serial_intervals) < 0:
                raise ValueError('Sum of serial interval values must be >= 0.')
            self._serial_interval = np.tile(
                np.asarray(serial_intervals)[::-1], (self._num_cat, 1))
        else:
            if np.asarray(serial_intervals).ndim != 2:
                raise ValueError(
                    'Serial interval values storage format must be\
                    2-dimensional')
            if np.asarray(serial_intervals).shape[0] != self._num_cat:
                raise ValueError(
                    'Serial interval values storage format must match\
                    number of categories')
            for _ in range(self._num_cat):
                if np.sum(serial_intervals[_, :]) < 0:
                    raise ValueError(
                        'Sum of serial interval values must be >= 0.')
            self._serial_interval = np.asarray(serial_intervals)[:, ::-1]

        self._normalizing_const = np.sum(self._serial_interval, axis=1)

    def _effective_no_infectives(self, t, incidences, contact_matrix):
        """
        Computes expected number of new cases at time t, using previous
        incidences and serial intervals at a rate of 1:1 reproduction.

        Parameters
        ----------
        t
            evaluation time
        incidences
            sequence of incidence numbers
        contact_matrix
            matrix of contacts between categories
        """
        mean = np.zeros(self._num_cat)

        for i in range(self._num_cat):
            for j in range(self._num_cat):
                if t > self._serial_interval.shape[1]:
                    start_date = t - self._serial_interval.shape[1]
                    sub_sum = math.fsum(np.multiply(
                            self._serial_interval[j, :],
                            incidences[start_date:t, j]
                            )) / self._normalizing_const[j]
                else:
                    sub_sum = math.fsum(np.multiply(
                            self._serial_interval[j, -t:],
                            incidences[:t, j]
                            )) / self._normalizing_const[j]
                sub_sum *= contact_matrix[i, j] * self._transm[j]
                mean[i] += sub_sum

        return mean

    def simulate(
            self, parameters, times, var_contacts=False, neg_binom=False,
            niu=0.1):
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
        var_contacts
            (boolean) Wheteher there exists noise in number of contacts.
        neg_binom
            (boolean) Wheteher the noise in number of contacts is Negative
            Binomial distributed.
        niu
            (float) Accepance probability.

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

        incidences = np.empty(shape=(last_time_point + 1, self._num_cat))
        incidences[0, :] = initial_cond

        # Construct simulation times in steps of 1 unit time each
        simulation_times = np.arange(start=1, stop=last_time_point+1, step=1)

        # Compute normalised daily means for full timespan
        # and draw samples for the incidences
        self.exact_contact_matrix = [np.random.poisson(self._contact_matrix)]

        for t in simulation_times:
            if var_contacts is False:
                contact_matrix = self._contact_matrix
            else:
                if neg_binom is False:
                    contact_matrix = np.random.poisson(self._contact_matrix)
                else:
                    contact_matrix = np.random.negative_binomial(
                        self._contact_matrix, niu)
                self.exact_contact_matrix.append(contact_matrix)
            norm_daily_mean = self._r_profile[t-1] * \
                self._effective_no_infectives(t, incidences, contact_matrix)
            incidences[t, :] = np.random.poisson(
                lam=norm_daily_mean, size=self._num_cat)

        mask = np.in1d(np.append(np.asarray(0), simulation_times), times)
        return incidences[mask]


class LocImpMultiCatPoissonBranchProModel(MultiCatPoissonBranchProModel):
    r"""LocImpMultiCatPoissonBranchProModel Class:
    Class for the models following a Branching Processes behaviour with
    local and imported cases and Poisson distribution noise and multiple
    population categories. It inherits from the ``BranchProModel`` class.

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
        R_{t}^{\text(imported)} = \epsilon R_{t}^{\text(local)}

    The local incidence at time t is modelled by a random variable distributed
    according to a negative binomial distribution with a mean that depends on
    previous number of cases both local and imported, according to the
    following formula:

    .. math::
        E(I_{t, i}^{\text(local)}|I_0, I_1, \dots I_{t-1}, w_{s}, R_{t}) =
            \sum_{j}R_{t} C{i,j} T_j \sum_{s=1}^{t}I_{t-s,j}w_{s}  +
            R_{t}^{\text(imported)}C{i,j}T_{j}\sum_{s=1}^{t}
            I_{t-s}^{\text(imported)}w_{s}

    The total reproduction number at time t is computed as follows:

    Always apply method :meth:`set_r_profile` before calling
    :meth:`NegBinBranchProModel.simulate` for a change of R_t profile!

    Parameters
    ----------
    initial_r
        (list) List of reproduction numbers per category at the beginning
        of the epidemic.
    serial_interval
        (list) Unnormalised probability distribution of that the recipient
        first displays symptoms s days after the infector first displays
        symptoms for each category.
    epsilon
        (numeric) Proportionality constant of the R number for imported cases
        with respect to its analog for local ones.
    num_cat
        (int) Number of categories in which the population is split.
    contact_matrix
        (array) Matrix of contacts between the different categories in which
        the population is split.
    transm
        (list) List of overall reductions in transmissibility per category.

    """

    def __init__(self, initial_r, serial_interval, epsilon, num_cat,
                 contact_matrix, transm):
        super().__init__(
            initial_r, serial_interval, num_cat, contact_matrix, transm)

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
            number of imported cases per category at that specified point.

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

        # Raise error if inputs do not have same shape
        if np.asarray(times).shape[0] != np.asarray(cases).shape[0]:
            raise ValueError('Both inputs should have same number of elements')

        self._imported_times = np.asarray(times, dtype=int)
        self._imported_cases = np.asarray(cases)

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
        var_contacts
            (boolean) Wheteher there exists noise in number of contacts.

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

        incidences = np.empty(shape=(last_time_point + 1, self._num_cat))
        incidences[0, :] = initial_cond

        # Create vector of imported cases
        imported_incidences = np.zeros(
            shape=(last_time_point + 1, self._num_cat), dtype=int)

        imports_times = self._imported_times[
            self._imported_times <= last_time_point]
        imports_cases = self._imported_cases[imports_times, :]

        np.put(
            imported_incidences, ind=imports_times, v=imports_cases)

        # Construct simulation times in steps of 1 unit time each
        simulation_times = np.arange(start=1, stop=last_time_point+1, step=1)

        # Compute normalised daily means for full timespan
        # and draw samples for the incidences
        for t in simulation_times:
            norm_daily_mean = self._r_profile[t-1] * np.matmul(
                self._contact_matrix,
                np.multiply(
                    self._transm,
                    self._effective_no_infectives(t, incidences) +
                    self.epsilon * self._effective_no_infectives(
                        t, imported_incidences)
                ))
            incidences[t, :] = np.random.poisson(
                lam=norm_daily_mean, size=self._num_cat)

        mask = np.in1d(np.append(np.asarray(0), simulation_times), times)
        return incidences[mask]


class AggMultiCatPoissonBranchProModel(BranchProModel):
    r"""AggMultiCatPoissonBranchProModel Class:
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
        E(I_{t}^{\text(local)}|I_0, I_1, \dots I_{t-1}, w_{s}, R_{t}) =
            R_{t}\sum_{s=1}^{t}(\sum_{j}C{i,j}T_{j}p_{t-s,j})I_{t-s}w_{s}

    Always apply method :meth:`set_r_profile` before calling
    :meth:`NegBinBranchProModel.simulate` for a change of R_t profile!

    Parameters
    ----------
    initial_r
        (list) List of reproduction numbers per category at the beginning
        of the epidemic.
    serial_interval
        (list) Unnormalised probability distribution of that the recipient
        first displays symptoms s days after the infector first displays
        symptoms.
    num_cat
        (int) Number of categories in which the population is split.
    contact_matrix
        (array) Matrix of contacts between the different categories in which
        the population is split.
    transm
        (list) List of overall reductions in transmissibility per category.

    """

    def __init__(self, initial_r, serial_interval, num_cat, contact_matrix,
                 transm, p):
        if np.asarray(serial_interval).ndim != 1:
            raise ValueError(
                'Serial interval values storage format must be 1-dimensional')
        if np.sum(serial_interval) < 0:
            raise ValueError('Sum of serial interval values must be >= 0.')
        if not isinstance(initial_r, (int, float)):
            raise TypeError('Value of R must be integer or float.')

        if not isinstance(num_cat, int):
            raise TypeError('Number of population categories must be integer.')
        if num_cat <= 0:
            raise ValueError('Number of population categories must be > 0.')

        if np.asarray(contact_matrix).ndim != 2:
            raise ValueError(
                'Contact matrix values storage format must be 2-dimensional')
        if np.asarray(contact_matrix).shape[0] != num_cat:
            raise ValueError(
                'Wrong number of rows in contact matrix values storage')
        if np.asarray(contact_matrix).shape[1] != num_cat:
            raise ValueError(
                'Wrong number of columns in contact matrix values storage')
        for c in np.asarray(contact_matrix):
            for _ in c:
                if _ < 0:
                    raise ValueError('Contact matrix values must be >= 0.')
                if not isinstance(_, (int, float)):
                    raise TypeError(
                        'Contact matrix values must be integer or float.')

        if np.asarray(transm).ndim != 1:
            raise ValueError(
                'Transmissiblity storage format must be 1-dimensional')
        if np.asarray(transm).shape[0] != num_cat:
            raise ValueError(
                'Wrong number of categories in transmissibility storage')
        for _ in transm:
            if _ < 0:
                raise ValueError('Transmissiblity values must be >= 0.')
            if not isinstance(_, (int, float)):
                raise TypeError(
                    'Transmissiblity values must be integer or float.')

        if np.asarray(transm).ndim != 1:
            raise ValueError(
                'Transmissiblity storage format must be 1-dimensional')
        if np.asarray(transm).shape[0] != num_cat:
            raise ValueError(
                'Wrong number of categories in transmissibility storage')
        for _ in transm:
            if _ < 0:
                raise ValueError('Transmissiblity values must be >= 0.')
            if not isinstance(_, (int, float)):
                raise TypeError(
                    'Transmissiblity values must be integer or float.')

        # Invert order of serial intervals for ease in _normalised_daily_mean
        self._num_cat = num_cat
        self._contact_matrix = np.asarray(contact_matrix)
        self._transm = np.asarray(transm)
        self._serial_interval = np.asarray(serial_interval)[::-1]
        self._r_profile = np.array([initial_r])
        self._normalizing_const = np.sum(self._serial_interval)

    def set_transmissibility(self, contact_matrix):
        """
        Updates contact matrix for the model.

        Parameters
        ----------
        contact_matrix
            New matrix of contacts between the different categories in which
            the population is split.

        """
        if np.asarray(contact_matrix).ndim != 2:
            raise ValueError(
                'Contact matrix values storage format must be 2-dimensional')
        if np.asarray(contact_matrix).shape[0] != self._num_cat:
            raise ValueError(
                'Wrong number of rows in contact matrix values storage')
        if np.asarray(contact_matrix).shape[1] != self._num_cat:
            raise ValueError(
                'Wrong number of columns in contact matrix values storage')
        for c in np.asarray(contact_matrix):
            for _ in c:
                if _ < 0:
                    raise ValueError('Contact matrix values must be >= 0.')
                if not isinstance(_, (int, float)):
                    raise TypeError(
                        'Contact matrix values must be integer or float.')

        self._contact_matrix = np.asarray(contact_matrix)

    def get_transmissibility(self):
        """
        Returns transmissibility vector for the model.

        """
        return self._transm

    def get_contact_matrix(self):
        """
        Returns contact matrix for the model.

        """
        return self._contact_matrix

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

    def simulate(self, parameters, times, p):
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
        p
            (list) List of time-dependent proprortions of the total infections
            that day in each category.
        var_contacts
            (boolean) Wheteher there exists noise in number of contacts.

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

        incidences = np.empty(shape=(last_time_point + 1, self._num_cat))
        incidences[0, :] = initial_cond

        # Construct simulation times in steps of 1 unit time each
        simulation_times = np.arange(start=1, stop=last_time_point+1, step=1)

        # Compute normalised daily means for full timespan
        # and draw samples for the incidences
        for t in simulation_times:
            norm_daily_mean = self._r_profile[t-1] * np.matmul(
                self._contact_matrix,
                np.multiply(
                    self._transm,
                    self._effective_no_infectives(t, incidences)
                ))
            incidences[t, :] = np.random.poisson(
                lam=norm_daily_mean, size=self._num_cat)

        mask = np.in1d(np.append(np.asarray(0), simulation_times), times)
        return incidences[mask]


class LocImpAggMultiCatPoissonBranchProModel(MultiCatPoissonBranchProModel):
    r"""LocImpAggMultiCatPoissonBranchProModel Class:
    Class for the models following a Branching Processes behaviour with
    local and imported cases and Poisson distribution noise and multiple
    population categories. It inherits from the ``BranchProModel`` class.

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
        R_{t}^{\text(imported)} = \epsilon R_{t}^{\text(local)}

    The local incidence at time t is modelled by a random variable distributed
    according to a negative binomial distribution with a mean that depends on
    previous number of cases both local and imported, according to the
    following formula:

    .. math::
        E(I_{t, i}^{\text(local)}|I_0, I_1, \dots I_{t-1}, w_{s}, R_{t}) =
            \sum_{j}R_{t} C{i,j} T_j \sum_{s=1}^{t}I_{t-s,j}w_{s}  +
            R_{t}^{\text(imported)}C{i,j}T_{j}\sum_{s=1}^{t}
            I_{t-s}^{\text(imported)}w_{s}

    The total reproduction number at time t is computed as follows:

    Always apply method :meth:`set_r_profile` before calling
    :meth:`NegBinBranchProModel.simulate` for a change of R_t profile!

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
    contact_matrix
        (array) Matrix of contacts between the different categories in which
        the population is split.
    transm
        (list) List of overall reductions in transmissibility per category.

    """

    def __init__(self, initial_r, serial_interval, epsilon, num_cat,
                 contact_matrix, transm):
        super().__init__(
            initial_r, serial_interval, num_cat, contact_matrix, transm)

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
            number of imported cases per category at that specified point.

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

        # Raise error if inputs do not have same shape
        if np.asarray(times).shape[0] != np.asarray(cases).shape[0]:
            raise ValueError('Both inputs should have same number of elements')

        self._imported_times = np.asarray(times, dtype=int)
        self._imported_cases = np.asarray(cases)

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
        var_contacts
            (boolean) Wheteher there exists noise in number of contacts.

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

        incidences = np.empty(shape=(last_time_point + 1, self._num_cat))
        incidences[0, :] = initial_cond

        # Create vector of imported cases
        imported_incidences = np.zeros(
            shape=(last_time_point + 1, self._num_cat), dtype=int)

        imports_times = self._imported_times[
            self._imported_times <= last_time_point]
        imports_cases = self._imported_cases[imports_times, :]

        np.put(
            imported_incidences, ind=imports_times, v=imports_cases)

        # Construct simulation times in steps of 1 unit time each
        simulation_times = np.arange(start=1, stop=last_time_point+1, step=1)

        # Compute normalised daily means for full timespan
        # and draw samples for the incidences
        for t in simulation_times:
            norm_daily_mean = self._r_profile[t-1] * np.matmul(
                self._contact_matrix,
                np.multiply(
                    self._transm,
                    self._effective_no_infectives(t, incidences) +
                    self.epsilon * self._effective_no_infectives(
                        t, imported_incidences)
                ))
            incidences[t, :] = np.random.poisson(
                lam=norm_daily_mean, size=self._num_cat)

        mask = np.in1d(np.append(np.asarray(0), simulation_times), times)
        return incidences[mask]
