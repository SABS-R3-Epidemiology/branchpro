#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import datetime
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import scipy.stats
import branchpro


class TestPlotForwardSimulations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the class using real data and inference.

        The purpose is to test the figure function under realistic conditions.
        """
        # Make a fake set of serial intervals
        serial_intervals = np.array([[0.1, 0.5, 0.2, 0.2],
                                     [0.2, 0.4, 0.2, 0.2],
                                     [0.1, 0.5, 0.4, 0.0]])

        # Read Ontario data
        data = pd.read_csv(
            '../branchpro/branchpro/data_library/covid_ontario/ON.csv')[:51]

        locally_infected_cases = data['Incidence Number']
        imported_cases = data['Imported Cases']

        # Get time points
        num_timepoints = max(data['Time'])
        start_times = np.arange(1, num_timepoints+1, dtype=int)
        times = np.arange(num_timepoints+1)

        # Inference R_t profile, but using the LocImpBranchProPosterior
        tau = 2
        R_t_start = tau+1
        a = 1
        b = 0.8

        # Transform our incidence data into pandas dataframes
        inc_data = pd.DataFrame(
            {
                'Time': start_times,
                'Incidence Number': locally_infected_cases
            }
        )

        imported_inc_data = pd.DataFrame(
            {
                'Time': start_times,
                'Incidence Number': imported_cases
            }
        )

        # Run inference with epsilon = 1 to get R trajectory
        inference = branchpro.LocImpBranchProPosteriorMultSI(
            inc_data=inc_data,
            imported_inc_data=imported_inc_data,
            epsilon=1,
            daily_serial_intervals=serial_intervals,
            alpha=a,
            beta=1/b)
        inference.run_inference(tau=tau)
        intervals = inference.get_intervals(central_prob=.95)
        new_rs = intervals['Mean'].values.tolist()

        # Run simulation of local cases for various values of epsilon
        epsilon_values = [0.25, 1.0, 1.5]
        initial_r = new_rs[0]
        si = np.median(serial_intervals, axis=0)
        parameters = 0  # initial number of cases
        all_local_cases = []
        num_simulations = 100
        samples = []
        central_prob = 0.95
        for _, epsilon in enumerate(epsilon_values):
            m = branchpro.LocImpBranchProModel(initial_r, si, epsilon)
            m.set_r_profile(new_rs, start_times[R_t_start:])
            m.set_imported_cases(start_times, imported_cases)

            for sim in range(num_simulations):
                samples.append(m.simulate(parameters, times))

            simulation_estimates = np.mean(np.vstack(samples), axis=0)
            lb = 100*(1-central_prob)/2
            ub = 100*(1+central_prob)/2
            simulation_interval = np.percentile(
                np.vstack(samples), q=np.array([lb, ub]), axis=0)
            simulation_df = pd.DataFrame(
                {
                    'Mean': simulation_estimates,
                    'Lower bound CI': simulation_interval[0],
                    'Upper bound CI': simulation_interval[1]
                }
            )
            all_local_cases.append(simulation_df)

        cls.imported_cases = imported_cases
        cls.start_times = start_times
        cls.R_t_start = R_t_start
        cls.new_rs = new_rs
        cls.epsilon_values = epsilon_values
        cls.all_local_cases = all_local_cases

    def test_plot(self):
        # Test plotting the figure
        with patch('matplotlib.pyplot.show') as _:
            fig = branchpro.figures.plot_forward_simulations(
                self.imported_cases.values,
                self.start_times[self.R_t_start:],
                self.new_rs,
                self.epsilon_values,
                self.all_local_cases,
                datetime.datetime(2020, 3, 1),
                show=True)

        # Check that all plots are present
        assert len(fig.axes) == 3


class TestPlotRInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the class using real data and inference.

        The purpose is to test the figure function under realistic conditions.
        """
        # Make a fake set of serial intervals
        serial_intervals = np.array([[0.1, 0.5, 0.2, 0.2],
                                     [0.2, 0.4, 0.2, 0.2],
                                     [0.1, 0.5, 0.4, 0.0]])

        # Read Ontario data
        data = pd.read_csv(
            '../branchpro/branchpro/data_library/covid_ontario/ON.csv')[:51]

        locally_infected_cases = data['Incidence Number']
        imported_cases = data['Imported Cases']

        # Get time points from the data
        num_timepoints = max(data['Time'])
        start_times = np.arange(1, num_timepoints+1, dtype=int)

        # Same inference, but using the LocImpBranchProPosterior
        tau = 6
        a = 1
        b = 0.2

        # Run inferences for different values of epsilon
        column_names = ['Time Points',
                        'Mean',
                        'Lower bound CI',
                        'Upper bound CI',
                        'Central Probability',
                        'Epsilon']
        epsilon_range = [0.2, 0.5, 1.0, 1.5, 2.0]
        all_intervals = pd.DataFrame(columns=column_names)

        # Transform our incidence data into pandas dataframes
        inc_data = pd.DataFrame(
            {
                'Time': start_times,
                'Incidence Number': locally_infected_cases
            }
        )
        imported_inc_data = pd.DataFrame(
            {
                'Time': start_times,
                'Incidence Number': imported_cases
            }
        )

        for epsilon in epsilon_range:
            inference = branchpro.LocImpBranchProPosteriorMultSI(
                inc_data=inc_data,
                imported_inc_data=imported_inc_data,
                epsilon=epsilon,
                daily_serial_intervals=serial_intervals,
                alpha=a,
                beta=b)

            inference.run_inference(tau=tau)
            intervals = inference.get_intervals(central_prob=.95)
            intervals['Epsilon'] = [epsilon] * len(intervals.index)
            all_intervals = all_intervals.append(intervals)

        prior_dist = scipy.stats.gamma(a, scale=1/b)
        median = prior_dist.median()

        cls.locally_infected_cases = locally_infected_cases
        cls.imported_cases = imported_cases
        cls.epsilon_range = epsilon_range
        cls.prior_median = median
        cls.all_intervals = all_intervals

    def test_plot(self):
        # Test plotting the figure
        with patch('matplotlib.pyplot.show') as _:
            fig = branchpro.figures.plot_r_inference(
                datetime.datetime(2020, 3, 1),
                self.locally_infected_cases,
                self.imported_cases,
                datetime.datetime(2020, 3, 7),
                self.epsilon_range,
                [self.all_intervals.loc[self.all_intervals['Epsilon'] == e]
                    for e in self.epsilon_range],
                self.prior_median,
                default_epsilon=1,
                show=True)

        # Check that all plots are present
        assert len(fig.axes) == 5


class TestPlotRegionsInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the class using real data and inference.

        The purpose is to test the figure function under realistic conditions.
        """
        # Make a fake set of serial intervals
        serial_intervals = np.array([[0.1, 0.5, 0.2, 0.2],
                                     [0.2, 0.4, 0.2, 0.2],
                                     [0.1, 0.5, 0.4, 0.0]])

        # Read Ontario data
        data = pd.read_csv(
            '../branchpro/branchpro/data_library/covid_ontario/ON.csv')[:51]

        locally_infected_cases = data['Incidence Number']
        imported_cases = data['Imported Cases']

        # Get time points from the data
        num_timepoints = max(data['Time'])
        start_times = np.arange(1, num_timepoints+1, dtype=int)

        # Same inference, but using the LocImpBranchProPosterior
        tau = 6
        a = 1
        b = 0.2

        # Run inferences for different values of epsilon
        column_names = ['Time Points',
                        'Mean',
                        'Lower bound CI',
                        'Upper bound CI',
                        'Central Probability',
                        'Epsilon']
        epsilon_range = [0.25, 1, 2]
        all_intervals = pd.DataFrame(columns=column_names)

        # Transform our incidence data into pandas dataframes
        inc_data = pd.DataFrame(
            {
                'Time': start_times,
                'Incidence Number': locally_infected_cases
            }
        )
        imported_inc_data = pd.DataFrame(
            {
                'Time': start_times,
                'Incidence Number': imported_cases
            }
        )

        for epsilon in epsilon_range:
            inference = branchpro.LocImpBranchProPosteriorMultSI(
                inc_data=inc_data,
                imported_inc_data=imported_inc_data,
                epsilon=epsilon,
                daily_serial_intervals=serial_intervals,
                alpha=a,
                beta=b)

            inference.run_inference(tau=tau)
            intervals = inference.get_intervals(central_prob=.95)
            intervals['Epsilon'] = [epsilon] * len(intervals.index)
            all_intervals = all_intervals.append(intervals)

        cls.locally_infected_cases = locally_infected_cases
        cls.imported_cases = imported_cases
        cls.epsilon_range = epsilon_range
        cls.all_intervals = all_intervals

    def test_plot(self):
        # Test plotting the figure
        with patch('matplotlib.pyplot.show') as _:
            fig = branchpro.figures.plot_regions_inference(
                datetime.datetime(2020, 3, 1),
                ['Ontario', 'Ontario'],
                [self.locally_infected_cases, self.locally_infected_cases],
                [self.imported_cases, self.imported_cases],
                datetime.datetime(2020, 3, 7),
                self.epsilon_range,
                [[self.all_intervals.loc[self.all_intervals['Epsilon'] == e]
                    for e in self.epsilon_range],
                 [self.all_intervals.loc[self.all_intervals['Epsilon'] == e]
                    for e in self.epsilon_range]],
                default_epsilon=1,
                inset_region=['Ontario'],
                show=True)

        # Check that all plots are present
        assert len(fig.axes) == 4

        # Test plotting the figure with mers
        with patch('matplotlib.pyplot.show') as _:
            fig = branchpro.figures.plot_regions_inference(
                datetime.datetime(2020, 3, 1),
                ['Ontario', 'Ontario'],
                [self.locally_infected_cases, self.locally_infected_cases],
                [self.imported_cases, self.imported_cases],
                datetime.datetime(2020, 3, 7),
                self.epsilon_range,
                [[self.all_intervals.loc[self.all_intervals['Epsilon'] == e]
                    for e in self.epsilon_range],
                 [self.all_intervals.loc[self.all_intervals['Epsilon'] == e]
                    for e in self.epsilon_range]],
                default_epsilon=1,
                inset_region=['Ontario'],
                show=True,
                mers=True)

        # Check that all plots are present
        assert len(fig.axes) == 4


class TestPlotHeatmap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the class using real data and inference.

        The purpose is to test the figure function under realistic conditions.
        """
        # Make a fake set of serial intervals
        serial_intervals = np.array([[0.1, 0.5, 0.2, 0.2],
                                     [0.2, 0.4, 0.2, 0.2],
                                     [0.1, 0.5, 0.4, 0.0]])

        # Read Ontario data
        data = pd.read_csv(
            '../branchpro/branchpro/data_library/covid_ontario/ON.csv')[:51]

        locally_infected_cases = data['Incidence Number']
        imported_cases = data['Imported Cases']

        # Get time points from the data
        num_timepoints = max(data['Time'])
        start_times = np.arange(1, num_timepoints+1, dtype=int)

        # Same inference, but using the LocImpBranchProPosterior
        tau = 6
        a = 1
        b = 0.2

        # Run inferences for different values of epsilon
        column_names = ['Time Points',
                        'Mean',
                        'Lower bound CI',
                        'Upper bound CI',
                        'Central Probability',
                        'Epsilon']
        epsilon_range = [0.1 + 0.1 * i for i in range(30)]
        all_intervals = pd.DataFrame(columns=column_names)

        # Transform our incidence data into pandas dataframes
        inc_data = pd.DataFrame(
            {
                'Time': start_times,
                'Incidence Number': locally_infected_cases
            }
        )
        imported_inc_data = pd.DataFrame(
            {
                'Time': start_times,
                'Incidence Number': imported_cases
            }
        )

        for epsilon in epsilon_range:
            inference = branchpro.LocImpBranchProPosteriorMultSI(
                inc_data=inc_data,
                imported_inc_data=imported_inc_data,
                epsilon=epsilon,
                daily_serial_intervals=serial_intervals,
                alpha=a,
                beta=b)

            inference.run_inference(tau=tau)
            intervals = inference.get_intervals(central_prob=.95)
            intervals['Epsilon'] = [epsilon] * len(intervals.index)
            all_intervals = all_intervals.append(intervals)

        cls.epsilon_range = epsilon_range
        cls.all_intervals = all_intervals

    def test_plot(self):
        # Test plotting the figure
        with patch('matplotlib.pyplot.show') as _:
            fig = branchpro.figures.plot_r_heatmap(
                ['Ontario', 'Ontario'],
                self.epsilon_range,
                [self.all_intervals, self.all_intervals],
                datetime.datetime(2020, 3, 8),
                show=True
                )

        # Check that all plots are present
        # There should be 2 (one for each region) plus 1 for the R colorbar
        assert len(fig.axes) == 3
