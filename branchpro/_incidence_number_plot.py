#
# IncidenceNumberPlot Class
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import pandas as pd
import plotly.graph_objs as go


class IncidenceNumberPlot():
    """IncidenceNumberPlot Class
    Stores the main figure for the Dash app.
    """
    def __init__(self):
        self._figure = go.Figure()

    def add_data(self, df, time_key='Time', inc_key='Incidence Number'):
        """
        Supplies data to the figure which will be used for the bar plot.

        Parameters
        ----------
        df
            (pandas DataFrame) contains numbers of new cases by days.
        time_key
            x-axis label for the bar plot.
        inc_key
            y-axis label for the bar plot.
        """
        if type(df) != pd.DataFrame:
            raise TypeError('Data needs to be a dataframe')

        trace = go.Bar(
            y=df[inc_key],
            x=df[time_key],
            name='cases'
        )

        self._figure.add_trace(trace)
        self._figure.update_layout(
            xaxis_title=time_key,
            yaxis_title=inc_key)

    def add_simulation(self, df, time_key='Time', inc_key='Incidence Number'):
        """
        Supplies simulated data to the figure which will be added the bar plot
        as lines and used for comparison with observed data.

        Parameters
        ----------
        df
            (pandas DataFrame) contains numbers of new cases by days.
        time_key
            x-axis label for the bar plot.
        inc_key
            y-axis label for the bar plot.
        """
        if type(df) != pd.DataFrame:
            raise TypeError('Simulation needs to be a dataframe')

        trace = go.Scatter(
            y=df[inc_key],
            x=df[time_key],
            mode='lines',
            name='simulation'
        )

        self._figure.add_trace(trace)
        self._figure.update_layout(
            xaxis_title=time_key,
            yaxis_title=inc_key)

    def show_figure(self):
        """
        Shows current figure object in the Dash app.
        """
        self._figure.show()
