#
# IncidenceNumberPlot Class
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import warnings
import pandas as pd
import plotly.graph_objs as go


class IncidenceNumberPlot():
    """IncidenceNumberPlot Class
    Stores the main figure for the Dash app.
    """
    def __init__(self):
        self.figure = go.Figure()

    def _label_warning(self, time_key, inc_key):
        x_label = self.figure['layout']['xaxis']['title']['text']
        y_label = self.figure['layout']['yaxis']['title']['text']
        if (x_label is not None) and (y_label is not None):
            if (x_label != time_key) or (y_label != inc_key):
                warnings.warn('Labels do not match. They will be updated.')

    def add_data(self, df, time_key='Time', inc_key='Incidence Number'):
        """
        Supplies data to the figure which will be used for the bar plot.

        Parameters
        ----------
        df
            (pandas DataFrame) contains numbers of new cases by days.
            Data stored in columns 'Time' and 'Incidence Number', respectively.
        time_key
            x-axis label for the bar plot.
        inc_key
            y-axis label for the bar plot.
        """
        if not issubclass(type(df), pd.DataFrame):
            raise TypeError('df needs to be a dataframe')
        self._label_warning(time_key, inc_key)

        trace = go.Bar(
            y=df[inc_key],
            x=df[time_key],
            name='Cases'
        )

        self.figure.add_trace(trace)
        self.figure.update_layout(
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
            Data stored in columns 'Time' and 'Incidence Number', respectively.
        time_key
            x-axis label for the bar plot.
        inc_key
            y-axis label for the bar plot.
        """
        if not issubclass(type(df), pd.DataFrame):
            raise TypeError('df needs to be a dataframe')
        self._label_warning(time_key, inc_key)

        trace = go.Scatter(
            y=df[inc_key],
            x=df[time_key],
            mode='lines',
            name='Simulation'
        )

        self.figure.add_trace(trace)
        self.figure.update_layout(
            xaxis_title=time_key,
            yaxis_title=inc_key,
            hovermode='x unified')

    def update_labels(self, time_label=None, inc_label=None):
        """
        Updates the figure labels with user inputed values.

        Parameters
        ----------
        time_label
            x-axis label for the bar plot.
        inc_label
            y-axis label for the bar plot.
        """
        if time_label is not None:
            self.figure.update_layout(xaxis_title=time_label)

        if inc_label is not None:
            self.figure.update_layout(yaxis_title=inc_label)

    def show_figure(self):
        """
        Shows current figure.
        """
        self.figure.show()
