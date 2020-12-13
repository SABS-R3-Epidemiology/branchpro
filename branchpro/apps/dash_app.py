#
# Dash app
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import pandas as pd
import dash
import dash_bootstrap_components as dbc

import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output
import branchpro as bp

import numpy as np


class IncidenceNumberSimulationApp:
    """IncidenceNumberSimulationApp Class:
    Class for the dash app with figure and sliders.
    """
    def __init__(self):
        self.app = dash.Dash(
            __name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.plot = bp.IncidenceNumberPlot()
        self.sliders = bp._SliderComponent()

        self.app.layout = dbc.Container(
            [
                html.H1('Example Title'),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(
                            figure=self.plot.figure, id='myfig')),
                        dbc.Col(self.sliders.get_sliders_div())
                    ],
                    align='center',
                ),
            ],
            fluid=True,
        )

    def add_data(self, df):
        """
        Adds incidence data to the plot in the dash app.

        Parameters
        ----------
        df
            (pandas DataFrame) contains numbers of new cases by days.
            Data stored in columns 'Time' and 'Incidence Number', respectively.
        """
        self.plot.add_data(df)

    def add_simulator(self, simulator):
        """
        Simulates an instance of a model, adds it as a line to the plot and
        adds sliders to the app.

        Parameters
        ----------
        simulator
            (SimulatorController) a BranchPro model and the time bounds
            between which you run the simulator.
        """
        if not issubclass(type(simulator), bp.SimulationController):
            raise TypeError('Simulatior needs to be a SimulationController')

        model = simulator.model

        if not issubclass(type(model), bp.BranchProModel):
            raise TypeError('Models needs to be a BranchPro')

        bounds = simulator.get_time_bounds()
        mid_point = sum(bounds)/2

        self.sliders.add_slider(
            'Initial Cases', 'init_cond', 10.0, 0.0, 100.0, 10.0)
        self.sliders.add_slider('Initial R', 'r0', 2.0, 0.1, 10.0, 0.01)
        self.sliders.add_slider('second R', 'r1', 0.5, 0.1, 10.0, 0.01)
        self.sliders.add_slider(
            'Time of change', 't1', mid_point, bounds[0], bounds[1], 1)

        new_rs = [2.0, 0.5]
        start_times = [0, mid_point]
        simulator.model.set_r_profile(new_rs, start_times)

        data = simulator.run(10)
        df = pd.DataFrame({
            'Time': simulator.get_regime(),
            'Incidence Number': data})

        self.plot.add_simulation(df)

        self.simulator = simulator

    def get_sliders_ids(self):
        """
        Returns the IDs aof all sliders accompaning the figure in the
        app.
        """
        return self.sliders.slider_ids()

    def update_simulation(self, new_init_cond, new_r0, new_r1, new_t1):
        """
        Updates the model parameters in the simulator and the
        simulated graph in the figure.
        """
        new_rs = [new_r0, new_r1]
        start_times = [0, new_t1]

        model = self.simulator.model
        model.set_r_profile(new_rs, start_times)

        data = self.simulator.run(new_init_cond)
        fig = self.plot.figure
        fig['data'][1]['y'] = data

        return fig


app = IncidenceNumberSimulationApp()
df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })

br_pro_model = bp.BranchProModel(2, np.array([1, 2, 3, 2, 1]))
simulationController = bp.SimulationController(br_pro_model, 1, 7)
app.add_simulator(simulationController)

sliders = app.get_sliders_ids()


@app.app.callback(
        Output('myfig', 'figure'),
        [Input(s, 'value') for s in sliders])
def update_simulation(*args):
    """
    Simulates the model for the current slider values and updates the
    plot in the figure.
    """
    parameters = args
    fig = app.update_simulation(*parameters)

    return fig


app.app.run_server(debug=True)
