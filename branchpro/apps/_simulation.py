#
# SimulationApp
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

import branchpro as bp


class IncidenceNumberSimulationApp:
    """IncidenceNumberSimulationApp Class:
    Class for the dash app with figure and sliders for the
    BranchPro models.
    """
    def __init__(self):
        self.app = dash.Dash(
            __name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.plot = bp.IncidenceNumberPlot()

        # Keeps traces visibility states fixed when changing sliders
        self.plot.figure['layout']['legend']['uirevision'] = True
        self.sliders = bp._SliderComponent()

        self.app.layout = dbc.Container(
            [
                html.H1('Branching Processes'),
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

    def add_simulator(self, simulator, init_cond=10.0, r0=2.0, r1=0.5):
        """
        Simulates an instance of a model, adds it as a line to the plot and
        adds sliders to the app.

        Parameters
        ----------
        simulator
            (SimulatorController) a BranchPro model and the time bounds
            between which you run the simulator.
        init_cond
            (int) start position on the slider for the number of initial
            cases for the Branch Pro model in the simulator.
        r0
            (float) start position on the slider for the initial reproduction
            number for the Branch Pro model in the simulator.
        r1
            (float) start position on the slider for the second reproduction
            number for the Branch Pro model in the simulator.
        """
        if not issubclass(type(simulator), bp.SimulationController):
            raise TypeError('Simulatior needs to be a SimulationController')

        model = simulator.model

        if not issubclass(type(model), bp.BranchProModel):
            raise TypeError('Models needs to be a BranchPro')

        bounds = simulator.get_time_bounds()
        mid_point = sum(bounds)/2

        self.sliders.add_slider(
            'Initial Cases', 'init_cond', init_cond, 0.0, 100.0, 10.0)
        self.sliders.add_slider('Initial R', 'r0', r0, 0.1, 10.0, 0.01)
        self.sliders.add_slider('second R', 'r1', r1, 0.1, 10.0, 0.01)
        self.sliders.add_slider(
            'Time of change', 't1', mid_point, bounds[0], bounds[1], 1)

        new_rs = [r0, r1]
        start_times = [0, mid_point]
        simulator.model.set_r_profile(new_rs, start_times)

        data = simulator.run(init_cond)
        df = pd.DataFrame({
            'Time': simulator.get_regime(),
            'Incidence Number': data})

        self.plot.add_simulation(df)

        self.simulator = simulator

        # Save the simulated figure for later update
        self._graph = self.plot.figure['data'][-1]

    def get_sliders_ids(self):
        """
        Returns the IDs of all sliders accompaning the figure in the
        app.
        """
        return self.sliders.slider_ids()

    def update_simulation(self, new_init_cond, new_r0, new_r1, new_t1):
        """
        Updates the model parameters in the simulator and the
        simulated graph in the figure.

        Parameters
        ----------
        new_init_cond
            (int) updated position on the slider for the number of initial
            cases for the Branch Pro model in the simulator.
        new_r0
            (float) updated position on the slider for the initial reproduction
            number for the Branch Pro model in the simulator.
        new_r1
            (float) updated position on the slider for the second reproduction
            number for the Branch Pro model in the simulator.
        new_t1
            (float) updated position on the slider for the time change in
            reproduction numbers for the Branch Pro model in the simulator.
        """
        new_rs = [new_r0, new_r1]
        start_times = [0, new_t1]

        model = self.simulator.model
        model.set_r_profile(new_rs, start_times)

        data = self.simulator.run(new_init_cond)
        self._graph['y'] = data

        return self.plot.figure
