#
# Dash app
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import pandas
import dash
import dash_bootstrap_components as dbc
 
import dash_core_components as dcc
import dash_html_components as html
 
from dash.dependencies import Input, Output
import branchpro as bp
 
import scipy.stats
import numpy as np

class IncidenceNumberSimulationApp:
 
    def __init__(self):
 
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.plot = bp.IncidenceNumberPlot()
        self.sliders = bp._SliderComponent()

        self.app.layout = dbc.Container(
            [
                html.H1('Example Title'),
                dbc.Row(
                    [dbc.Col(dcc.Graph(figure=self.plot.figure, id='myfig')),
                    dbc.Col(self.sliders.get_sliders_div())],
                    align='center',
                ),
            ],
            fluid=True,
        )
    
    def add_data(self, df):
        self.plot.add_data(df)

    def add_simulator(self, simulator):
        if not issubclass(type(simulator), bp.SimulationController):
            raise TypeError('Simulatior needs to be a SimulationController')

        model = simulator.model

        if not issubclass(type(model), bp.BranchProModel):
            raise TypeError('Models needs to be a BranchPro')
    

 
        self.sliders.add_slider('Initial R', 'init_r', 2.0, 0.1, 10.0, 0.01)
        self.sliders.add_slider('second R', 'two_r', 2.0, 0.1, 10.0, 0.01)
        self.sliders.add_slider('Time of change', 't', 2.5, 0.1, 5.0, 0.01)
 
        num_timepoints = 30
        data = model.simulate(1, np.arange(1, num_timepoints+1))
        df = pandas.DataFrame({'Time': np.arange(1, num_timepoints+1),
                               'Incidence Number': data})
 
        self.plot.add_simulation(df)
 
        self.model = model