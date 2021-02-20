#
# Dash app
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#
"""This is an app which shows forward simulation of the branching process model
with fixed example data. To run the app, use ``python dash_app.py``.
"""

import os

import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import branchpro as bp
from branchpro.apps import IncidenceNumberSimulationApp


app = IncidenceNumberSimulationApp()
full_ffd = bp.DatasetLibrary().french_flu()
small_ffd = full_ffd.query('year == 2020')
df = pd.DataFrame({
            'Weeks': small_ffd['week'],
            'Incidence Number': small_ffd['inc']
        })

french_flu_data = df

br_pro_model = bp.BranchProModel(2, np.array([1, 2, 3, 2, 1]))
simulationController = bp.SimulationController(
    br_pro_model, 1, len(small_ffd['week']))
app.add_simulator(
    simulationController,
    magnitude_init_cond=max(df['Incidence Number']))
app.add_data(df, time_label='Weeks')

sliders = app.get_sliders_ids()

# Add the explanation texts
fname = os.path.join(os.path.dirname(__file__), 'data', 'dash_app_text.md')
with open(fname) as f:
    app.add_text(dcc.Markdown(f.read(), dangerously_allow_html=True))

fname = os.path.join(os.path.dirname(__file__),
                     'data',
                     'dash_app_collapsible_text.md')
with open(fname) as f:
    app.add_collapsed_text(dcc.Markdown(f.read(), dangerously_allow_html=True))

# Get server of the app; necessary for correct deployment of the app.
server = app.app.server


@app.app.callback(
    Output('incidence-data-upload', 'children'),
    Output('data_storage', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
)
def load_data(*args):
    """Load data from a file and save it in storage.
    """
    list_contents, list_names = args

    if list_contents is not None:
        # Run content parser for each file and get message
        # Only use latest file
        message, data = app.parse_contents(list_contents[-1], list_names[-1])
        data = data.to_json()

    else:
        message = html.Div(['No data file selected.'])
        data = french_flu_data.to_json()

    return message, data


@app.app.callback(
    Output('all-sliders', 'children'),
    Input('data_storage', 'children'),
)
def update_slider_ranges(*args):
    """Update sliders when a data file is uploaded.
    """
    data = french_flu_data if args[0] is None else args[0]
    app.refresh_user_data_json(data_storage=data)
    return app.update_sliders()


@app.app.callback(
        Output('myfig', 'figure'),
        [Input(s, 'value') for s in sliders],
        Input('sim-button', 'n_clicks'),
        )
def manage_simulation(*args):
    """
    Simulates the model for the current slider values or adds a new
    simulation for the current slider values and updates the
    plot in the figure.
    """
    ctx = dash.callback_context
    source = ctx.triggered[0]['prop_id'].split('.')[0]
    if source == 'sim-button':
        fig = app.add_simulation()
        for i in range(len(app.plot.figure['data'])-2):
            # Change opacity of all traces in the figure but for the
            # first - the barplot of incidences
            # last - the latest simulation
            app.plot.figure['data'][i+1]['line'].color = 'rgba(255,0,0,0.25)'
            app.plot.figure['data'][i+1]['showlegend'] = False
    elif source in sliders:
        parameters = args[:-1]
        fig = app.update_simulation(*parameters)
        fig = app.clear_simulations()
    else:
        # The input source is not recognized, so make no change to the output
        raise dash.exceptions.PreventUpdate()

    return fig


@app.app.callback(
    Output('collapsedtext', 'is_open'),
    [Input('showhidebutton', 'n_clicks')],
    [State('collapsedtext', 'is_open')],
)
def toggle_hidden_text(num_clicks, is_it_open):
    """
    Switches the visibility of the hidden text.
    """
    if num_clicks:
        return not is_it_open
    return is_it_open


if __name__ == "__main__":
    app.app.run_server(debug=True)
