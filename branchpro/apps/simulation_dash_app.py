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
from dash.dependencies import Input, Output, State
from flask_caching import Cache

import branchpro as bp
from branchpro.apps import IncidenceNumberSimulationApp


app = IncidenceNumberSimulationApp()
full_ffd = bp.DatasetLibrary().french_flu()
small_ffd = full_ffd.query('year == 2020')
df = pd.DataFrame({
            'Weeks': small_ffd['week'],
            'Incidence Number': small_ffd['inc']
        })

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
cache = Cache(app.app.server, config={
    'CACHE_TYPE': 'simple',
    'CACHE_THRESHOLD': 100})


@app.app.callback(
        Output('incidence-data-upload', 'children'),
        Input('upload-data', 'contents'),
        State('upload-data', 'filename')
        )
def update_current_df(*args):
    """
    Update when a data file is uploaded.
    """
    list_of_contents, list_of_names = args

    if list_of_contents is not None:
        # Run content parser for each file and get message
        # Only use latest file
        message = app.parse_contents(
            list_of_contents[-1], list_of_names[-1])

        if app.current_df is not None:
            # Make new empty plot and add data
            app.plot = bp.IncidenceNumberPlot()
            app.add_data(app.current_df)

            # Clear sliders - prevents from doubling sliders upon
            # page reload
            app.sliders = bp._SliderComponent()

            # Make a new simulation controller for this data
            simulationController = bp.SimulationController(
                br_pro_model, 1, len(app.current_df['Time']))
            app.add_simulator(
                simulationController,
                magnitude_init_cond=max(app.current_df['Incidence Number']))

        return message


@app.app.callback(
    Output('all-sliders', 'children'),
    Input('incidence-data-upload', 'children')
)
def update_sliders(*args):
    """
    Update sliders when a data file is uploaded.
    """
    data = app.current_df
    if data is not None:
        # Send the new sliders div to the callback output
        return app.sliders.get_sliders_div()
    else:
        # There is no loaded data, so make no change to the output
        raise dash.exceptions.PreventUpdate()


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
    cache.clear()
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
