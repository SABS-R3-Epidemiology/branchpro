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
import json

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
french_flu_data = pd.DataFrame({
            'Weeks': small_ffd['week'],
            'Incidence Number': small_ffd['inc']
        })

serial_interval = np.array([1, 2, 3, 2, 1])

sliders = ['epsilon', 'init_cond', 'r0', 'r1', 't1']

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

    with app.lock:
        if list_contents is not None:
            # Run content parser for each file and get message
            # Only use latest file
            message, data = app.parse_contents(list_contents[-1],
                                               list_names[-1],
                                               sim_app=True)

            if data is None:
                # The file could not be loaded, so keep the current data and
                # try to prevent updates to any other part of the app
                return message, dash.no_update

            data = data.to_json()

        else:
            message = html.Div(['No data file selected.'])
            data = french_flu_data.to_json()

        return message, data


@app.app.callback(
    Output('ser-interval-upload', 'children'),
    Output('interval_storage', 'children'),
    Input('upload-interval', 'contents'),
    State('upload-interval', 'filename'),
)
def load_interval(*args):
    """Load serial interval from a file and save it in storage.
    """
    list_contents, list_names = args

    with app.lock:
        if list_contents is not None:
            # Run content parser for each file and get message
            # Only use latest file
            message, data = app.parse_contents(list_contents[-1],
                                               list_names[-1],
                                               is_si=True)

            if data is None:
                # The file could not be loaded, so keep the current data and
                # try to prevent updates to any other part of the app
                return message, dash.no_update
            else:
                data = json.dumps(data.tolist())

        else:
            message = html.Div(['Default serial interval in use.'])
            data = json.dumps(serial_interval.tolist())

        return message, data


@app.app.callback(
    Output('all-sliders', 'children'),
    Input('data_storage', 'children'),
)
def update_slider_ranges(*args):
    """Update sliders when a data file is uploaded.
    """
    data = french_flu_data if args[0] is None else args[0]
    with app.lock:
        app.refresh_user_data_json(data_storage=data)
        return app.update_sliders()


@app.app.callback(
    Output('myfig', 'figure'),
    Output('confirm', 'displayed'),
    Input('all-sliders', 'children'),
    [Input(s, 'value') for s in sliders],
    Input('sim-button', 'n_clicks'),
    Input('interval_storage', 'children'),
    State('myfig', 'figure'),
    State('data_storage', 'children'),
)
def update_figure(*args):
    """Handles all updates to the incidence number figure.
    """
    _, epsilon, init_cond, r0, r1, t1, _, interval_json, fig, data_json = args

    ctx = dash.callback_context
    source = ctx.triggered[0]['prop_id'].split('.')[0]

    with app.lock:
        app.refresh_user_data_json(
            data_storage=data_json, interval_storage=interval_json)

        new_sim = app.update_simulation(init_cond, r0, r1, t1, epsilon)
        if np.all(new_sim.iloc[:, -1] == -1):
            overflow = True
        else:
            overflow = False

        return (app.update_figure(fig=fig, simulations=new_sim, source=source),
                overflow)


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


@app.app.callback(
    Output('si_modal', 'is_open'),
    [Input('si-tooltip', 'n_clicks'), Input('si_modal_close', 'n_clicks')],
    [State('si_modal', 'is_open')],
)
def toggle_modal_inc(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.app.callback(
    Output('inc_modal', 'is_open'),
    [Input('inc-tooltip', 'n_clicks'), Input('inc_modal_close', 'n_clicks')],
    [State('inc_modal', 'is_open')],
)
def toggle_modal_si(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.app.run_server(debug=True)
