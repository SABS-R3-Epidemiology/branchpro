#
# Inference Dash app
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#
"""This is an app which shows inference of the reproduction number values of
the branching process model with genreated example data. To run the app, use
``python inference_dash_app.py``.
"""

import os

import numpy as np
import pandas as pd
import scipy.stats
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import branchpro as bp
from branchpro.apps import BranchProInferenceApp


app = BranchProInferenceApp()

# Generate synthetic data

num_timepoints = 30  # number of days for incidence data

# Build the serial interval w_s
ws_mean = 2.6
ws_var = 1.5**2
theta = ws_var / ws_mean
k = ws_mean / theta
w_dist = scipy.stats.gamma(k, scale=theta)
disc_w = w_dist.pdf(np.arange(1, num_timepoints+1))

# Simulate incidence data
initial_r = 3
serial_interval = disc_w
model = bp.BranchProModel(initial_r, serial_interval)
new_rs = [3, 0.5]          # sequence of R_0 numbers
start_times = [0, 15]      # days at which each R_0 period begins
model.set_r_profile(new_rs, start_times)
parameters = 10  # initial number of cases
times = np.arange(num_timepoints)

cases = model.simulate(parameters, times)
example_data = pd.DataFrame({
            'Days': times,
            'Incidence Number': cases,
            'R_t': [np.nan] + list(model.get_r_profile())
        })

sliders = ['epsilon', 'mean', 'stdev', 'tau', 'central_prob']

# Add the explanation texts
fname = os.path.join(os.path.dirname(__file__),
                     'data',
                     'dash_app_text_inference.md')
with open(fname) as f:
    app.add_text(dcc.Markdown(f.read(), dangerously_allow_html=True))

fname = os.path.join(os.path.dirname(__file__),
                     'data',
                     'dash_app_collapsible_text_inference.md')
with open(fname) as f:
    app.add_collapsed_text(dcc.Markdown(f.read(), dangerously_allow_html=True))

# Get server of the app; necessary for correct deployment of the app.
server = app.app.server

# Serial interval is currently fixed and constant
app.serial_interval = serial_interval


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
                                               list_names[-1])

            if data is None:
                # The file could not be loaded, so keep the current data and
                # try to prevent updates to any other part of the app
                return message, dash.no_update

            data = data.to_json()

        else:
            message = html.Div(['No data file selected.'])
            data = example_data.to_json()

        return message, data


@app.app.callback(
    Output('data-fig', 'figure'),
    Input('data_storage', 'children'),
)
def update_data_figure(*args):
    """Handles all updates to the data figure.
    """
    with app.lock:
        app.refresh_user_data_json(data_storage=args[0])
        return app.update_data_figure()


@app.app.callback(
    Output('all-sliders', 'children'),
    Input('data_storage', 'children'),
)
def update_slider_ranges(*args):
    """Update sliders when a data file is uploaded.
    """
    data = example_data if args[0] is None else args[0]
    with app.lock:
        app.refresh_user_data_json(data_storage=data)
        return app.update_sliders()


@app.app.callback(
    Output('posterior-fig', 'figure'),
    Input('posterior_storage', 'children'),
    State('data_storage', 'children'),
)
def update_posterior_figure(*args):
    """Handles all updates to the posterior figure.
    """
    with app.lock:
        app.refresh_user_data_json(
            data_storage=args[1], posterior_storage=args[0])
        return app.update_inference_figure()


@app.app.callback(
    Output('posterior_storage', 'children'),
    [Input(s, 'value') for s in sliders],
    State('data_storage', 'children'),
)
def calculate_posterior(*args):
    """Calculate the posterior distribution.
    """
    epsilon, mean, stdev, tau, central_prob, data_json = args

    with app.lock:
        app.refresh_user_data_json(data_storage=data_json)
        return app.update_posterior(
            mean, stdev, tau, central_prob, epsilon).to_json()


@app.app.callback(
    Output('collapsedtext', 'is_open'),
    [Input('showhidebutton', 'n_clicks')],
    [State('collapsedtext', 'is_open')],
)
def toggle_hidden_text(num_clicks, is_it_open):
    """Switches the visibility of the hidden text.
    """
    if num_clicks:
        return not is_it_open
    return is_it_open


if __name__ == "__main__":
    app.app.run_server(debug=True)
