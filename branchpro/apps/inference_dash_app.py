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
import dash_core_components as dcc
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
data = pd.DataFrame({
            'Days': times,
            'Incidence Number': cases,
            'R_t': [np.nan] + list(model.get_r_profile())
        })

sliders = ['mean', 'stdev', 'tau', 'central_prob']

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
    Output('data_storage', 'children'),
    Input('page-title', 'children'),
)
def update_data(*args):
    """Load incidence number data into app storage.

    Currently, this only runs once at page load, whereupon it saves the default
    data defined in the script above. When data upload functionality is added
    to the inference app, it can go here.
    """
    return data.to_json()


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
    Input('data_storage', 'children'),
    [Input(s, 'value') for s in sliders],
)
def calculate_posterior(*args):
    """Calculate the posterior distribution.
    """
    data_json, mean, stdev, tau, central_prob = args

    with app.lock:
        app.refresh_user_data_json(data_storage=data_json)
        return app.update_posterior(mean, stdev, tau, central_prob).to_json()


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
