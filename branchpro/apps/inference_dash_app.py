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

import numpy as np
import pandas as pd
import scipy.stats
from dash.dependencies import Input, Output

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
            'Incidence Number': cases
        })

r_df = pd.DataFrame({
            'Days': times[1:],
            'R_t': model.get_r_profile()
        })

posterior = bp.BranchProPosterior(
    data, serial_interval, 1, 0.2, time_key='Days')
app.add_posterior(posterior)
app.add_ground_truth_rt(r_df, time_label='Days')

sliders = app.get_sliders_ids()

# Get server of the app; necessary for correct deployment of the app.
server = app.app.server


@app.app.callback(
        Output('fig2', 'figure'),
        [Input(s, 'value') for s in sliders])
def update_simulation(*args):
    """
    Simulates the model for the current slider values and updates the
    plot in the figure.
    """
    parameters = args
    fig = app.update_inference(*parameters)

    return fig


if __name__ == "__main__":
    app.app.run_server(debug=True)
