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

import numpy as np
import pandas as pd
from dash.dependencies import Input, Output

import branchpro as bp
from branchpro.apps import IncidenceNumberSimulationApp


app = IncidenceNumberSimulationApp()
full_ffd = bp.DatasetLibrary().french_flu()
small_ffd = full_ffd.query('year == 2020')
df = pd.DataFrame({
            'Week': small_ffd['week'],
            'Incidence Number': small_ffd['inc']
        })

br_pro_model = bp.BranchProModel(2, np.array([1, 2, 3, 2, 1]))
simulationController = bp.SimulationController(
    br_pro_model, 1, len(small_ffd['week']))
app.add_simulator(
    simulationController,
    magnitude_init_cond=max(df['Incidence Number']))
app.add_data(df, time_label='Week')

sliders = app.get_sliders_ids()

# Get server of the app; necessary for correct deployment of the app.
server = app.app.server


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


if __name__ == "__main__":
    app.app.run_server(debug=True)
