#
# Dash app
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import pandas as pd

from dash.dependencies import Input, Output
import branchpro as bp
from branchpro.apps import IncidenceNumberSimulationApp

import numpy as np


app = IncidenceNumberSimulationApp()
df = pd.DataFrame({
            'Time': [1, 2, 3, 5, 6],
            'Incidence Number': [10, 3, 4, 6, 9]
        })

br_pro_model = bp.BranchProModel(2, np.array([1, 2, 3, 2, 1]))
simulationController = bp.SimulationController(br_pro_model, 1, 7)
app.add_simulator(simulationController)
app.add_data(df)

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


if __name__ == "__main__":
    app.app.run_server(debug=True)
