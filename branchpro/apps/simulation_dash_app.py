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
import dash_html_components as html

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


@app.app.callback(
        Output('incidence-data-upload', 'children'),
        Output('myfig', 'figure'),
        [Input(s, 'value') for s in sliders],
        Input('upload-data', 'contents'),
        State('upload-data', 'filename')
        )
def update_simulation(*args):
    """
    Simulates the model for the current slider values and updates the
    plot in the figure.
    """
    parameters = args[:-2]
    contents = args[-2]
    name = args[-1]

    context = dash.callback_context
    print(context.triggered[0]['prop_id'])

    children = html.Div([])
    if context.triggered[0]['prop_id'] == 'upload-data.contents':
        if contents is not None:
            children = [
                app.parse_contents(c, n) for c, n in zip(contents, name)]
            df = app.current_df
            new_fig = bp.IncidenceNumberPlot()
            new_sliders = bp._SliderComponent()
            new_fig.add_data(df)
            app.plot = new_fig
            app.sliders = new_sliders
            app.set_app_layout()
            simulationController = bp.SimulationController(
                br_pro_model, 1, max(df['Time']))
            app.add_simulator(
                simulationController,
                magnitude_init_cond=max(df['Incidence Number']))
        fig = app.plot.figure
    else:
        fig = app.update_simulation(*parameters)

    return children, fig


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
