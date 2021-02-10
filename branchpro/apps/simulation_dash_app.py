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

# Get the sliders that change with new data
changing_sliders = ['init_cond', 't1']

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

        if (
            'Time' not in app.current_df.columns) or (
                'Incidence Number' not in app.current_df.columns):
            return html.Div(['Incorrect format; file must contain a `Time` \
                and `Incidence Number` column'])

        else:
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
    data = app.current_df
    if data is not None:
        # Send the new sliders div to the callback output
        return app.sliders.get_sliders_div()
    else:
        # There is no loaded data, so make no change to the output
        raise dash.exceptions.PreventUpdate()


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
