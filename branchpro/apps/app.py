import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

import pandas as pd

import branchpro
import scipy.stats
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Build the serial interval w_s
num_timepoints = 31
ws_mean = 2.6
ws_var = 1.5**2
theta = ws_var / ws_mean
k = ws_mean / theta

w_dist = scipy.stats.gamma(k, scale=theta)
disc_w = w_dist.pdf(np.arange(1, num_timepoints+1))

# Set time frame for SimulationControllers
start_sim_time = 0
end_sim_time = 30

# Construct BranchProModel objects
initial_r = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 5])
second_r = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 5])
change_time = np.array([1, 15, 30])
serial_interval = disc_w

# Set up BranchPro object
m = branchpro.BranchProModel(1, serial_interval)

app.layout = html.Div(children=[
    html.H1(children='Daily incidences of Flu'),

    html.Div(children='''
        A web application framework for incidence numbers
        by day.
    '''),

    dcc.Graph(id='graph-with-slider'),

    html.Label('Initial R'),
    dcc.Slider(
        id='initial-r-slider',
        min=np.amin(initial_r),
        max=np.amax(initial_r),
        value=np.amin(initial_r),
        marks={str(ri): str(ri) for ri in np.unique(initial_r)},
        step=None
    ),

    html.Label('Time of change'),
    dcc.Slider(
        id='time-slider',
        min=np.amin(change_time),
        max=np.amax(change_time),
        value=np.amin(change_time),
        marks={str(t): str(t) for t in np.unique(change_time)},
        step=None
    ),

    html.Label('Second R'),
    dcc.Slider(
        id='second-r-slider',
        min=np.amin(second_r),
        max=np.amax(second_r),
        value=np.amin(second_r),
        marks={str(r): str(r) for r in np.unique(second_r)},
        step=None
    )
    ]
)


@app.callback(
    Output('graph-with-slider', 'figure'),
    [
        Input('initial-r-slider', 'value'),
        Input('second-r-slider', 'value'),
        Input('time-slider', 'value')
    ]
)
def update_figure(selected_initial_r, selected_second_r, selected_change_time):

    new_rs = [selected_initial_r, selected_second_r]
    start_times = [0, selected_change_time]
    m.set_r_profile(new_rs, start_times)

    # Set up SimulationController object
    parameters = 10  # initial number of cases

    sc = branchpro.SimulationController(m, start_sim_time, end_sim_time)

    sc.switch_resolution(num_timepoints)

    cases = sc.run(parameters)

    df = pd.DataFrame({
        "time": np.arange(start_sim_time, end_sim_time+1),
        "incidence": cases
    })

    fig = px.bar(
        df, x="time", y="incidence")

    fig.update_layout(transition_duration=500)

    return fig


if __name__ == '__main__':
    print("starting app")
    app.run_server(debug=True)
