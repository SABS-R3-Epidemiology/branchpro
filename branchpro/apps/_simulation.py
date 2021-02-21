#
# SimulationApp
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

# import threading

import numpy as np
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

import branchpro as bp
from branchpro.apps import BranchProDashApp


class IncidenceNumberSimulationApp(BranchProDashApp):
    """IncidenceNumberSimulationApp Class:
    Class for the simulation dash app with figure and sliders for the
    BranchPro models.
    """
    def __init__(self):
        super().__init__()

        self.session_data = {'data_storage': None, 'sim_storage': None}

        self.app = dash.Dash(__name__, external_stylesheets=self.css)
        self.app.title = 'BranchproSim'

        self.app.layout = \
            html.Div([
                dbc.Container([
                    html.H1('Branching Processes'),
                    html.Div([]),  # Empty div for top explanation texts
                    dbc.Row([
                        dbc.Col([
                            html.Button(
                                'Add new simulation',
                                id='sim-button',
                                n_clicks=0),
                            dcc.Graph(
                                figure=bp.IncidenceNumberPlot().figure,
                                id='myfig')
                        ]),
                        dbc.Col(
                            self.update_sliders(), id='all-sliders')
                    ], align='center'),
                    html.H4([
                        'You can upload your own incidence data here. It will'
                        'appear as bars, while the simulation will be a line.'
                    ]),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files',
                                   style={'text-decoration': 'underline'}),
                            ' to upload your Incidence Number data.'
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        multiple=True  # Allow multiple files to be uploaded
                    ),
                    html.Div(id='incidence-data-upload'),
                    html.Div([]),  # Empty div for bottom text
                    # dcc.Store(id='data_storage'),
                    # dcc.Store(id='sim_storage')
                    html.Div(id='data_storage', style={'display': 'none'}),
                    html.Div(id='sim_storage', style={'display': 'none'})
                    ], fluid=True),
                self.mathjax_script
                ])

        # Set the app index string for mathjax
        self.app.index_string = self.mathjax_html

    def update_sliders(self,
                       init_cond=10.0,
                       r0=2.0,
                       r1=0.5,
                       magnitude_init_cond=None):
        """Generate sliders for the app.

        This method tunes the bounds of the sliders to the time period and
        magnitude of the data.

        Parameters
        ----------
        init_cond : int
            start position on the slider for the number of initial cases for
            the Branch Pro model in the simulator.
        r0 : float
            start position on the slider for the initial reproduction number
            for the Branch Pro model in the simulator.
        r1 : float
            start position on the slider for the second reproduction number for
            the Branch Pro model in the simulator.
        magnitude_init_cond : int
            maximal start position on the slider for the number of initial
            cases for the Branch Pro model in the simulator. By default, it
            will be set to the maximum value observed in the data.

        Returns
        -------
        html.Div
            A dash html component containing the sliders
        """
        data = self.session_data['data_storage']

        # Calculate slider values that depend on the data
        if data is not None:
            time_label, inc_label = data.columns
            if magnitude_init_cond is None:
                magnitude_init_cond = max(data[inc_label])
            bounds = (1, max(data[time_label]))

        else:
            # choose values to use if there is no data
            if magnitude_init_cond is None:
                magnitude_init_cond = 10
            bounds = (1, 30)

        mid_point = round(sum(bounds) / 2)

        # Make new sliders
        sliders = bp._SliderComponent()
        sliders.add_slider(
            'Initial Cases', 'init_cond', init_cond, 0.0, magnitude_init_cond,
            1, as_integer=True)
        sliders.add_slider('Initial R', 'r0', r0, 0.1, 10.0, 0.01)
        sliders.add_slider('Second R', 'r1', r1, 0.1, 10.0, 0.01)
        sliders.add_slider(
            'Time of change', 't1', mid_point, bounds[0], bounds[1], 1,
            as_integer=True)

        return sliders.get_sliders_div()

    def update_figure(self):
        """Generate a plotly figure of incidence numbers and simulated cases.

        Returns
        -------
        plotly.Figure
            Figure with updated data and simulations
        """
        data = self.session_data['data_storage']
        simulations = self.session_data['sim_storage']

        time_label, inc_label = data.columns
        num_simulations = len(simulations.columns) - 1

        plot = bp.IncidenceNumberPlot()
        plot.add_data(data, time_key=time_label, inc_key=inc_label)

        # Keeps traces visibility states fixed when changing sliders
        plot.figure['layout']['legend']['uirevision'] = True

        for sim in range(num_simulations):
            df = simulations[[time_label, 'sim{}'.format(sim + 1)]]
            df.columns = [time_label, inc_label]
            plot.add_simulation(df, time_key=time_label, inc_key=inc_label)

            # Unless it is the most recent simulation, decrease the opacity to
            # 25% and remove it from the legend
            if sim < num_simulations - 1:
                plot.figure['data'][-1]['line'].color = 'rgba(255,0,0,0.25)'
                plot.figure['data'][-1]['showlegend'] = False

        return plot.figure

    def clear_simulations(self):
        """Remove all previous simulations from sim storage.
        """
        sim = self.session_data['sim_storage']
        data = self.session_data['data_storage']

        time_label, inc_label = data.columns

        # Only attempt the purge if there is data there
        if sim is not None:
            self.session_data['sim_storage'] = data[[time_label]]

    def add_text(self, text):
        """Add a block of text at the top of the app.

        This can be used to add introductory text that everyone looking at the
        app will see right away.

        Parameters
        ----------
        text : str
            The text to add to the html div
        """
        self._load_text(text)
        self.app.layout.children[0].children[1].children.append(self.text)

    def add_collapsed_text(self, text, title='More details...'):
        """Add a block of text at the top of the app.

        By default, this text will be hidden. The user can click on a button
        with the specified title in order to view the text.

        Parameters
        ----------
        text : str
            The text to add to the html div
        title : str
            str which will be displayed on the show/hide button
        """
        self._load_collapsed_text(text, title)
        self.app.layout.children[0].children[-3].children.append(
            self.collapsed_text)

    def update_simulation(self, new_init_cond, new_r0, new_r1, new_t1):
        """Run a simulation of the branchpro model at the given slider values.

        Parameters
        ----------
        new_init_cond
            (int) updated position on the slider for the number of initial
            cases for the Branch Pro model in the simulator.
        new_r0
            (float) updated position on the slider for the initial reproduction
            number for the Branch Pro model in the simulator.
        new_r1
            (float) updated position on the slider for the second reproduction
            number for the Branch Pro model in the simulator.
        new_t1
            (float) updated position on the slider for the time change in
            reproduction numbers for the Branch Pro model in the simulator.

        Returns
        -------
        pandas.DataFrame
            Simulations storage dataframe
        """
        data = self.session_data['data_storage']
        simulations = self.session_data['sim_storage']

        time_label, inc_label = data.columns
        times = data[time_label]

        # There might be no simulation data if it got just cleared, or this is
        # the first call --- start a new dataframe in this case
        if simulations is None:
            simulations = data[[time_label]]

        # Add the correct R profile to the branchpro model
        br_pro_model = bp.BranchProModel(new_r0, np.array([1, 2, 3, 2, 1]))
        br_pro_model.set_r_profile([new_r0, new_r1], [0, new_t1])

        # Generate one simulation trajectory from this model
        simulation_controller = bp.SimulationController(
            br_pro_model, 1, len(times))
        data = simulation_controller.run(new_init_cond)

        # Add data to simulations storage
        num_sims = len(simulations.columns)
        simulations['sim{}'.format(num_sims)] = data

        return simulations
