#
# SimulationApp
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import copy
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

        self.session_data = {'data_storage': None}

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
                    html.Div(id='data_storage', style={'display': 'none'}),
                    html.Div(id='sim_storage', style={'display': 'none'})
                    ], fluid=True),
                self.mathjax_script
                ])

        # Set the app index string for mathjax
        self.app.index_string = self.mathjax_html

        # Save the locations of texts from the layout
        self.main_text = self.app.layout.children[0].children[1].children
        self.collapsed_text = self.app.layout.children[0].children[-3].children

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
        data = self.session_data.get('data_storage')

        # Calculate slider values that depend on the data
        if data is not None:
            time_label, inc_label = data.columns
            if magnitude_init_cond is None:
                magnitude_init_cond = max(data[inc_label])
            bounds = (1, max(data[time_label]))

        else:
            # choose values to use if there is no data
            if magnitude_init_cond is None:
                magnitude_init_cond = 1000
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

    def update_figure(self,
                      fig=None,
                      simulations=None,
                      source=None):
        """Generate a plotly figure of incidence numbers and simulated cases.

        By default, this method uses the information saved in self.session_data
        to populate the figure with data. If a current figure and dash callback
        source are passed, it will try to just update the existing figure for
        speed improvements.

        Parameters
        ----------
        fig : dict
            Current copy of the figure
        simulations : pd.DataFrame
            Simulation trajectories to add to the figure.
        source : str
            Dash callback source

        Returns
        -------
        plotly.Figure
            Figure with updated data and simulations
        """
        if fig is not None and simulations is not None:
            # Check if there is a faster way to update the figure
            if len(fig['data']) > 0 and source in ['init_cond', 'r0', 'r1',
                                                   't1']:
                # Clear all traces except one simulation and the data
                fig['data'] = [fig['data'][0], fig['data'][-1]]

                # Set the y values of that trace equal to an updated simulation
                fig['data'][-1]['y'] = simulations.iloc[:, -1]

                return fig

            elif len(fig['data']) > 0 and source == 'sim-button':
                # Add one extra simulation, and set its y values
                fig['data'].append(copy.deepcopy(fig['data'][-1]))
                fig['data'][-1]['y'] = simulations.iloc[:, -1]

                for i in range(len(fig['data'])-2):
                    # Change opacity of all traces in the figure but for the
                    # first - the barplot of incidences
                    # last - the latest simulation
                    fig['data'][i+1]['line']['color'] = 'rgba(255,0,0,0.25)'
                    fig['data'][i+1]['showlegend'] = False

                return fig

        data = self.session_data.get('data_storage')

        if data is None:
            raise dash.exceptions.PreventUpdate()

        time_label, inc_label = data.columns
        num_simulations = len(simulations.columns) - 1

        # Make a new figure
        plot = bp.IncidenceNumberPlot()
        plot.add_data(data, time_key=time_label, inc_key=inc_label)

        # Keeps traces visibility states fixed when changing sliders
        plot.figure['layout']['legend']['uirevision'] = True

        for sim in range(num_simulations):
            df = simulations.iloc[:, [0, sim+1]]
            df.columns = [time_label, inc_label]
            plot.add_simulation(df, time_key=time_label, inc_key=inc_label)

            # Unless it is the most recent simulation, decrease the opacity to
            # 25% and remove it from the legend
            if sim < num_simulations - 1:
                plot.figure['data'][-1]['line'].color = 'rgba(255,0,0,0.25)'
                plot.figure['data'][-1]['showlegend'] = False

        return plot.figure

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
        data = self.session_data.get('data_storage')

        if data is None:
            raise dash.exceptions.PreventUpdate()

        time_label, inc_label = data.columns
        times = data[time_label]

        # Make a new dataframe to save the simulation result
        simulations = data[[time_label]]

        # Add the correct R profile to the branchpro model
        br_pro_model = bp.BranchProModel(new_r0, np.array([1, 2, 3, 2, 1]))
        br_pro_model.set_r_profile([new_r0, new_r1], [0, new_t1])

        # Generate one simulation trajectory from this model
        simulation_controller = bp.SimulationController(
            br_pro_model, 1, len(times))
        data = simulation_controller.run(new_init_cond)

        # Add data to simulations storage
        simulations[inc_label] = data

        return simulations
