#
# InferenceApp
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#
from math import floor
import pandas as pd

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

import branchpro as bp
from branchpro.apps import BranchProDashApp


class BranchProInferenceApp(BranchProDashApp):
    """BranchProInferenceApp Class:
    Class for the inference dash app with figure and sliders for the
    BranchPro models.
    """
    def __init__(self):
        super(BranchProInferenceApp, self).__init__()

        self.app = dash.Dash(__name__, external_stylesheets=self.css)
        self.app.title = 'BranchproInf'

        self.session_data = {
            'data_storage': None,
            'interval_storage': None,
            'posterior_storage': None}

        button_style = {
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        }

        self.app.layout = html.Div([
            dbc.Container(
                [
                    html.H1('Branching Processes', id='page-title'),
                    html.Div([]),  # Empty div for top explanation texts
                    html.H2('Incidence Data'),
                    dbc.Row(
                        dbc.Col(dcc.Graph(
                                figure=bp.IncidenceNumberPlot().figure,
                                id='data-fig'))
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                children=[
                                    html.H6([
                                        'You can upload your own ',
                                        html.Span(
                                            'incidence data',
                                            id='id-tooltip',
                                            style={
                                                'textDecoration':
                                                    'underline',
                                                'cursor':
                                                    'pointer'},
                                        ),
                                        ' here. It will appear as bars.']),
                                    dbc.Tooltip(
                                        'here it is',
                                        target='id-tooltip',
                                    ),
                                    html.Div([
                                        'Data must be in the following column '
                                        'format: `Time`, `Incidence number`, '
                                        '`Imported Cases` (optional), '
                                        '`R_t` (true value of R, optional).']),
                                    dcc.Upload(
                                        id='upload-data',
                                        children=html.Div(
                                            [
                                                'Drag and Drop or ',
                                                html.A(
                                                    'Select Files',
                                                    style={
                                                        'text-decoration':
                                                            'underline'}),
                                                ' to upload your Incidence \
                                                    Number data.'
                                            ]),
                                        style=button_style,
                                        # Allow multiple files to be uploaded
                                        multiple=True
                                    ),
                                    html.Div(id='incidence-data-upload')]),
                            dbc.Col(
                                children=[
                                    html.H6([
                                        'You can upload your own ',
                                        html.Span(
                                            'serial interval',
                                            id='si-tooltip',
                                            style={
                                                'textDecoration':
                                                    'underline',
                                                'cursor':
                                                    'pointer'}
                                        ),
                                        ' here.'
                                    ]),
                                    dbc.Tooltip(
                                        'here is the other one',
                                        target='si-tooltip',
                                    ),
                                    html.Div([
                                        'Data must contain one or more serial '
                                        'intervals to be used for constructing'
                                        ' the posterior distributions each '
                                        'included as a column.']),
                                    dcc.Upload(
                                        id='upload-interval',
                                        children=html.Div(
                                            [
                                                'Drag and Drop or ',
                                                html.A(
                                                    'Select Files',
                                                    style={
                                                        'text-decoration': '\
                                                            underline'}),
                                                ' to upload your Serial \
                                                    Interval.'
                                            ]),
                                        style=button_style,
                                        # Allow multiple files to be uploaded
                                        multiple=True
                                    ),
                                    html.Div(id='ser-interval-upload')])
                        ],
                        align='center',
                    ),
                    html.H2('Plot of R values'),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Loading(
                                    id='loading',
                                    children=dcc.Graph(
                                        figure=bp.ReproductionNumberPlot(
                                            ).figure,
                                        id='posterior-fig'),
                                    type="circle")),
                            dbc.Col(self.update_sliders(), id='all-sliders')
                        ],
                        align='center',
                    ),
                    html.Div([]),  # Empty div for bottom text
                    html.Div(id='data_storage', style={'display': 'none'}),
                    html.Div(id='interval_storage', style={'display': 'none'}),
                    html.Div(id='posterior_storage', style={'display': 'none'})
                    ],
                fluid=True),
            self.mathjax_script])

        # Set the app index string for mathjax
        self.app.index_string = self.mathjax_html

        # Save the locations of texts from the layout
        self.main_text = self.app.layout.children[0].children[1].children
        self.collapsed_text = self.app.layout.children[0].children[-4].children

    def update_sliders(self,
                       mean=5.0,
                       stdev=5.0,
                       tau=6,
                       central_prob=.95):
        """Generate sliders for the app.

        Parameters
        ----------
        mean
            (float) start position on the slider for the mean of the
            prior for the Branch Pro model in the posterior.
        stdev
            (float) start position on the slider for the standard deviation of
            the prior for the Branch Pro model in the posterior.
        tau
            (int) start position on the slider for the tau window used in the
            running of the inference of the reproduction numbers of the Branch
            Pro model in the posterior.
        central_prob
            (float) start position on the slider for the level of the computed
            credible interval of the estimated R number values.

        Returns
        -------
        html.Div
            A dash html component containing the sliders
        """
        data = self.session_data.get('data_storage')
        if data is not None:
            time_label, inc_label = data.columns[:2]
            times = data[time_label]
            max_tau = floor((times.max() - times.min() + 1)/3)
            if tau > max_tau:
                # If default value of tau exceeds maximum accepted
                # choose tau to be this maximum value
                tau = max_tau
        else:
            max_tau = 7

        sliders = bp._SliderComponent()
        if (data is not None) and ('Imported Cases' in data.columns):
            # Add slider for epsilon only when imported cases are detected
            # in the data with default assuming equal R numbers for local
            # and imported cases
            sliders.add_slider(
                'Epsilon', 'epsilon', 0, -0.99, 2.0, 0.01)
        else:
            sliders.add_slider(
                'Epsilon', 'epsilon', 0, -0.99, 2.0, 0.01, invisible=True)

        sliders.add_slider(
            'Prior Mean', 'mean', mean, 0.1, 10.0, 0.01)
        sliders.add_slider(
            'Prior Standard Deviation', 'stdev', stdev, 0.1, 10.0, 0.01)
        sliders.add_slider(
            'Inference Sliding Window', 'tau', tau, 0, max_tau, 1,
            as_integer=True)
        sliders.add_slider(
            'Central Posterior Probability', 'central_prob', central_prob, 0.1,
            0.99, 0.01)

        return sliders.get_sliders_div()

    def update_posterior(self, mean, stdev, tau, central_prob, epsilon=None):
        """Update the posterior distribution based on slider values.

        Parameters
        ----------
        mean
            (float) updated position on the slider for the mean of
            the prior for the Branch Pro model in the posterior.
        stdev
            (float) updated position on the slider for the standard deviation
            of the prior for the Branch Pro model in the posterior.
        tau
            (int) updated position on the slider for the tau window used in the
            running of the inference of the reproduction numbers of the Branch
            Pro model in the posterior.
        central_prob
            (float) updated position on the slider for the level of the
            computed credible interval of the estimated R number values.
        epsilon
            (float) updated position on the slider for the constant of
            proportionality between local and imported cases for the Branch Pro
            model in the posterior.

        Returns
        -------
        pandas.DataFrame
            The posterior distribution, summarized in a dataframe with the
            following columns: 'Time Points', 'Mean', 'Lower bound CI' and
            'Upper bound CI'
        """
        new_alpha = (mean / stdev) ** 2
        new_beta = mean / (stdev ** 2)

        data = self.session_data.get('data_storage')

        if data is None:
            raise dash.exceptions.PreventUpdate()

        time_label, inc_label = data.columns[:2]
        num_cols = len(self.session_data.get('interval_storage').columns)

        prior_params = (new_alpha, new_beta)
        labels = {'time_key': time_label, 'inc_key': inc_label}

        if num_cols == 1:
            serial_interval = self.session_data.get(
                'interval_storage').iloc[:, 0].values

            if 'Imported Cases' in data.columns:
                # Separate data into local and imported cases
                imported_data = pd.DataFrame({
                    time_label: data[time_label],
                    inc_label: data['Imported Cases']
                })

                # Posterior follows the LocImp behaviour
                posterior = bp.LocImpBranchProPosterior(
                    data,
                    imported_data,
                    epsilon,
                    serial_interval,
                    *prior_params,
                    **labels)

            else:
                # Posterior follows the simple behaviour
                posterior = bp.BranchProPosterior(
                    data,
                    serial_interval,
                    *prior_params,
                    **labels)

        else:
            serial_intervals = self.session_data.get(
                'interval_storage').values.T

            if 'Imported Cases' in data.columns:
                # Separate data into local and imported cases
                imported_data = pd.DataFrame({
                    time_label: data[time_label],
                    inc_label: data['Imported Cases']
                })

                # Posterior follows the LocImp behaviour
                posterior = bp.LocImpBranchProPosteriorMultSI(
                    data,
                    imported_data,
                    epsilon,
                    serial_intervals,
                    *prior_params,
                    **labels)
            else:
                # Posterior follows the simple behaviour
                posterior = bp.BranchProPosteriorMultSI(
                    data,
                    serial_intervals,
                    *prior_params,
                    **labels)

        posterior.run_inference(tau)
        return posterior.get_intervals(central_prob)

    def update_inference_figure(self,
                                source=None):
        """Update the inference figure based on currently stored information.

        Parameters
        ----------
        source : str
            Dash callback source

        Returns
        -------
        plotly.Figure
            Figure with updated posterior distribution
        """
        data = self.session_data.get('data_storage')
        posterior = self.session_data.get('posterior_storage')

        if data is None or posterior is None:
            raise dash.exceptions.PreventUpdate()

        time_label, inc_label = data.columns[:2]

        plot = bp.ReproductionNumberPlot()
        plot.add_interval_rt(posterior)

        if 'R_t' in data.columns:
            plot.add_ground_truth_rt(
                data[[time_label, 'R_t']],
                time_key=time_label,
                r_key='R_t')

        # Keeps traces visibility states fixed when changing sliders
        plot.figure['layout']['legend']['uirevision'] = True

        return plot.figure

    def update_data_figure(self):
        """Update the data figure based on currently stored information.

        Returns
        -------
        plotly.Figure
            Figure with updated data
        """
        data = self.session_data.get('data_storage')

        if data is None:
            raise dash.exceptions.PreventUpdate()

        time_label, inc_label = data.columns[:2]

        plot = bp.IncidenceNumberPlot()

        if 'Imported Cases' in data.columns:
            # Separate data into local and imported cases
            imported_data = pd.DataFrame({
                time_label: data[time_label],
                inc_label: data['Imported Cases']
            })

            # Bar plot of local cases
            plot.add_data(
                data,
                time_key=time_label,
                inc_key=inc_label,
                name='Local Cases')

            # Bar plot of imported cases
            plot.add_data(
                imported_data,
                time_key=time_label,
                inc_key=inc_label,
                name='Imported Cases')

        else:
            # If no imported cases are present
            plot.add_data(data, time_key=time_label, inc_key=inc_label)

        # Keeps traces visibility states fixed when changing sliders
        plot.figure['layout']['legend']['uirevision'] = True

        return plot.figure
