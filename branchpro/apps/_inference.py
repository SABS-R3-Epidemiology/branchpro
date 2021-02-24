#
# InferenceApp
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#
from math import floor

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

        self.session_data = {'data_storage': None, 'posterior_storage': None}

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
                    html.H2('Plot of R values'),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(
                                figure=bp.ReproductionNumberPlot().figure,
                                id='posterior-fig')),
                            dbc.Col(self.update_sliders())
                        ],
                        align='center',
                    ),
                    html.Div([]),  # Empty div for bottom text
                    html.Div(id='data_storage', style={'display': 'none'}),
                    html.Div(id='posterior_storage', style={'display': 'none'})
                    ],
                fluid=True),
            self.mathjax_script])

        # Set the app index string for mathjax
        self.app.index_string = self.mathjax_html

        # Save the locations of texts from the layout
        self.main_text = self.app.layout.children[0].children[1].children
        self.collapsed_text = self.app.layout.children[0].children[-3].children

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
            max_tau = floor(times.max() - times.min() + 1)/3
        else:
            max_tau = 7

        sliders = bp._SliderComponent()
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

    def update_posterior(self, mean, stdev, tau, central_prob):
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

        posterior = bp.BranchProPosterior(
            data,
            self.serial_interval,
            new_alpha,
            new_beta,
            time_key=time_label,
            inc_key=inc_label)

        posterior.run_inference(tau)
        return posterior.get_intervals(central_prob)

    def update_inference_figure(self):
        """Update the inference figure based on currently stored information.

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
        plot.add_data(data, time_key=time_label, inc_key=inc_label)

        # Keeps traces visibility states fixed when changing sliders
        plot.figure['layout']['legend']['uirevision'] = True

        return plot.figure
