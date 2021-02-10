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
import dash_defer_js_import as dji  # For mathjax
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

import branchpro as bp
from branchpro.apps import IncidenceNumberSimulationApp


# Import the mathjax
mathjax_script = dji.Import(
    src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js'
        '?config=TeX-AMS-MML_SVG')


class BranchProInferenceApp(IncidenceNumberSimulationApp):
    """BranchProInferenceApp Class:
    Class for the inference dash app with figure and sliders for the
    BranchPro models.
    """
    def __init__(self):
        super(BranchProInferenceApp, self).__init__()
        self.app.title = 'BranchproInf'
        self.plot1 = self.plot
        self.plot2 = bp.ReproductionNumberPlot()

        # Keeps traces visibility states fixed when changing sliders
        # in the second figure
        self.plot2.figure['layout']['legend']['uirevision'] = True

        self.app.layout = html.Div([
            dbc.Container(
                [
                    html.H1('Branching Processes'),
                    html.Div([]),  # Empty div for top explanation texts
                    html.H2('Incidence Data'),
                    dbc.Row(
                        dbc.Col(dcc.Graph(
                                figure=self.plot1.figure, id='fig1'))
                    ),
                    html.H2('Plot of R values'),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(
                                figure=self.plot2.figure, id='fig2')),
                            dbc.Col(self.sliders.get_sliders_div())
                        ],
                        align='center',
                    ),
                    html.Div([])],  # Empty div for bottom text
                fluid=True),
            mathjax_script])

    def add_ground_truth_rt(self, df, time_label='Time Points', r_label='R_t'):
        """
        Adds incidence data to the plot in the dash app.

        Parameters
        ----------
        df
            (pandas DataFrame) contains the true values of the reproduction
            number by time unit. Data stored in columns of time and
            reproduction number, respectively.
        time_label
            label key given to the temporal data in the dataframe.
        r_label
            label key given to the reproduction number data in the dataframe.
        """
        self.plot2.add_ground_truth_rt(df, time_key=time_label, r_key=r_label)

    def add_posterior(self,
                      posterior,
                      mean=5.0,
                      stdev=5.0,
                      tau=6,
                      central_prob=.95):
        """
        Computes the posterior probability of some instances dataframe, infers
        reproduction numbers with quantiles, them as lines to the second
        plot and adds sliders to the app.

        Parameters
        ----------
        posterior
            (BranchProPosterior) a dataframe of instances and the parameters of
            the prior used to infer the reproduction numbers of a BranchPro
            model used to model the data.
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
        """
        if not issubclass(type(posterior), bp.BranchProPosterior):
            raise TypeError('Posterior needs to be a BranchProPosterior')

        fig1_labels = posterior.cases_labels

        df1 = pd.DataFrame({
            fig1_labels[0]: posterior.cases_times,
            fig1_labels[1]: posterior.cases_data})

        max_tau = floor(
            posterior.cases_times.max() - posterior.cases_times.min() + 1)/3

        self.sliders.add_slider(
            'Prior Mean', 'mean', mean, 0.1, 10.0, 0.01)
        self.sliders.add_slider(
            'Prior Standard Deviation', 'stdev', stdev, 0.1, 10.0, 0.01)
        self.sliders.add_slider(
            'Inference Sliding Window', 'tau', tau, 0, max_tau, 1,
            as_integer=True)
        self.sliders.add_slider(
            'Central Posterior Probability', 'central_prob', central_prob, 0.1,
            0.99, 0.01)

        alpha = (mean/stdev)**2
        beta = mean/(stdev**2)
        prior_parameters = (alpha, beta)
        posterior.prior_parameters = prior_parameters

        posterior.run_inference(tau)
        df2 = posterior.get_intervals(central_prob)

        self.plot1.add_data(
            df1, time_key=fig1_labels[0], inc_key=fig1_labels[1])
        self.plot2.add_interval_rt(df2)

        self.posterior = posterior

        # Save the infered figure for later update
        self._graph = self.plot2.figure['data'][-1]
        self._graph_mean = self.plot2.figure['data'][-2]

    def update_inference(
            self, new_mean, new_stdev, new_tau, new_central_prob):
        """
        Updates the parameters in the inference and the estimated reproduction
        numbers and quantiles graph in the second figure.

        Parameters
        ----------
        new_mean
            (float) updated position on the slider for the mean of
            the prior for the Branch Pro model in the posterior.
        new_stdev
            (float) updated position on the slider for the standard deviation
            of the prior for the Branch Pro model in the posterior.
        new_tau
            (int) updated position on the slider for the tau window used in the
            running of the inference of the reproduction numbers of the Branch
            Pro model in the posterior.
        new_central_prob
            (float) start position on the slider for the level of the computed
            credible interval of the estimated R number values.
        """
        new_alpha = (new_mean/new_stdev)**2
        new_beta = new_mean/(new_stdev**2)
        new_prior_parameters = (new_alpha, new_beta)

        self.posterior.prior_parameters = new_prior_parameters

        self.posterior.run_inference(new_tau)
        df = self.posterior.get_intervals(new_central_prob)

        time_key = 'Time Points'
        ur_key = 'Upper bound CI'
        lr_key = 'Lower bound CI'
        cp_key = 'Central Probability'
        new_x = list(df[time_key]) + list(df[time_key])[::-1]
        new_y = list(df[ur_key]) + list(df[lr_key])[::-1]

        self._graph['x'] = new_x
        self._graph['y'] = new_y

        self._graph_mean['x'] = df[time_key]
        self._graph_mean['y'] = df['Mean']

        self._graph['name'] = 'Credible interval ' + str(df[cp_key][0])

        return self.plot2.figure
