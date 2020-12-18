#
# InferenceApp
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import pandas as pd
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

import branchpro as bp
from branchpro.apps import IncidenceNumberSimulationApp


class BranchProInferenceApp(IncidenceNumberSimulationApp):
    """BranchProInferenceApp Class:
    Class for the inference dash app with figure and sliders for the
    BranchPro models.
    """
    def __init__(self):
        self.app = dash.Dash(
            __name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.plot1 = bp.IncidenceNumberPlot()
        self.plot2 = bp.ReproductionNumberPlot()

        # Keeps traces visibility states fixed when changing sliders
        # in the second figure
        self.plot2.figure['layout']['legend']['uirevision'] = True
        self.sliders = bp._SliderComponent()

        self.app.layout = dbc.Container(
            [
                html.H1('Branching Processes'),
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
            ],
            fluid=True,
        )

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
                      alpha=1.0,
                      beta=0.2,
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
        alpha
            (float) start position on the slider for the shape parameter of the
            prior for the Branch Pro model in the posterior.
        beta
            (float) start position on the slider for the rate parameter of the
            prior for the Branch Pro model in the posterior.
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

        df1 = pd.DataFrame({
            'Time': posterior.cases_times,
            'Incidence Number': posterior.cases_data})

        max_tau = posterior.cases_times.max() - posterior.cases_times.min() + 1

        self.sliders.add_slider(
            'Shape Parameter', 'alpha', alpha, 0.1, 10.0, 0.01)
        self.sliders.add_slider(
            'Rate Parameter', 'beta', beta, 0.1, 10.0, 0.01)
        self.sliders.add_slider(
            'Time of change', 'tau', tau, 0, max_tau, 1,
            as_integer=True)
        self.sliders.add_slider(
            'Central Probability', 'central_prob', central_prob, 0.1, 1.0,
            0.01)

        prior_parameters = (alpha, beta)
        posterior.prior_parameters = prior_parameters

        posterior.run_inference(tau)
        full_df2 = posterior.get_intervals(central_prob)
        df2 = full_df2.iloc[:, :-1]

        self.plot1.add_data(df1)
        self.plot2.add_interval_rt(df2)

        self.posterior = posterior

        # Save the infered figure for later update
        self._graph = self.plot2.figure['data'][-1]

    def get_sliders_ids(self):
        """
        Returns the IDs of all sliders accompaning the figure in the
        app.
        """
        return self.sliders.slider_ids()

    def update_inference(
            self, new_alpha, new_beta, new_tau, new_central_prob):
        """
        Updates the parameters in the inference and the estimated reproduction
        numbers and quantiles graph in the second figure.

        Parameters
        ----------
        new_alpha
            (float) updated position on the slider for the shape parameter of
            the prior for the Branch Pro model in the posterior.
        new_beta
            (float) updated position on the slider for the rate parameter of
            the prior for the Branch Pro model in the posterior.
        new_tau
            (int) updated position on the slider for the tau window used in the
            running of the inference of the reproduction numbers of the Branch
            Pro model in the posterior.
        new_central_prob
            (float) start position on the slider for the level of the computed
            credible interval of the estimated R number values.
        """
        new_prior_parameters = (new_alpha, new_beta)

        self.posterior.prior_parameters = new_prior_parameters

        self.posterior.run_inference(new_tau)
        full_df = self.posterior.get_intervals(new_central_prob)
        df = full_df.iloc[:, :-1]

        self._graph['y'] = df

        return self.plot2.figure
