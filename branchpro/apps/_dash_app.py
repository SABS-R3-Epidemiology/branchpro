#
# BranchProDashApp Class
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import threading
import base64
import io
import os
import pandas as pd
import dash_defer_js_import as dji  # For mathjax
import dash_bootstrap_components as dbc
import dash_html_components as html


# Import the mathjax
mathjax_script = dji.Import(
    src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js'
        '?config=TeX-AMS-MML_SVG')


# Write the mathjax index html
# https://chrisvoncsefalvay.com/2020/07/25/dash-latex/
index_str_math = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            <script type="text/x-mathjax-config">
            MathJax.Hub.Config({
                tex2jax: {
                inlineMath: [ ['$','$'],],
                processEscapes: true
                }
            });
            </script>
            {%renderer%}
        </footer>
    </body>
</html>
"""


class BranchProDashApp:
    """Base class for dash apps for branching processes.

    Notes
    -----
    When deploying objects of this class in a server environment, it is
    recommended to use the lock to prevent unpleasant bugs and interference
    between threads:

    .. code-block:: python

        @app.app.callback(...)
        def callback(...):
            with app.lock:
                ...  # your callback code here
                return ...
    """
    def __init__(self):
        # Default CSS style files
        self.css = [dbc.themes.BOOTSTRAP,
                    'https://codepen.io/chriddyp/pen/bWLwgP.css']

        self.session_data = {}
        self.mathjax_html = index_str_math
        self.mathjax_script = mathjax_script

        self.lock = threading.Lock()

    def refresh_user_data_json(self, **kwargs):
        """Load the user's session data from JSON.

        To be called at the beginning of a callback so that this object
        contains the appropriate information for completing the request.

        The inputs are translated from JSON to pandas dataframes, and saved in
        the self.session_data dictionary. All previous entries in
        self.session_data are cleared.

        Parameters
        ----------
        kwargs
            Each key should be the id or name of a storage container recognized
            by the particular app, and each value should be a string containing
            the JSON data accessed from that storage.
        """
        new_session_data = {}
        for k, v in kwargs.items():
            if v is not None:
                v = pd.read_json(v)
            new_session_data[k] = v
        self.session_data = new_session_data

    def parse_contents(self, contents, filename):
        """Load a text (csv) file into a pandas dataframe.

        This method is for loading incidence number data. It expects files to
        have at least two columns, the first with title ``Time`` and the second
        with title ``Incidence Number``.

        Parameters
        ----------
        contents : str
            File contents in binary encoding
        filename : str
            Name of the file

        Returns
        -------
        html.Div
            A div which contains a message for the user.
        pandas.DataFrame
            A dataframe with the loaded file. If the file load was not
            successful, it will be None.
        """
        content_type, content_string = contents.split(',')
        _, extension = os.path.splitext(filename)

        decoded = base64.b64decode(content_string)
        try:
            if extension in ['.csv', '.txt']:
                # Assume that the user uploaded a CSV or TXT file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
            else:
                return html.Div(['File type must be CSV or TXT.']), None
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ]), None

        if ('Time' not in df.columns) or (
                'Incidence Number' not in df.columns):
            return html.Div(['Incorrect format; file must contain a `Time` \
                and `Incidence Number` column.']), None
        else:
            return html.Div(['Loaded data from: {}'.format(filename)]), df

    def _load_text(self, text):
        """Load text into an HTML div saved at self.text.
        """
        self.text = html.Div([text])

    def _load_collapsed_text(self, text, title):
        """Load text into a hidden HTML div with button at self.collapsed_text.
        """
        collapse = html.Div([
                dbc.Button(
                    title,
                    id='showhidebutton',
                    color='primary',
                ),
                dbc.Collapse(
                    dbc.Card(dbc.CardBody(text)),
                    id='collapsedtext',
                ),
            ])
        self.collapsed_text = collapse
