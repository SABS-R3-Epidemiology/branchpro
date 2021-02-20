#
# BranchProDashApp Class
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import pandas as pd
import dash_defer_js_import as dji  # For mathjax
import dash_bootstrap_components as dbc


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
    """
    def __init__(self):
        # Default CSS styles
        self.css = [dbc.themes.BOOTSTRAP,
                    'https://codepen.io/chriddyp/pen/bWLwgP.css']

        self.session_data = {}
        self.mathjax_html = index_str_math
        self.mathjax_script = mathjax_script

    def refresh_user_data_json(self, **kwargs):
        """Load the user's session data from JSON.

        To be called at the beginning of an HTTP request so that this object
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
        self.session_data = {}
        for k, v in kwargs.items():
            if v is not None:
                v = pd.read_json(v)
            self.session_data[k] = v
