#
# BranchProDashApp Class
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import pandas as pd


class BranchProDashApp:
    """Base class for dash apps for branching processes.
    """
    def __init__(self):
        self.session_data = {}

    def refresh_user_data(self, **kwargs):
        """Load the user's session data.

        To be called at the beginning of an HTTP request so that this object
        contains the appropriate information for completing the request.

        The inputs are translated from JSON to pandas dataframes, and saved in
        the self.session_data dictionary. All previous entries in
        self.session_data are cleared.

        Parameters
        ----------
        kwargs
            Each key should be the id of a storage HTML div recognized by the
            particular app, and each value should be a string containing the
            JSON data accessed from that container.
        """
        self.session_data = {}
        for k, v in kwargs.items():
            if v is not None:
                v = pd.read_json(v)
            self.session_data[k] = v
