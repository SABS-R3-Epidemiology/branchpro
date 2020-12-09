#
# Sliders Class
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import dash_core_components as dcc
import dash_html_components as html

import numpy as np


class _SliderComponent():
    """_SliderComponent Class
    Stores slider information for sliders in branchpro Dash app.
    """
    def __init__(self):
        self._sliders = []
        self._slider_ids = []

    def add_slider(self, label, new_id, init_val, min_val, max_val, step_size):
        """
        Creates a new slider with label for the Dash app plot.

        Parameters
        ----------
        label
            Title text shown above slider.
        new_id
            ID of slider (internal, not shown in app).
        init_val
            Initial slider value (default position).
        min_val
            Minimum (leftmost) slider value.
        max_val
            Maximum (rightmost) slider value.
        step_size
            Incremement between slider values.
        """
        marks_list = np.arange(
                                start=min_val,
                                stop=max_val+step_size,
                                step=step_size
                                )
        if marks_list.size > 11:
            marks_list = np.linspace(
                                        start=min_val,
                                        stop=max_val,
                                        num=11
                                        )

        new_slider = [
                        html.Label(label),
                        dcc.Slider(
                            id=new_id,
                            min=min_val,
                            max=max_val,
                            value=init_val,
                            marks={str(ri): str(ri) for ri in marks_list},
                            step=step_size
                        )
                    ]
        self._sliders += new_slider
        self._slider_ids.append(new_id)

    def get_sliders_div(self):
        """
        Combines all sliders into a html.Div object.
        """
        return html.Div(self._sliders)

    def slider_ids(self):
        """
        Returns list of all slider IDs.
        """
        return self._slider_ids
