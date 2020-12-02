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
    def __init__(self):
        self._sliders = []
        self._slider_ids = []

    def add_slider(self, label, new_id, init_val, min_val, max_val, step_size):
        new_slider = [
                        html.Label(label),
                        dcc.Slider(
                            id=new_id,
                            min=min_val,
                            max=max_val,
                            value=init_val,
                            marks={str(ri): str(ri) for ri in np.arange(
                                start=min_val,
                                stop=max_val+step_size,
                                step=step_size
                                )
                            },
                            step=step_size
                        )
                    ]
        self._sliders += new_slider
        self._slider_ids.append(new_id)

    def group_sliders(self):
        return html.Div(self._sliders)

    def slider_ids(self):
        return self._slider_ids
