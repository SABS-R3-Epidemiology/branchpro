#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import unittest

import dash_core_components as dcc
import dash_html_components as html

import branchpro as bp


class Test_SliderComponent(unittest.TestCase):
    """
    Test the '_SliderComponent' class.
    """
    def test__init__(self):
        bp._SliderComponent()

    def test_add_slider(self):
        sliders = bp._SliderComponent()
        sliders.add_slider('param1', '1', 0, 0, 1, 0.5)
        self.assertEqual(
            [vars(x) for x in sliders._sliders],
            [
                vars(x) for x in [
                    html.Label('param1'),
                    dcc.Slider(
                        id='1',
                        min=0,
                        max=1,
                        value=0,
                        marks={str(ri): str(ri) for ri in [0.0, 0.5, 1.0]},
                        step=0.5
                    )
                    ]
            ]
        )

    def test_group_sliders(self):
        sliders = bp._SliderComponent()
        sliders.add_slider('param1', '1', 0, 0, 1, 0.5)
        sliders.add_slider('param2', '2', 0.5, 0, 1, 0.25)
        sliders.group_sliders()
        self.assertEqual(
            [vars(x) for x in sliders.group_sliders()],
            [vars(x) for x in html.Div(
                [
                    html.Label('param1'),
                    dcc.Slider(
                        id='1',
                        min=0,
                        max=1,
                        value=0,
                        marks={str(ri): str(ri) for ri in [0.0, 0.5, 1.0]},
                        step=0.5
                    ),

                    html.Label('param2'),
                    dcc.Slider(
                        id='2',
                        min=0,
                        max=1,
                        value=0.5,
                        marks={
                            str(ri): str(ri) for ri in [
                                0.0, 0.25, 0.5, 0.75, 1.0
                                ]
                            },
                        step=0.25
                    )
                ]
            )
            ]
        )

    def test_slider_ids(self):
        sliders = bp._SliderComponent()
        sliders.add_slider('param1', '1', 0, 0, 1, 0.5)
        sliders.add_slider('param2', '2', 0.5, 0, 1, 0.25)
        self.assertEqual(sliders._slider_ids, ['1', '2'])
