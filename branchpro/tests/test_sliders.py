#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import unittest

import numpy as np

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
        sliders.add_slider('param2', '2', 0, 0, 1, 0.05)
        self.assertEqual(sliders._sliders[0].children, 'param1')
        self.assertEqual(sliders._sliders[1].id, '1')
        self.assertEqual(sliders._sliders[1].min, 0)
        self.assertEqual(sliders._sliders[1].max, 1)
        self.assertEqual(sliders._sliders[1].value, 0)
        self.assertEqual(
            sliders._sliders[1].marks,
            {str(ri): str(ri) for ri in [0.0, 0.5, 1.0]}
            )
        self.assertEqual(sliders._sliders[1].step, 0.5)

        self.assertEqual(sliders._sliders[2].children, 'param2')
        self.assertEqual(sliders._sliders[3].id, '2')
        self.assertEqual(sliders._sliders[3].min, 0)
        self.assertEqual(sliders._sliders[3].max, 1)
        self.assertEqual(sliders._sliders[3].value, 0)
        self.assertEqual(
            sliders._sliders[3].marks,
            {str(ri): str(ri) for ri in np.linspace(0, 1, 11)}
            )
        self.assertEqual(sliders._sliders[3].step, 0.05)

    def test_get_sliders_div(self):
        sliders = bp._SliderComponent()
        sliders.add_slider('param1', '1', 0, 0, 1, 0.5)
        sliders.add_slider('param2', '2', 0.5, 0, 1, 0.25)
        div = sliders.get_sliders_div()

        self.assertEqual(div.children[0].children, 'param1')
        self.assertEqual(div.children[1].id, '1')
        self.assertEqual(div.children[1].min, 0)
        self.assertEqual(div.children[1].max, 1)
        self.assertEqual(div.children[1].value, 0)
        self.assertEqual(
            div.children[1].marks,
            {str(ri): str(ri) for ri in [0.0, 0.5, 1.0]}
            )
        self.assertEqual(div.children[1].step, 0.5)

        self.assertEqual(div.children[2].children, 'param2')
        self.assertEqual(div.children[3].id, '2')
        self.assertEqual(div.children[3].min, 0)
        self.assertEqual(div.children[3].max, 1)
        self.assertEqual(div.children[3].value, 0.5)
        self.assertEqual(
            div.children[3].marks,
            {str(ri): str(ri) for ri in [0.0, 0.25, 0.5, 0.75, 1.0]}
        )
        self.assertEqual(div.children[3].step, 0.25)

    def test_slider_ids(self):
        sliders = bp._SliderComponent()
        sliders.add_slider('param1', '1', 0, 0, 1, 0.5)
        sliders.add_slider('param2', '2', 0.5, 0, 1, 0.25)
        self.assertEqual(sliders.slider_ids(), ['1', '2'])
