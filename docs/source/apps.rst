******************
Visualisation Apps 
******************

.. currentmodule:: branchpro

Overview:

- :class:`IncidenceNumberPlot`
- :class:`ReproductionNumberPlot`
- :class:`_SliderComponent`
- :class:`IncidenceNumberSimulationApp`

IncidenceNumberPlot
********************

.. autoclass:: IncidenceNumberPlot
  :members: add_data, add_simulation, update_labels, show_figure

ReproductionNumberPlot
**********************

.. autoclass:: ReproductionNumberPlot
  :members: add_ground_truth_rt, add_interval_rt, update_labels, show_figure


Sliders
*******

.. autoclass:: _SliderComponent
  :members: add_slider, get_sliders_div, slider_ids

Simulation Apps
***************

.. autoclass:: IncidenceNumberSimulationApp
  :members: add_data, add_simulator, get_sliders_ids, update_simulation