******************
Visualisation Apps 
******************

.. currentmodule:: branchpro

Overview:

- :class:`IncidenceNumberPlot`
- :class:`ReproductionNumberPlot`
- :class:`_SliderComponent`
- :class:`IncidenceNumberSimulationApp`
- :class:`BranchProInferenceApp`

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

Inference Apps
***************

.. autoclass:: BranchProInferenceApp
  :members: add_ground_truth_rt, add_posterior, get_sliders_ids, update_inference