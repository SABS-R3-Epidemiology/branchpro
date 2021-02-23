******************
Visualisation Apps
******************

.. currentmodule:: branchpro

Overview:

- :class:`IncidenceNumberPlot`
- :class:`ReproductionNumberPlot`
- :class:`_SliderComponent`
- :class:`BranchProDashApp`
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


Apps
****

.. autoclass:: BranchProDashApp
  :members: refresh_user_data_json, parse_contents, add_text, add_collapsed_text


Simulation Apps
***************

.. autoclass:: IncidenceNumberSimulationApp
  :members: update_sliders, update_figure, update_simulation

Inference Apps
***************

.. autoclass:: BranchProInferenceApp
  :members: update_sliders, update_posterior, update_inference_figure, update_data_figure
