#
# SimulationController Class
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import numpy as np

from branchpro import ForwardModel


class SimulationController:
    """SimulationController Class:
    Class for the simulation of models in any of
    the subclasses in the ``ForwardModel`` class.

    Parameters
    ----------
    model
        (ForwardModel) Instance of the ``ForwardModel`` class used for the
        simulation.
    start_sim_time
        (integer) Time from which we start running the SimulationController.
    end_sim_time
        (integer) Time at which we stop running the SimulationController.

    Methods
    -------
    switch_resolution: change the number of points we wish to keep from our
        simulated sample of incidences.
    run: operates the ``simulate`` method present in any subclass of the
        ``ForwardModel``.

    Always apply method switch_resolution before calling
    :meth:`SimulationController.run` for a change of resolution!

    """

    def __init__(self, model, start_sim_time, end_sim_time):
        if not isinstance(model, ForwardModel):
            raise TypeError(
                'Model needs to be a subclass of the branchpro.ForwardModel')

        self._model = model
        self._sim_end_points = (start_sim_time, end_sim_time)

        # Set default regime 'simulate in full'
        default_regime = np.linspace(start=start_sim_time, stop=end_sim_time)
        self._regime = np.ceil(default_regime).astype(int)

    def switch_resolution(self, num_points):
        """
        Change the number of points we wish to keep from our simulated sample
        of incidences.

        Parameters
        ----------
        num_points
            (integer) number of points we wish to keep from our simulated
            sample of incidences.

        """
        start_sim_time, end_sim_time = self._sim_end_points
        new_regime = np.linspace(
            start=start_sim_time, stop=end_sim_time, num=num_points)

        # Transform evaluation points into integers
        self._regime = np.ceil(new_regime).astype(int)

    def run(self, parameters):
        """
        Operates the ``simulate`` method present in any subclass of the
        ``ForwardModel``.

        Parameters
        ----------
        parameters
            An ordered sequence of parameter values.

        """
        return self._model.simulate(parameters, self._regime)
