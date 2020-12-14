#
# Root of the branchpro module.
# Provides access to all shared functionality (models, simulation, etc.).
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#
"""branchpro is a Branching Processes modelling library.
It contains functionality for modelling, simulating, and visualising the
number of cases of infections by day during an outbreak of the influenza virus.
"""

# Import version info
from .version_info import VERSION_INT, VERSION  # noqa

# Import main classes
from .models import ForwardModel, BranchProModel    # noqa
from .simulation import SimulationController  # noqa
from .apps import IncidenceNumberPlot, _SliderComponent, IncidenceNumberSimulationApp # noqa
from ._dataset_library_api import DatasetLibraryAPI # noqa
