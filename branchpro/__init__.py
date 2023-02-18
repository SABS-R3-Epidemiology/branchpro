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
from .models import ForwardModel, BranchProModel, LocImpBranchProModel    # noqa
from .simulation import SimulationController  # noqa
from .apps import IncidenceNumberPlot, _SliderComponent, BranchProDashApp, IncidenceNumberSimulationApp, ReproductionNumberPlot, BranchProInferenceApp # noqa
from ._dataset_library_api import DatasetLibrary # noqa
from .posterior import GammaDist, BranchProPosterior, BranchProPosteriorMultSI, LocImpBranchProPosterior, LocImpBranchProPosteriorMultSI # noqa
from . import figures  # noqa

# Import main classes for negative binomial noise
from .new_models import NegBinBranchProModel, LocImpNegBinBranchProModel  # noqa

# Import log-likelihood classes
from .new_posterior import PoissonBranchProLogLik, NegBinBranchProLogLik  # noqa

# Import log-posterior classes
from .new_posterior import PoissonBranchProLogPosterior, NegBinBranchProLogPosterior  # noqa

import fast_posterior
