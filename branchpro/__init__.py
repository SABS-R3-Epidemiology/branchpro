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
from .apps import (  # noqa
    IncidenceNumberPlot,
    _SliderComponent,
    BranchProDashApp,
    IncidenceNumberSimulationApp,
    ReproductionNumberPlot,
    BranchProInferenceApp)
from ._dataset_library_api import DatasetLibrary # noqa
from .posterior import (  # noqa
    GammaDist,
    BranchProPosterior,
    BranchProPosteriorMultSI,
    LocImpBranchProPosterior,
    LocImpBranchProPosteriorMultSI)
from . import figures  # noqa

# Import main classes for negative binomial noise
from .negbin_models import (  # noqa
    NegBinBranchProModel,
    LocImpNegBinBranchProModel,
    StochasticNegBinBranchProModel,
    StochasticLocImpNegBinBranchProModel)

# Import log-likelihood classes
from .negbin_posterior import PoissonBranchProLogLik, NegBinBranchProLogLik  # noqa

# Import log-posterior classes
from .negbin_posterior import PoissonBranchProLogPosterior, NegBinBranchProLogPosterior  # noqa

import fast_posterior # noqa

# Import main classes for poisson noise and multiple population categories
from .multicat_models import (  # noqa
    MultiCatPoissonBranchProModel,
    LocImpMultiCatPoissonBranchProModel,
    AggMultiCatPoissonBranchProModel,
    LocImpAggMultiCatPoissonBranchProModel)

# Import log-likelihood classes
from .multicat_posterior import MultiCatPoissonBranchProLogLik  # noqa

# Import log-posterior classes
from .multicat_posterior import MultiCatPoissonBranchProLogPosterior  # noqa


# Import main classes for poisson binomial noise
from .poibin_models import (  # noqa
    PoiBinBranchProModel)
