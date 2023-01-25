**********************
Posterior Distribution
**********************

.. currentmodule:: branchpro

Overview:
  - Exact Posterior Classes
    - :class:`BranchProPosterior`
    - :class:`BranchProPosteriorMultSI`
    - :class:`LocImpBranchProPosterior`
    - :class:`LocImpBranchProPosteriorMultSI`
    - :class:`GammaDist`

  - MCMC Sampling-based Log-Posterior Classes
    - :class:`PoissonBranchProLogLik`
    - :class:`PoissonBranchProLogPosterior`
    - :class:`NegBinBranchProLogLik`
    - :class:`NegBinBranchProLogPosterior`

Branch Process Posterior Distribution
*************************************

.. autoclass:: BranchProPosterior
  :members:

Branch Process Posterior Distribution with Multiple Serial Intervals
********************************************************************

.. autoclass:: BranchProPosteriorMultSI
  :members:

Local and Imported Branch Process Posterior Distribution
********************************************************

.. autoclass:: LocImpBranchProPosterior
  :members:

Local and Imported Branch Process Posterior Distribution with Multiple Serial Intervals
***************************************************************************************

.. autoclass:: LocImpBranchProPosteriorMultSI
  :members:

Gamma distribution
******************

.. autoclass:: GammaDist
  :members:

New Posterior Classes using MCMC Sampling algorithms
****************************************************
Branch Process with Poisson Noise
*********************************
Log-likelihood Class
********************

.. autoclass:: PoissonBranchProLogLik
  :members:

Log-posterior Class
*******************

.. autoclass:: PoissonBranchProLogPosterior
  :members:

Branch Process with Negative Binomial Noise
*******************************************
Log-likelihood Class
********************

.. autoclass:: NegBinBranchProLogLik
  :members:

Log-posterior Class
*******************

.. autoclass:: NegBinBranchProLogPosterior
  :members:
