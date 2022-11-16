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
    - Log-likelihood 
      - :class:`PoissonBranchProLogLik`

    - Log-prior
      - :class:`PoissonBranchProLogPrior`

    - Log-posterior
      - :class:`PoissonBranchProLogPosterior`

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

Log-prior Class
***************
  
.. autoclass:: PoissonBranchProLogPrior
  :members:

Log-posterior Class
*******************

.. autoclass:: PoissonBranchProLogPosterior
  :members:
