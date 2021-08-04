#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

r"""Processing script for COVID serial interval data from [1]_.

It generates multiple serial intervals from a lognormal distribution
with parameters as given in the reference.
https://github.com/aakhmetz/COVID19SerialInterval

To covert the continuous serial interval estimates into discrete daily terms,
we use the method of [2]_ (appendix 11).

With the assumption that the exact time of infection from day :math:`t` is
uniformly distributed, :math:`f_U(u) = \mathbbm{1}_{k-1<u<k+1}(1-|u-k|)`
describes the distribution of the delay :math:`u` between true times of 2
infections on days :math:`t` and :math:`t+k`. The probability density
function is weighted with the probability function of each delay to
discretise the serial interval distribution.

.. math::
    \begin{align} w_k &= \int_{k-1}^{k+1}f_{SI}(u)f_U(u)du \\
    &= (1+k)F_{SI}(k+1) - 2kF_{SI}(k) + (K-1)F_{SI}(k-1) +
    \int_{k-1}^{k}uf_{SI}(u)du - \int_{k}^{k+1}uf_{SI}(u)du \end{align}

References
----------
.. [1] Nishiura, Hiroshi, Natalie M. Linton, and Andrei R. Akhmetzhanov.
       "Serial interval of novel coronavirus (COVID-19) infections."
       International journal of infectious diseases 93 (2020): 284-286.
.. [2] Cori, Anne, et al. "A new framework and software to estimate
       time-varying reproduction numbers during epidemics." American Journal of
       Epidemiology 178.9 (2013): 1505-1512.
"""

import os
import pandas as pd
import numpy as np
import scipy as sc
import scipy.stats


def write_ser_int_data(name):
    """Write a new csv file for 60-day-long serial intervals
    to be used for the analysis of the Australian data.

    Parameters
    ----------
    name
        Name given to the serial intervals file.

    """
    # Select data from the given state
    path = os.path.join(
            os.path.dirname(__file__), 'lognormal_param.csv')
    data = pd.read_csv(path, dtype=np.float64)

    # Seed the generator
    np.random.seed(0)

    # Split it into param1 and param2 of the lognormal distribution
    # Keep only 1000 of the pairs
    data = data.sample(1000)
    s_data = data['param2'].to_numpy()
    scale_data = data['param1'].to_numpy()

    si_data = np.zeros((60, 1000))

    # Compute lognormal pdf for given paired parameters and
    # Discretize distribution and record it
    for i, _ in enumerate(s_data):
        w_dist = sc.stats.lognorm(
            s=s_data[i], scale=np.exp(scale_data[i]))

        # Get the density weighted by the delay
        def weighted_density(u):
            return u * w_dist.pdf(u)

        # Calculate discrete serial interval terms, for k = 1,2,3,...
        disc_w = [
            (1 + k) * w_dist.cdf(k + 1)
            - 2 * k * w_dist.cdf(k)
            + (k - 1) * w_dist.cdf(k - 1)
            + scipy.integrate.quad(weighted_density, k - 1, k)[0]
            - scipy.integrate.quad(weighted_density, k, k + 1)[0]
            for k in np.arange(1, 61)
            ]

        si_data[:, i] = disc_w

    # Transform recorded matrix of serial intervals to csv file
    path = os.path.join(
            os.path.dirname(__file__), '{}.csv'.format(name))
    np.savetxt(path, si_data, delimiter=',')


if __name__ == '__main__':
    write_ser_int_data('si-epinow')
