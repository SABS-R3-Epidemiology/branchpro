#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

"""Processing script for COVID serial interval data from [1]_.

It generates multiple serial intervals from a lognormal distribution
with parameters as given in the reference.
https://github.com/aakhmetz/COVID19SerialInterval

References
----------
.. [1] Nishiura, Hiroshi, Natalie M. Linton, and Andrei R. Akhmetzhanov.
       "Serial interval of novel coronavirus (COVID-19) infections."
       International journal of infectious diseases 93 (2020): 284-286.
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

    # Split it into param1 and param2 of the lognormal distribution
    # Keep only 1000 of the pairs
    s_data = data['param2'].to_numpy()[:1000]
    scale_data = data['param1'].to_numpy()[:1000]

    si_data = np.zeros((60, 1000))

    # Compute lognormal pdf for given paired parameters and
    # Discretize distribution and record it
    for i, _ in enumerate(s_data):
        w_dist = sc.stats.lognorm(
            s=s_data[i], scale=np.exp(scale_data[i]))
        disc_w = [w_dist.cdf(s+0.5) - w_dist.cdf(s-0.5)
                  for s in np.arange(1, 61)]
        si_data[:, i] = disc_w

    # Transform recorded matrix of serial intervals to csv file
    path = os.path.join(
            os.path.dirname(__file__), '{}.csv'.format(name))
    np.savetxt(path, si_data, delimiter=',')


if __name__ == '__main__':
    write_ser_int_data('si-epinow')
