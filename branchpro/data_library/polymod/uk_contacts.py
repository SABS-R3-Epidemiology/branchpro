#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""Processing script for the contact matrices and Google mobility data from
[1]_ and [2]_.

It computes the baseline and time-dependent region-specific contact matrices
which are then stored in separate csv files.

References
----------
.. [1] Prem K, Cook AR, Jit M (2017) Projecting social contact matrices in 152
       countries using contact surveys and demographic data. PLOS Computational
       Biology 13(9): e1005697.
       https://doi.org/10.1371/journal.pcbi.1005697

.. [2] COVID-19 Community Mobility Reports
       https://www.google.com/covid19/mobility/
"""

import os
import pandas as pd
import numpy as np


def read_contact_matrices(
        file_index: int = 2,
        state: str = 'United Kingdom of Great Britain'):
    """
    Read the baseline contact matices for different activities recorded
    for the given state from the appropriate Excel file.

    Parameters
    ----------
    file_index : int
        Index of the file containg the baseline contact matrices
        used in the model.
    state : str
        Name of the country whose the baseline contact matrices are used in
        the model.

    Retruns
    -------
    list of pandas.Dataframe
        List of the baseline contact matices for each activitiy recorded
        for different for the given state and population frequencies.

    """
    # Select contact matrices from the given state and activity
    path = os.path.join(
            os.path.dirname(__file__), 'raw_contact_matrices/')
    school = pd.read_excel(
        os.path.join(path, 'MUestimates_school_{}.xlsx').format(file_index),
        sheet_name=state, header=None).to_numpy()
    home = pd.read_excel(
        os.path.join(path, 'MUestimates_home_{}.xlsx').format(file_index),
        sheet_name=state, header=None).to_numpy()
    work = pd.read_excel(
        os.path.join(path, 'MUestimates_work_{}.xlsx').format(file_index),
        sheet_name=state, header=None).to_numpy()
    others = pd.read_excel(
        os.path.join(path, 'MUestimates_other_locations_{}.xlsx').format(
            file_index),
        sheet_name=state, header=None).to_numpy()
    pop_freq = pd.read_csv(
        os.path.join(path, 'UK_Ages.csv'), header=None)

    return school, home, work, others, pop_freq


def change_age_groups(matrix: np.array, pop_freq: np.array):
    """
    Reprocess the contact matrix so that it has the appropriate age groups.

    Parameters
    ----------
    matrix : numpy.array
        Contact matrix with old age groups.
    pop_freq : numpy.array
        Population frequencies

    Returns
    -------
    numpy.array
        New contact matrix with correct age groups.

    """
    new_matrix = np.empty((3, 3))

    ind_old = [
        np.array(range(0, 4)),
        np.array(range(4, 13)),
        np.array(range(13, 16))]

    pop_freq = pop_freq.to_numpy()[0, :]
    frac_pop_over75 = pop_freq[:15].tolist()
    frac_pop_over75.append(np.sum(pop_freq[15:]))
    frac_pop_over75 = np.asarray(frac_pop_over75)

    print(frac_pop_over75.shape, matrix.shape)

    for i in range(3):
        for j in range(3):
            new_matrix[i, j] = np.average(
                np.sum(
                    matrix[ind_old[i][:, None], ind_old[j]], axis=0),
                weights=frac_pop_over75[ind_old[j]])

    return new_matrix


def main():
    """
    Combines the timelines of deviation percentages and baseline
    activity-specific contact matrices to get weekly, region-specific
    contact matrices.

    Returns
    -------
    csv
        Processed files for the baseline and region-specific time-dependent
        contact matrices for each different region found in the default file.

    """
    activity = ['school', 'home', 'work', 'others']
    baseline_matrices = read_contact_matrices()[:-1]
    pop_freq = read_contact_matrices()[-1]
    baseline_contact_matrix = np.zeros_like(baseline_matrices[0])

    for ind, a in enumerate(activity):
        baseline_contact_matrix += baseline_matrices[ind]

    # Transform recorded matrix of serial intervals to csv file
    path_ = os.path.join(
        os.path.dirname(__file__), 'final_contact_matrices/')
    path6 = os.path.join(path_, 'BASE.csv')

    np.savetxt(
        path6, change_age_groups(baseline_contact_matrix, pop_freq),
        delimiter=',')


if __name__ == '__main__':
    main()
