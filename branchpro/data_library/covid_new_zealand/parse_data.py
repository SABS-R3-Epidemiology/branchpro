#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

"""Processing script for New Zealand data from [1]_.
It rewrites the data file into the format expected by our app.
References
----------
.. [1] Institute of Environmental Science and Research (ESR), (updated April 14
       2021). https://nzcoviddashboard.esr.cri.nz/#!/source
"""

import datetime
import os
import pandas
from collections import defaultdict


def main():
    """
    Rewrite a new csv file for the data in the desired format.
    We combine the daily import-related and imported cases as the imported
    case, and we add the daily under investigation to the daily locally
    acquired cases (with unknown source and epidemiologically linked)
    """
    # Read the original data
    data = pandas.read_csv(
        os.path.join(os.path.dirname(__file__), 'cases.csv'))
    # Initialize a dictionary for the new data
    new_data = defaultdict(lambda : [0, 0])

    for i, row in data.iterrows():
        date = row['Date']
        day, month, year = date.split('/')
        date = '{}-{:02d}-{:02d}'.format(year, int(month),int(day))

        # Add up the import-related case and imported case
        new_data[date][1] += row['Daily import-related case']
        new_data[date][1] += row['Daily imported case']

        # add the daily under investigation to the daily locally acquired cases
        new_data[date][0] += row['Daily locally acquired case  unknown source']
        new_data[date][0] += row['Daily locally acquired  epidemiologically linked']
        new_data[date][0] += row['Daily under investigation']

    all_dates = sorted(list(new_data.keys()))

    # Create a pandas DataFrame for the data
    data = pandas.DataFrame()
    data['Incidence Number'] = [new_data[d][0] for d in all_dates]
    data['Imported Cases'] = [new_data[d][1] for d in all_dates]
    data['date'] = all_dates

    start_date = all_dates[0]

    start = datetime.date(*map(int, start_date.split('-')))
    data['Time'] = [(datetime.date(*map(int, x.split('-'))) - start).days + 1
                    for x in data['date']]

    # Name the columns
    data = data[['Time', 'Incidence Number', 'Imported Cases', 'date']]

    # Convert the file to csv
    data.to_csv(
        os.path.join(os.path.dirname(__file__), '{}.csv'.format('NZ')),
        index=False)


if __name__ == '__main__':
    main()
