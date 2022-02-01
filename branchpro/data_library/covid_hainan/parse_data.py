#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

"""Processing script for Hainan, China data from [1]_.
It rewrites the data file into the format expected by our app.
References
----------
.. [1] National Risk Management Area Classification and Prevention and Control
       Measures, (updated February 01 2022). http://wst.hainan.gov.cn/yqfk/
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
    new_data = defaultdict(lambda: [0, 0])

    for i, row in data.iterrows():
        date = row['date']
        day, month, year = date.split('/')
        date = '{}-{:02d}-{:02d}'.format(year, int(month), int(day))

        # Select imported cases
        new_data[date][1] += row['imported']

        # Select locally acquired cases
        new_data[date][0] += row['local']

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
        os.path.join(os.path.dirname(__file__), '{}.csv'.format('HN')),
        index=False)


if __name__ == '__main__':
    main()
