#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

"""Processing script for Hawaiian data from [1]_.

It splits the cases by county, selects certain dates, and rewrites the data file
into the format expected by our app.

References
----------
.. [1] State of Hawaii Department of Health, Disease Outbreak Control Division (DOCD), (last viewed April 14 2021).
       https://health.hawaii.gov/coronavirusdisease2019/what-you-should-know/current-situation-in-hawaii/
"""

import datetime
import os
import pandas


def write_state_data(state, start_date='3/5/2020'):
    """Write a new csv file for the selected state.

    The new filename is given by the abbreviation of the state.

    Parameters
    ----------
    state : str
        'Hawaii', 'Honolulu', 'Kauai', 'Maui'
    start_date : str
        First day (month/day/year)
    """
    # Read data
    data = pandas.read_csv(
        os.path.join(os.path.dirname(__file__), 'cases.csv'))

    # Process dates
    data['processed-date'] = [process_dates(x) for x in data['Date Added']]

    # Select data from the given state
    data = data[data['County2'] == state]

    # Split it into local, imported, and unknown cases
    # Unknown cases is treated as 'No Travel History' in the data
    imported_data = \
        data[data['Traveler'] == 'Travel History']
    imported_data = imported_data.groupby('processed-date',as_index=False).sum()
    local_data = data[data['Traveler'] == 'No Travel History']
    local_data = local_data.groupby('processed-date',as_index=False).sum()
    
    # Rename columns to the names we are using in the app
    imported_data = imported_data.rename(columns={'Total Number of Cases': 'Imported Cases'})
    local_data = local_data.rename(columns={'Total Number of Cases': 'Incidence Number'})

    # Join all sets of cases on date
    data = pandas.merge(local_data, imported_data, on=['processed-date'], how='outer')
    data = data.fillna(value=0)

    # # Process dates
    # data['processed-date'] = [process_dates(x) for x in data['date']]
    data = data.sort_values('processed-date')

    # Add a column 'Time', which is the number of days from start_date
    start = process_dates(start_date)
    data['Time'] = [(x - start).days + 1
                    for x in data['processed-date']]

    # Keep only those columns we are using
    data = data[['Time', 'Incidence Number', 'Imported Cases', 'processed-date']]
    data = data.rename(columns={'processed-date': 'date'})

    data.to_csv(
        os.path.join(os.path.dirname(__file__), '{}.csv'.format(state)),
        index=False)    

def process_dates(date):
    return datetime.date(
        *map(int, [date.split('/')[2]] + date.split('/')[:2]))

def main():
    all_states = ['Hawaii', 'Honolulu', 'Kauai', 'Maui']

    for state in all_states:
        write_state_data(state)

if __name__ == '__main__':
    main()
