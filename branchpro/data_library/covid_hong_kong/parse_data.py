#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

"""Processing script for Hong Kong data from [1]_.

It splits the cases by state, selects certain dates, and rewrites the data file
into the format expected by our app.

References
----------
.. [1] Department of Health. (2020, January 27). Data in Coronavirus Disease
       (COVID-19) - Latest local situation of COVID-19 (English). DATA.GOV.HK.
       (Retrieved February 2, 2022).
       https://data.gov.hk/en-data/dataset/hk-dh-chpsebcddr-novel-infectious-agent/resource/ec4b49af-83e0-4c71-a3ba-14120e453b9d
"""

import datetime
import os
import pandas


def write_state_data(start_date='23/01/2020', end_date='31/03/2020'):
    """Write a new csv file for the data.

    Parameters
    ----------
    start_date : str
        First day (Week_day Month Day Year)
    end_date : str
        Last day (Week_day Month Day Year)
    """
    # Select data from the given state
    data = pandas.read_csv(
        os.path.join(os.path.dirname(__file__), 'cases.csv'))

    # Keep only the Confirmed and Asymptomatic cases to avoid
    # double counting
    data = data[data['Case status*'].isin(['Confirmed', 'Asymptomatic'])]

    # Process dates
    data['processed-date'] = [process_dates(x) for x in data['Report date']]

    # Keep only those values within the date range
    data = data[data['processed-date'] >= process_dates(start_date)]
    data = data[data['processed-date'] <= process_dates(end_date)]

    # Add counter column
    data['count'] = 1

    # Split it into local, imported, and unknown cases
    imported_data = data[data['Classification*'] == 'Imported case']
    imported_data = imported_data.groupby(
        'processed-date', as_index=False).sum()

    local_data = data[data['Classification*'].isin([
        'Local case', 'Epidemiologically linked with imported case',
        'Epidemiologically linked with local case'])]
    local_data = local_data.groupby(
        'processed-date', as_index=False).sum()

    unknown_data = data[data['Classification*'].isin([
        'Epidemiologically linked with possibly local case',
        'Possibly import-related case', 'Possibly local case'])]
    unknown_data = unknown_data.groupby(
        'processed-date', as_index=False).sum()

    # Rename columns to the names we are using in the app
    imported_data = imported_data.rename(
        columns={'count': 'Imported Cases'})
    local_data = local_data.rename(
        columns={'count': 'Incidence Number'})
    unknown_data = unknown_data.rename(
        columns={'count': 'Unknown Cases'})

    # Join all sets of cases on date
    data = pandas.merge(
        local_data, imported_data, on='processed-date', how='outer') \
        .merge(unknown_data, on='processed-date', how='outer')
    data = data.fillna(value=0)
    data = data.sort_values('processed-date')

    # Treat unknown cases as local
    data['Incidence Number'] = data['Incidence Number'] + data['Unknown Cases']

    # Add a column 'Time', which is the number of days from start_date
    start = process_dates(start_date)
    data['Time'] = [(x - start).days + 1
                    for x in data['processed-date']]

    # Keep only those columns we are using
    data = data[
        ['Time', 'Incidence Number', 'Imported Cases', 'processed-date']]
    data = data.rename(columns={'processed-date': 'date'})

    data.to_csv(
        os.path.join(os.path.dirname(__file__), 'HK.csv'),
        index=False)


def process_dates(date):
    day, month, year = date.split('/')
    new_date = '{}-{:02d}-{:02d}'.format(year, int(month), int(day))
    return datetime.date(
        *map(int, new_date.split('-')))


if __name__ == '__main__':
    write_state_data()
