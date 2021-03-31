#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

"""Processing script for Ontario State data from [1]_.

It splits the cases by state, selects certain dates, and rewrites the data file
into the format expected by our app.

References
----------
.. [1] Price, David J., et al. "Early analysis of the Australian COVID-19
       epidemic." Elife 9 (2020): e58785.
       https://elifesciences.org/articles/58785/figures#content
"""

import time
from time import mktime
import datetime
import os
import pandas


def write_state_data(start_date='Sun Mar 01 2020', end_date='Wed Mar 17 2021'):
    """Write a new csv file for the selected state.

    The new filename is given by the argument state.

    Parameters
    ----------
    start_date : str
        First day (year-month-day)
    end_date : str
        Last day (year-month-day)
    """
    # Select data from the given state
    data = pandas.read_csv(
        os.path.join(os.path.dirname(__file__), 'cases.csv'))

    # Rename date column.
    data = data.rename(columns={'category': 'date'})

    # Split it into local, imported, and unknown cases
    imported_data = \
        data[['Travel', 'date']]
    local_data = data[
        ['Close contact', 'Community spread', 'Outbreak setting', 'date']]
    unknown_data = data[['Other', 'date']]

    # Add up the columns all the the locally produced cases by date
    local_data['Close contact'] = local_data['Close contact'] + local_data[
        'Community spread'] + local_data['Outbreak setting']

    # Rename columns to the names we are using in the app
    imported_data = imported_data.rename(columns={'Travel': 'Imported Cases'})
    local_data = local_data.rename(
        columns={'Close contact': 'Incidence Number'})
    unknown_data = unknown_data.rename(columns={'Other': 'Unknown Cases'})

    # Join all sets of cases on date
    data = pandas.merge(local_data, imported_data, on='date', how='outer') \
                 .merge(unknown_data, on='date', how='outer')
    data = data.fillna(value=0)

    # Treat unknown cases as local
    data['Incidence Number'] = data['Incidence Number'] + data['Unknown Cases']

    # Process dates
    data['processed-date'] = [process_dates(x) for x in data['date']]
    data = data.sort_values('processed-date')

    # Keep only those values within the date range
    data = data[data['processed-date'] >= process_dates(start_date)]
    data = data[data['processed-date'] <= process_dates(end_date)]

    # Add a column 'Time', which is the number of days from start_date
    start = process_dates(start_date)
    data['Time'] = [(x - start).days + 1
                    for x in data['processed-date']]

    # Keep only those columns we are using
    data = data[['Time', 'Incidence Number', 'Imported Cases', 'date']]

    data.to_csv(
        os.path.join(os.path.dirname(__file__), 'ON.csv'),
        index=False)


def process_dates(date):
    struct = time.strptime(date, '%a %b %d %Y')
    return datetime.datetime.fromtimestamp(mktime(struct))


if __name__ == '__main__':
    write_state_data()
