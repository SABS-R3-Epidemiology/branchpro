
import datetime
import os
import pandas
from collections import defaultdict


def main():
    data = pandas.read_csv('source_case_curve__202104140300.csv')
    new_data = defaultdict(lambda : [0, 0])

    for i, row in data.iterrows():
        date = row['Date']
        day, month, year = date.split('/')
        date = '{}-{:02d}-{:02d}'.format(year, int(month),int(day))

        new_data[date][1] += row['Daily import-related case']
        new_data[date][1] += row['Daily imported case']

        new_data[date][0] += row['Daily locally acquired case  unknown source']
        new_data[date][0] += row['Daily locally acquired  epidemiologically linked']
        new_data[date][0] += row['Daily under investigation']

    all_dates = sorted(list(new_data.keys()))

    data = pandas.DataFrame()
    data['Incidence Number'] = [new_data[d][0] for d in all_dates]
    data['Imported Cases'] = [new_data[d][1] for d in all_dates]
    data['date'] = all_dates

    start_date = all_dates[0]

    start = datetime.date(*map(int, start_date.split('-')))
    data['Time'] = [(datetime.date(*map(int, x.split('-'))) - start).days + 1
                    for x in data['date']]

    data = data[['Time', 'Incidence Number', 'Imported Cases', 'date']]

    data.to_csv(
        os.path.join(os.path.dirname(__file__), '{}.csv'.format('HI')),
        index=False)


if __name__ == '__main__':
    main()
