"""
https://stackoverflow.com/questions/19350806/how-to-convert-columns-into-one-datetime-column-in-pandas
https://moonbooks.org/Articles/How-to-create-a-datetime-column-from-year-month-and-day-columns-in-pandas-/
https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
"""

from datetime import datetime, timedelta
import pandas as pd
import pytz

print(f"\nMethod 1")
# 1. make a DataFrame with columns: 'year', 'month', 'day', 'hour', 'minute', 'second'
# ordering is arbitrary
dtdf1 = pd.DataFrame({'year': [2001, 2001, 2001],
                      'month': [1, 2, 3],
                      'day': [4, 5, 6],
                      'hour': [7, 8, 9]})
print(dtdf1)
dti1 = pd.to_datetime(dtdf1)  # convert to DateTimeIndex intuitively
dtdf1['Datetime'] = pd.to_datetime(dtdf1)  # convert and assign to new column in the same DataFrame intuitively
print(dti1)
print(dtdf1)

print(f"\nMethod 2")
# 2. use columns of a dataFrame with arbitrary names
# ordering is arbitrary
dtdf2 = pd.DataFrame({'YY': [2001, 2001, 2001],
                      'MM': [1, 2, 3],
                      'DD': [4, 5, 6],
                      'HH': [7, 8, 9]})

dti2 = pd.to_datetime(dict(year=dtdf2['YY'], month=dtdf2['MM'], day=dtdf2['DD']))
dtdf2['Datetime'] = pd.to_datetime(dict(year=dtdf2['YY'], month=dtdf2['MM'], day=dtdf2['DD']))
print(dti2)
print(dtdf2)
dti3 = dti2.dt.tz_localize(tz=pytz.utc)
dtdf2['Datetime'].dt.tz_localize(tz='UTC')
print(dtdf2)
