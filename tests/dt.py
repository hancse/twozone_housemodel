# https://stackoverflow.com/questions/993358/creating-a-range-of-dates-in-python
# https://stackoverflow.com/questions/5868130/generating-all-dates-within-a-given-range-in-python
# https://stackoverflow.com/questions/1060279/iterating-through-a-range-of-dates-in-python/1060330#1060330
# https://stackoverflow.com/questions/13445174/date-ranges-in-pandas
# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html
# https://www.w3resource.com/pandas/date_range.php
# https://queirozf.com/entries/python-datetime-with-timezones-examples-and-reference
# https://docs.python.org/3/library/datetime.html
# https://www.tutorialspoint.com/python-pandas-create-a-datetime-with-datetimeindex

from datetime import datetime, timedelta
# https://stackoverflow.com/questions/993358/creating-a-range-of-dates-in-python
# naive way
a = datetime.today()
numdays = 100
dateList = []
for x in range(0, numdays):
    dateList.append(a - timedelta(days=x))
print(dateList)

# marginally better (although with list comprehension :-))
base = datetime.today()
date_list = [base - timedelta(days=x) for x in range(numdays)]
print(date_list)

# standard with timezone
# https://queirozf.com/entries/python-datetime-with-timezones-examples-and-reference
from datetime import timezone

# Current datetime in UTC
# To get the current timestamp in UTC timezone, use datetime.now() passing timezone.utc:
dt1 = datetime.now(timezone.utc)
dt2 = datetime.now(tz=timezone.utc)

# Current datetime in Timezone
# Use datetime.now(timezone(timedelta(hours=<offset_hours>)))
dt3 = datetime.now(timezone(timedelta(hours=-3)))

# Datetime with Offset Timezone
# Timezones can be defined by the offset with respect to UTC.
# UTC+03:00
my_tz = timezone(timedelta(hours=3))
# May 10, 2016 at 12:30:00, on UTC-03:00
dt = datetime(2016, 5, 10, 12, 30, 0, 0, tzinfo=my_tz)

# Add UTC Timezone to naive datetime
# In other words, attach Timezone information
# to datetime objects that were created without timezone information,
# without adjusting the time
# this is a naive datetime without timezone information
naive_dt = datetime.now()
dt_with_tz = naive_dt.replace(tzinfo=timezone.utc)  # use replace because immutable

# Convert datetime to another timezone
# Use .astimezone(<other_time_zone>):
# 21:36 in UTC
base_dt = datetime.now(timezone.utc)
# becomes 23:36 in UTC+02:00
dt_in_utc_plus_2 = base_dt.astimezone(timezone(timedelta(hours=2)))

# enter Pandas...
import pandas as pd

datelist = pd.date_range(datetime.today(), periods=100).tolist()
print(datelist)

# converting Pandas timestamps to Python datetimes
pd.date_range(datetime.today(), periods=100).to_pydatetime().tolist()
pd.date_range(end=datetime.today(), periods=100).to_pydatetime().tolist()  # Python
# OR
pd.date_range(start="2018-09-09", end="2020-02-02")  # Pandas

# Get range of dates between specified start and end date (Optimized for time & space complexity):
start = datetime.strptime("21-06-2014", "%d-%m-%Y")
end = datetime.strptime("07-07-2014", "%d-%m-%Y")
date_generated = [start + timedelta(days=x) for x in range(0, (end - start).days)]
for date in date_generated:
    print(date.strftime("%d-%m-%Y"))


# You can write a generator function that returns date objects starting from today:
def date_generator():
    from_date = datetime.today()
    while True:
        yield from_date
        from_date = from_date - timedelta(days=1)

import itertools

# A more compact version using a generator expression instead of a function:
date_generator2 = (datetime.today() - timedelta(days=i) for i in itertools.count())

from dateutil import rrule

list(rrule.rrule(rrule.DAILY, count=100, dtstart=datetime.now()))

# We choose Pandas:
# Generate series from start of 2016 to end of 2020
series = pd.date_range(start='2016-01-01', end='2020-12-31', freq='D')
print(series)

series2020 = pd.date_range(start='2019-01-01 00:00:00', end='2019-12-31 23:00:00', freq='H', tz='Europe/Amsterdam')
print(series2020)
print(series2020[2130:2145])
