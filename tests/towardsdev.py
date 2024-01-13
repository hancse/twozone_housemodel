# https://towardsdev.com/date-and-time-objects-in-python-everything-you-need-to-know-10aa3bf121be
# Import datetime submodule
from datetime import datetime

dt = datetime(year=2020, month=11, day=20,
              hour=14, minute=20, second=25,
              microsecond=50000)

print(dt)

print(dt.year, dt.month, dt.day)

# Create a random date and print
dt = datetime(2015, 8, 14, 7, 15, 49)
print("Previous date:", dt)
# Store today's date
today = datetime.now() # or .now()
# Change the random today with today's components and print
dt_changed = dt.replace(year=today.year,
                        month=today.month,
                        day=today.day,
                        hour=today.hour,
                        minute=today.minute,
                        second=today.second)

print("     New date:", dt_changed)

dt_tuple = dt_changed.timetuple()
print("Year:", dt_tuple[0], "\nMonth:", dt_tuple[1])

# Example one
date_st = '2020-11-20'
# Matching format string
format_st = '%Y-%m-%d'
dt = datetime.strptime(date_st, format_st)
print(dt)

# Example two
date_st = '14:05:26 12/12/97'
# Matching format string
format_st = '%H:%M:%S %d/%m/%y'
dt = datetime.strptime(date_st, format_st)
print(dt)

# Today's date with time
now = datetime.now()
# Convert to timestamp
ts = datetime.timestamp(now)
print(ts)

ts = 2147483648
end = datetime.fromtimestamp(ts)
print(end)

# https://towardsdev.com/giant-mess-dealing-with-timezones-and-daylight-saving-time-in-python-7222d37658cf

from datetime import timedelta, timezone
import pandas as pd

# rides = pd.read_csv('data/tripdata.csv',
                 #   usecols=['duration_sec', 'start_time', 'end_time'],
                #    parse_dates=['start_time', 'end_time'])
# rides.head()