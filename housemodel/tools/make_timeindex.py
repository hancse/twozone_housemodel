
"""

"""

import datetime as dt
from datetime import timedelta

import pandas as pd
import pytz


def make_timeindex(yr, mon, dy, hr, offset=0):
    """ calculates a pandas DateTimeIndex from "naive" datetime information
    e.g. from NEN5060
    Convert UTC to local time (CET/CEST):

    :param yr:  array of year values
    :param mon: array of month values
    :param dy:  array of day values
    :param hr:  array of hour values
    :param offset: "naive" time offset in winter (CET = 1)
    :return:
    """
    utz = pytz.timezone('UTC')
    nltz = pytz.timezone('Europe/Amsterdam')

    naive_datelist = []                   # use list comprehension where possible
    utc_datelist = []
    for iRow in range(len(yr)):
        if (hr[iRow] == 24):        # Replace 24:00:00 with 23:59:59, so that datetime will swallow it.
                                    # This does not affect the outcome, because the Sun is well below the horizon
                                    # and this only involves a single second
            naive_datelist.append(dt.datetime(int(yr[iRow]), int(mon[iRow]), int(dy[iRow]), 23, 59, 59))
        else:
            naive_datelist.append(dt.datetime(int(yr[iRow]), int(mon[iRow]), int(dy[iRow]), int(hr[iRow]), 0, 0))

    new_datelist = []
    for d in naive_datelist:
        dnew = d - timedelta(hours=offset)
        new_datelist.append(dnew)

    utc_datelist = [utz.localize(d) for d in new_datelist]
    local_aware_datelist = [d.astimezone(nltz) for d in utc_datelist]

    for iRow in range(len(naive_datelist)):
        print(naive_datelist[iRow], "   ", new_datelist[iRow], "   ",
              utc_datelist[iRow], "   ", local_aware_datelist[iRow])

    dti = pd.to_datetime(local_aware_datelist)
    return dti


if __name__ == '__main__':
    pdti_march = make_timeindex(yr=[2004, 2004, 2004, 2004, 2004, 2004, 2004, 2004, 2004, 2004],
                          mon=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                          dy=[27, 27, 28, 28, 28, 28, 28, 28, 28, 28],
                          hr=[22, 23, 0, 1, 2, 3, 4, 5, 6, 7], offset=1)
    print('\n', pdti_march, '\n')

    pdti_october = make_timeindex(yr=[2010, 2010, 2010, 2010, 2010, 2010, 2010, 2010, 2010, 2010],
                          mon=[10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                          dy=[30, 30, 31, 31, 31, 31, 31, 31, 31, 31],
                          hr=[22, 23, 0, 1, 2, 3, 4, 5, 6, 7], offset=1)
    print('\n', pdti_october)

"""
snippet from MvdS
# Convert UTC to local time (CET/CEST):
ut = tz.timezone('UTC')

datelist = []
for iRow in range(len(Weather)):
    if (Weather.hr[iRow] == 24):  # Replace 24:00:00 with 23:59:59, so that datetime will swallow it. This does
        # not affect the outcome, because the Sun is well below the horizon and this only involves a single second

        utc = ut.localize(dt.datetime(int(Weather.year[iRow]), int(Weather.mo[iRow]),
                                      int(Weather.dy[iRow]), 23, 59, 59))  # Mark as being local time
    else:
        utc = ut.localize(dt.datetime(int(Weather.year[iRow]), int(Weather.mo[iRow]),
                                      int(Weather.dy[iRow]), Weather.hr[iRow], 0, 1))  # Mark as being local time

    lt = utc.astimezone(tz.timezone('Europe/Amsterdam'))  # Convert UTC to LT

    years.append(lt.year)
    months.append(lt.month)
    days.append(lt.day)
    hours.append(lt.hour)
"""
