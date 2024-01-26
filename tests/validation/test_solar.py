#!/bin/env python
# copied from MMIP-HP-configurator-PythonCode-git commit 3c83d1a

# Allow relative imports of modules, which only works if this project is a package with known path:
from pathlib import Path
import inspect
import sys


if __name__ == "__main__" and __package__ is None:
    # Add the name of the package top directory as the package name:
    # Note: inspect.getfile(inspect.currentframe()) ~ __file__:
    CURRENTFILE = inspect.getfile(inspect.currentframe())
    CWD = Path(CURRENTFILE).absolute().parent  # Absolute path of current working directory
    __package__ = CWD.parent.name              # The name of the package top dir, one dir up
    
    # Add the *parent directory* of the git-repo top dir to the PYTHONPATH:
    PARENTDIR = str(CWD.parent.parent)  # Parent directory of the package top dir (2 dirs up, i.e. top dir of git repo)
    if PARENTDIR not in sys.path:
        sys.path.insert(0, PARENTDIR)   # Prepend; would overrule an installed version of the package
    # print(__name__, __package__, CWD.parent.parent)
    # print(sys.path)


# from soltrack import Position, Location, Time, computeSunPosition
from solarenergy import *
# from hpconfig.weather.getweather import get_api_weather
from housemodel.weather_solar.weatherdata import read_hourly_knmi_weather_from_csv

# import pandas as pd
import numpy as np
import pytz as tz
# from dateutil import tz
import datetime as dt
import astrotool as at

import colored_traceback
colored_traceback.add_hook()

import logging
logging.basicConfig()
logger = logging.getLogger('test_solar')
logger.setLevel(logging.INFO)


def test_se_knmi_csv():
    # Location: get_location(s)
    latitude = 52.0*d2r  # >0 = N
    longitude = 5.0*d2r  # >0 = E

    # Orientation panel: get_orientation(s)
    panelAz   =  0*d2r   # S=0
    panelIncl = 45*d2r   # Horizontal = 0

    # Date and time:
    yr = 2019
    mon = 6
    day = 1
    hr = 12

    # Get weather data from KNMI API:
    # station_DeBilt = '260'
    # vars_from_station = 'SUNR'
    # dfsolar = get_api_weather(station_DeBilt, vars_from_station, yr, mon, day, yr+1, mon, day)

    # Read weather data from KNMI file:
    nltz = tz.timezone('Europe/Amsterdam')
    start_date = nltz.localize(dt.datetime(2019, 1, 1))
    end_date   = nltz.localize(dt.datetime(2019, 1, 2))
    dfsolar    = read_hourly_knmi_weather_from_csv("uurgeg_260_2011-2020_Bilt.csv", start_date, end_date)
    logger.info(dfsolar)

    # using year, month, day, hour and timezone
    azimuth, altitude, distance = sun_position_from_date_and_time(longitude, latitude, yr, mon, day, hr, timezone="Europe/Amsterdam")
    # print("%4i %2i %2i %2i               %11.5f %11.5f %11.7f" % (yr, mon, day, hr, azimuth*r2d, altitude*r2d, distance))
    logger.info(" %4i %2i %2i %2i               %11.5f %11.5f %11.7f" % (yr, mon, day, hr, azimuth*r2d, altitude*r2d, distance))

    # using timezone-naive datetime object
    myDatetime = dt.datetime(yr, mon, day, hr, 0, 0)  # tz-naive
    azimuth, altitude, distance = sun_position_from_datetime(longitude, latitude, myDatetime)
    # print("%s         %11.5f %11.5f %11.7f " % (myDatetime, azimuth*r2d, altitude*r2d, distance))
    logger.info(" %s         %11.5f %11.5f %11.7f " % (myDatetime, azimuth*r2d, altitude*r2d, distance))

    # using timezone-aware datetime object created with localize
    myDatetime = nltz.localize(dt.datetime(yr, mon, day, hr, 0, 0))  # tz-aware
    azimuth, altitude, distance = sun_position_from_datetime(longitude, latitude, myDatetime)
    # print("%s   %11.5f %11.5f %11.7f " % (myDatetime, azimuth*r2d, altitude*r2d, distance))
    logger.info(" %s   %11.5f %11.5f %11.7f " % (myDatetime, azimuth*r2d, altitude*r2d, distance))

    # print(type(dfsolar['datetime']))
    myDatetime = dfsolar['datetime'].values
    dfsolar['sunAz'], dfsolar['sunAlt'], dfsolar['sunDist'] = sun_position_from_datetime(longitude, latitude, myDatetime)

    dfsolar['Iext'] = sol_const/np.square(dfsolar['sunDist'])    # Extraterrestrial radiation [W/m^2]
    dfsolar['AM']   = airmass(dfsolar['sunAlt'])                # Air mass [-]
    dfsolar['EF']   = extinction_factor(dfsolar['AM'])          # Extinction factor [-]

    dfsolar['DHR'], dfsolar['BHR'], dfsolar['DNI'] = \
    diffuse_radiation_from_global_radiation_and_sunshine(dfsolar['Q'], dfsolar['SQ'], dfsolar['sunAlt'], dfsolar['Iext'])

    # Projection of direct sunlight on panel:
    dfsolar['cosTheta'] = cos_angle_sun_panels(panelAz, panelIncl, dfsolar['sunAz'], dfsolar['sunAlt'])
    dfsolar['panelDir'] = dfsolar['DNI'] * dfsolar['cosTheta']

    # Projection of diffuse daylight on panel:
    DoY = at.doy_from_datetime(myDatetime)  # Day of year
    dfsolar['panelDif'] = diffuse_radiation_projection_perez87(DoY, dfsolar['sunAlt'], panelIncl,
                                                              np.arccos(dfsolar['cosTheta']), dfsolar['DNI'], dfsolar['DHR'])

    # Total light on panel:
    dfsolar['panelTot'] = dfsolar['panelDir'] + dfsolar['panelDif']

    print(dfsolar)


if __name__ == "__main__":
    test_se_knmi_csv()
