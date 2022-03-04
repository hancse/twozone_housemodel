#!/bin/env python3

""" test_weatherdata.py:  test for code in the directory weather/.
    2021-05-10, MvdS: initial version.
"""
import context
import colored_traceback
colored_traceback.add_hook()

import datetime as dt
import pytz as tz
from housemodel.weather_solar.weatherdata import (read_hourly_knmi_weather_from_csv,
                                                  read_nen_weather_from_csv)
# from insolation import insolation_on_surface_from_daterange

import solarenergy as se

d2r = se.d2r
r2d = se.r2d


def test_read_knmi_weather():
    """Test weatherdata.py / read_knmi_weather()"""

    # Set the date/time range in timezone-aware datetime objects:
    nltz = tz.timezone('Europe/Amsterdam')
    start_time = nltz.localize(dt.datetime(2019, 1, 1))
    end_time = nltz.localize(dt.datetime(2019, 1, 2))

    # Read KNMI data:
    weather_knmi = read_hourly_knmi_weather_from_csv("uurgeg_260_2011-2020_Bilt.csv",
                                                     start_time, end_time)

    row0 = weather_knmi.loc[0]
    print(row0)

    assert (row0['YYYYMMDD'] == 20181231)
    assert (row0['HH'] == 24)
    assert (row0['T10'] == 76.0)
    assert (row0['Q'] == 0.0)
    assert (row0['WW'] == 10.0)
    # assert(row0['']==)

    weather_nen = read_nen_weather_from_csv()
    print(weather_nen)

    return


def test_insolation_on_surface_from_daterange():
    """Test insolation.py / insolation_on_surface_from_daterange()."""

    # Set the location of the surface (rad):
    geo_lat = 52.0 * d2r  # >0 = N
    geo_lon = 5.0 * d2r  # >0 = E

    # Set the orientation of the surface (rad):
    ori_az = 0 * d2r  # S=0
    ori_incl = 45 * d2r  # Horizontal = 0

    # Set the date/time range in timezone-aware datetime objects:
    nltz = tz.timezone('Europe/Amsterdam')
    start_date = nltz.localize(dt.datetime(2019, 6, 1))
    end_date = nltz.localize(dt.datetime(2019, 6, 2))

    # df_insol = insolation_on_surface_from_daterange(start_date,end_date, geo_lon,geo_lat, ori_az,ori_incl)

    row0 = df_insol.loc[12]
    print(row0)
    assert (row0['YYYYMMDD'] == 20190601)
    assert (row0['HH'] == 11)
    assert (row0['T'] == 21.7)
    assert (row0['Q'] > 785)
    assert (row0['Q'] < 793)
    # assert(row0['WW']==2.0)

    assert (row0['sunAz'] > -0.35)
    assert (row0['sunAz'] < -0.25)
    assert (row0['sunAlt'] > 1.00)
    assert (row0['sunAlt'] < 1.06)

    assert (row0['Iext'] > 1320)
    assert (row0['Iext'] < 1330)
    assert (row0['AM'] > 1.15)
    assert (row0['AM'] < 1.17)
    assert (row0['EF'] > 1.40)
    assert (row0['EF'] < 1.45)

    assert (row0['DHR'] > 68)
    assert (row0['DHR'] < 74)
    assert (row0['BHR'] > 710)
    assert (row0['BHR'] < 730)
    assert (row0['DNI'] > 825)
    assert (row0['DNI'] < 845)

    assert (row0['cosTheta'] > 0.93)
    assert (row0['cosTheta'] < 0.97)

    assert (row0['panelDir'] > 790)
    assert (row0['panelDir'] < 802)
    assert (row0['panelDif'] > 75)
    assert (row0['panelDif'] < 85)
    assert (row0['panelTot'] > 870)
    assert (row0['panelTot'] < 884)

    return


if __name__ == "__main__":
    test_read_knmi_weather()
    # test_insolation_on_surface_from_daterange()
