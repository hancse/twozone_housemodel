
import context

from housemodel.weather_solar.weatherdata import (read_nen_weather_from_xl,
                                                  NENdatehour2datetime,
                                                  get_hourly_knmi_weather_from_api)
import pandas as pd
from pytz import timezone
import numpy as np

from solarenergy import *
from pvlib.solarposition import get_solarposition, nrel_earthsun_distance
from pvlib.atmosphere import get_relative_airmass, alt2pres, get_absolute_airmass
from pvlib.irradiance import get_extra_radiation

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")


def test_positions():
    # read NEN5060 data from spreadsheet NEN5060-2018.xlsx "as is" into pandas DataFrame
    # use first header line as column names
    df_xl = read_nen_weather_from_xl()
    # generate and insert timezone-aware UTC and local timestamps (with DST)
    df_xl = NENdatehour2datetime(df_xl)

    times5060 = df_xl['local_time']
    # Location of my solar panels:
    lon_deg = 5.0
    lat_deg = 52.0
    lon_rad = 5.0 * d2r  # Geographic longitude (>0 for eastern hemisphere; ° -> rad)
    lat_rad = 52.0 * d2r  # Geographic latitude  (>0 for northern hemisphere; ° -> rad)

    # VALIDATION of solar position calculations in PVLIB and solarenergy
    pos_pv = get_solarposition(times5060, lat_deg, lon_deg, method='nrel_numpy')

    # solarenergy
    # times2020_dt = times5060.to_pydatetime()
    sunAz, sunAlt, sunDist = sun_position_from_datetime(lon_rad, lat_rad, times5060)

    # calculate differences
    diff_az = (sunAz*r2d + 180.0) - pos_pv['azimuth']
    diff_elev = sunAlt*r2d - pos_pv['apparent_elevation']
    index = range(len(sunDist))

    # plot
    index = df_xl.index

    fig, ax = plt.subplots(3, figsize=(15, 8), sharex=True)
    se_style = dict(linestyle='none', marker='o',
                    markerfacecolor='none', markeredgecolor='g', markersize=5)
    pv_style = dict(linestyle='none', marker='.',
                    markerfacecolor='r', markeredgecolor='r', markersize=3)
    ax[0].set_ylabel('Azimuth [$\degree$]')
    ax[0].plot(index, pos_pv['azimuth'], ',r', label='PVLIB basic', **pv_style)
    ax[0].plot(index, sunAz*r2d+180.0, label='SE', **se_style)

    ax[1].set_ylabel('Altitude/apparent elevation [$\degree$]')
    ax[1].plot(index, pos_pv['apparent_elevation'], **pv_style)
    ax[1].plot(index, sunAlt*r2d, **se_style)

    # ax[2].plot(times2020, pos1['equation_of_time'])
    ax[2].set_ylabel('Difference [$\degree$]')
    ax[2].plot(index, diff_az, label='diff azimuth', **pv_style)
    ax[2].plot(index, diff_elev, label='diff alt', **se_style)

    ax[0].legend()
    ax[2].legend()
    ax[2].set_ylim(-1, 1)
    ax[2].set_xlabel('Time')
    plt.suptitle('Validation of NEN5060 positions for SE.sun_position_from_datetime vs. PVLIB.get_solarposition (apparent_elevation)')
    plt.show()


if __name__ == "__main__":
    test_positions()
