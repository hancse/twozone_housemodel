
import context

from housemodel.weather_solar.weatherdata import (read_nen_weather_from_xl,
                                                  NENdatehour2datetime,
                                                  get_hourly_knmi_weather_from_api)
import pandas as pd
from pytz import timezone
import numpy as np

import solarenergy as se
from pvlib.solarposition import get_solarposition, nrel_earthsun_distance
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")


def test_5060_timestamps():
    # read NEN5060 data from spreadsheet NEN5060-2018.xlsx "as is" into pandas DataFrame
    # use first header line as column names
    df_xl = read_nen_weather_from_xl()

    # generate and insert timezone-aware UTC and local timestamps (with DST)
    df_xl = NENdatehour2datetime(df_xl)

    # collect data from KNMI and compare with NEN5060
    df_api, exp = get_hourly_knmi_weather_from_api(260, 'ALL',
                                                   start=2001010101,
                                                   end=2001010124)

    np.testing.assert_allclose(df_xl.loc[0:23, 'temperatuur'], df_api.loc[0:23, 'T'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[0:23, 'luchtdruk'], df_api.loc[0:23, 'P'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[0:23, 'relatieve_vochtigheid'], df_api.loc[0:23, 'U'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[0:23, 'globale_zonnestraling'], df_api.loc[0:23, 'Q'], atol=0.5)

    df_api, exp = get_hourly_knmi_weather_from_api(260, 'ALL',
                                                   start=2004032801,
                                                   end=2004032824)

    np.testing.assert_allclose(df_xl.loc[2064:2064+23, 'temperatuur'], df_api.loc[0:23, 'T'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[2064:2064+23, 'luchtdruk'], df_api.loc[0:23, 'P'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[2064:2064+23, 'relatieve_vochtigheid'], df_api.loc[0:23, 'U'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[2064:2064+23, 'globale_zonnestraling'], df_api.loc[0:23, 'Q'], atol=0.5)

    df_api, exp = get_hourly_knmi_weather_from_api(260, 'ALL',
                                                   start=2010103101,
                                                   end=2010103124)

    # np.testing.assert_allclose(df_xl.loc[7272:7272+23, 'temperatuur'], df_api.loc[0:23, 'T'], rtol=1e-3)
    # np.testing.assert_allclose(df_xl.loc[7272:7272+23, 'luchtdruk'], df_api.loc[0:23, 'P'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[7272:7272+23, 'relatieve_vochtigheid'], df_api.loc[0:23, 'U'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[7272:7272+23, 'globale_zonnestraling'], df_api.loc[0:23, 'Q'], atol=0.5)

    df_api, exp = get_hourly_knmi_weather_from_api(260, 'ALL',
                                                   start=2003123101,
                                                   end=2003123124)

    np.testing.assert_allclose(df_xl.loc[8736:8736+23, 'temperatuur'], df_api.loc[0:23, 'T'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[8736:8736+23, 'luchtdruk'], df_api.loc[0:23, 'P'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[8736:8736+233, 'relatieve_vochtigheid'], df_api.loc[0:23, 'U'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[8736:8736+23, 'globale_zonnestraling'], df_api.loc[0:23, 'Q'], atol=0.5)


def test_5060_positions():
    # read NEN5060 data from spreadsheet NEN5060-2018.xlsx "as is" into pandas DataFrame
    # use first header line as column names
    df_xl = read_nen_weather_from_xl()
    # generate and insert timezone-aware UTC and local timestamps (with DST)
    df_xl = NENdatehour2datetime(df_xl)

    times5060 = df_xl['local_time']
    # Location of my solar panels:
    lon_deg = 5.0
    lat_deg = 52.0
    lon_rad = 5.0 * se.d2r  # Geographic longitude (>0 for eastern hemisphere; ° -> rad)
    lat_rad = 52.0 * se.d2r  # Geographic latitude  (>0 for northern hemisphere; ° -> rad)

    # VALIDATION of solar position calculations in PVLIB and solarenergy
    pos_pv = get_solarposition(times5060, lat_deg, lon_deg, method='nrel_numpy')

    # solarenergy
    #times2020_dt = times5060.to_pydatetime()
    sunAz, sunAlt, sunDist = se.sun_position_from_datetime(lon_rad, lat_rad, times5060)

    # calculate differences
    diff_az = (sunAz*se.r2d + 180.0) - pos_pv['azimuth']
    diff_elev = sunAlt*se.r2d - pos_pv['apparent_elevation']

    # plot
    fig, ax = plt.subplots(3, figsize=(15, 8), sharex=True)
    se_style = dict(linestyle='none', marker='o',
                    markerfacecolor='none', markeredgecolor='g', markersize=5)
    pv_style = dict(linestyle='none', marker='.',
                    markerfacecolor='r', markeredgecolor='r', markersize=3)
    ax[0].set_ylabel('Azimuth [$\degree$]')
    ax[0].plot(times5060, pos_pv['azimuth'], ',r', label='PVLIB basic', **pv_style)
    ax[0].plot(times5060, sunAz*se.r2d+180.0, label='SE', **se_style)

    ax[1].set_ylabel('Altitude/apparent elevation [$\degree$]')
    ax[1].plot(times5060, pos_pv['apparent_elevation'], **pv_style)
    ax[1].plot(times5060, sunAlt*se.r2d, **se_style)

    # ax[2].plot(times2020, pos1['equation_of_time'])
    ax[2].set_ylabel('Difference [$\degree$]')
    ax[2].plot(times5060, diff_az, label='diff azimuth', **pv_style)
    ax[2].plot(times5060, diff_elev, label='diff alt', **se_style)

    ax[0].legend()
    ax[2].legend()
    ax[2].set_ylim(-1, 1)
    ax[2].set_xlabel('Time')
    plt.suptitle('Validation of NEN5060 positions for SE.sun_position_from_datetime vs. PVLIB.get_solarposition (apparent_elevation)')
    plt.show()


def test_5060_distance():
    # read NEN5060 data from spreadsheet NEN5060-2018.xlsx "as is" into pandas DataFrame
    # use first header line as column names
    df_xl = read_nen_weather_from_xl()
    # generate and insert timezone-aware UTC and local timestamps (with DST)
    df_xl = NENdatehour2datetime(df_xl)

    times5060 = df_xl['local_time']
    # Location of my solar panels:
    lon_deg = 5.0
    lat_deg = 52.0
    lon_rad = 5.0 * se.d2r  # Geographic longitude (>0 for eastern hemisphere; ° -> rad)
    lat_rad = 52.0 * se.d2r  # Geographic latitude  (>0 for northern hemisphere; ° -> rad)

    # VALIDATION of solar position calculations in PVLIB and solarenergy
    dist_pv = nrel_earthsun_distance(times5060)

    # solarenergy
    #times2020_dt = times5060.to_pydatetime()
    sunAz, sunAlt, sunDist = se.sun_position_from_datetime(lon_rad, lat_rad, times5060)

    # calculate ratio and difference
    ratio_dist = sunDist / dist_pv.values
    diff_dist = sunDist - dist_pv.values

    index = range(len(sunDist))
    # plot
    fig, ax = plt.subplots(3, figsize=(15, 8), sharex=True)
    se_style = dict(linestyle='none', marker='o',
                    markerfacecolor='none', markeredgecolor='g', markersize=5)
    pv_style = dict(linestyle='none', marker='.',
                    markerfacecolor='r', markeredgecolor='r', markersize=3)
    ax[0].set_ylabel('Distance [A.U.]')
    ax[0].plot(index, dist_pv.values, ',r', label='PVLIB basic', **pv_style)
    ax[0].plot(index, sunDist, label='SE', **se_style)

    ax[1].set_ylabel('Ratio SE/PVLIB')
    ax[1].plot(index, ratio_dist, **pv_style)

    ax[2].set_ylabel('Difference SE - PVLIB')
    ax[2].plot(index, diff_dist, **pv_style)

    ax[0].legend()
    ax[2].legend()
    ax[1].set_ylim(0.999, 1.001)
    ax[2].set_ylim(-0.001, 0.001)
    ax[2].set_xlabel('Time')
    plt.suptitle('Validation of NEN5060 AU distance for SE.sun_position_from_datetime vs. PVLIB.get_solarposition (apparent_elevation)')
    plt.show()

def test_5060_airmass():
    # read NEN5060 data from spreadsheet NEN5060-2018.xlsx "as is" into pandas DataFrame
    # use first header line as column names
    df_xl = read_nen_weather_from_xl()
    # generate and insert timezone-aware UTC and local timestamps (with DST)
    df_xl = NENdatehour2datetime(df_xl)

    times5060 = df_xl['local_time']
    # Location of my solar panels:
    lon_deg = 5.0
    lat_deg = 52.0
    lon_rad = 5.0 * se.d2r  # Geographic longitude (>0 for eastern hemisphere; ° -> rad)
    lat_rad = 52.0 * se.d2r  # Geographic latitude  (>0 for northern hemisphere; ° -> rad)




if __name__ == "__main__":
     test_5060_timestamps()
     test_5060_positions()
     test_5060_distance()
