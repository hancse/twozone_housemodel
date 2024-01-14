#!/bin/env python3
# copied from MMIP-HP-configurator-PythonCode-git commit 3c83d1a

""" insolation.py:  code that deals with insolation (direct and diffuse radiation on a surface).
    2021-05-17, MvdS: initial version.
"""

import colored_traceback
colored_traceback.add_hook()

from solarenergy import *
# from getweather import get_api_weather
from housemodel.weather_solar.weatherdata import read_hourly_knmi_weather_from_csv

import numpy as np
import astrotool as at

import colored_traceback

colored_traceback.add_hook()


def insolation_on_surface_from_daterange(start_date, end_date, geo_lon, geo_lat, ori_az, ori_incl):
    """Compute the insolation on a surface with given orientation, geographic location and date/time range.
    
    Parameters:  
      start_date (datetime):  Start of the date/time range to compute the insolation for.
      end_date   (datetime):  End of the date/time range to compute the insolation for.
      geo_lon    (float):     Geographical longitude of the location of the insolated surface (rad; >0 = east of Greenwich).
      geo_lat    (float):     Geographical latitude of the location of the insolated surface (rad; >0 = northern hemisphere).
      ori_az     (float):     Azimuth of the orientation of the surface (rad; 0 = south, pi/2 = west, pi = north, -pi/2 = east).
      ori_incl   (float):     Inclination of the orientation of the surface (rad; 0 = horizontal, pi/2 = vertical).
    
    Returns:  
      (pandas.DataFrame):     Pandas dataframe containing weather data and insolation data for the specified date/time range.
    
    """

    # Get weather data from the KNMI API:
    # station_DeBilt = '260'
    # vars_from_station = 'SUNR'
    # df_insol = get_api_weather(station_DeBilt, vars_from_station, yr, mon, day, yr+1, mon, day)  # MvdS: this syntax should be replaced by the syntax below.
    # df_insol = get_api_weather(station_DeBilt, vars_from_station, start_date, end_date)

    # Read weather data from KNMI file:
    df_insol = read_hourly_knmi_weather_from_csv("uurgeg_260_2011-2020_Bilt.csv", start_date, end_date)

    # Compute the positions of the Sun for the dates and time in df_insol:
    myDatetime = df_insol['datetime'].values  # Get the datetimes to compute the Sun position for from the df
    df_insol['sunAz'], df_insol['sunAlt'], df_insol['sunDist'] = sun_position_from_datetime(geo_lon, geo_lat,
                                                                                            myDatetime)

    # Compute additional insolation data:
    df_insol['Iext'] = sol_const / np.square(df_insol['sunDist'])  # Extraterrestrial radiation [W/m^2]
    df_insol['AM'] = airmass(df_insol['sunAlt'])  # Air mass [-]
    df_insol['EF'] = extinction_factor(df_insol['AM'])  # Extinction factor [-]

    df_insol['DHR'], df_insol['BHR'], df_insol['DNI'] = \
        diffuse_radiation_from_global_radiation_and_sunshine(df_insol['Q'], df_insol['SQ'], df_insol['sunAlt'],
                                                             df_insol['Iext'])

    # Projection of direct sunlight on panel:
    df_insol['cosTheta'] = cos_angle_sun_panels(ori_az, ori_incl, df_insol['sunAz'], df_insol['sunAlt'])
    df_insol['panelDir'] = df_insol['DNI'] * df_insol['cosTheta']

    # Projection of diffuse daylight on panel:
    DoY = at.doy_from_datetime(myDatetime)  # Day of year
    df_insol['panelDif'] = diffuse_radiation_projection_perez87(DoY, df_insol['sunAlt'], ori_incl,
                                                                np.arccos(df_insol['cosTheta']), df_insol['DNI'],
                                                                df_insol['DHR'])

    # Total light on panel:
    df_insol['panelTot'] = df_insol['panelDir'] + df_insol['panelDif']

    # print(df_insol)
    return df_insol


if __name__ == "__main__":
    import test_weatherdata as test
    test.test_insolation_on_surface_from_daterange()
