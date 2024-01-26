
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

# import context

from solarenergy import *
from pvlib.solarposition import get_solarposition, nrel_earthsun_distance
from pvlib.atmosphere import get_relative_airmass, alt2pres, get_absolute_airmass
from pvlib.irradiance import get_extra_radiation
from astrotool import doy_from_datetime

from housemodel.weather_solar.weatherdata import (read_nen_weather_from_xl,
                                                  NENdatehour2datetime)

matplotlib.use("Qt5Agg")


def atest_5060_yield():
    # read NEN5060 data from spreadsheet NEN5060-2018.xlsx "as is" into pandas DataFrame
    # use first header line as column names
    df_xl = read_nen_weather_from_xl()
    # generate and insert timezone-aware UTC and local timestamps (with DST)
    df_xl = NENdatehour2datetime(df_xl)
    print(df_xl)

    times5060 = df_xl['local_time']
    doy = times5060.dt.day_of_year

    # Location of solar panels:
    lon_deg = 5.0
    lat_deg = 52.0
    lon_rad = 5.0 * d2r   # Geographic longitude (>0 for eastern hemisphere; ° -> rad)
    lat_rad = 52.0 * d2r  # Geographic latitude  (>0 for northern hemisphere; ° -> rad)

    # orientation of solar panels
    spAz = -2.0 * d2r  # Azimuth ('wind direction') of my panels are facing.
    # Note: South=0, W=90° (pi/2 rad) in the northern hemisphere!  (rad)
    spIncl = 28.0 * d2r  # Inclination of my panels w.r.t. the horizontal  (rad)

    # PVLIB
    pos_pv = get_solarposition(times5060, lat_deg, lon_deg, method='nrel_numpy')
    true_zenith = pos_pv['zenith']
    am_pv = get_relative_airmass(true_zenith, model='young1994')
    # I_extra_pv = get_extra_radiation(times5060.values, solar_constant=1361.5, epoch_year=2020, method='asce')

    # solarenergy
    sunAz, sunAlt, sunDist = sun_position_from_datetime(lon_rad, lat_rad, times5060)
    am_se = airmass(sunAlt)                 # Air mass for this Sun altitude
    extFac = extinction_factor(am_se)       # Extinction factor at sea level for this air mass
    I_extra_se = sol_const / sunDist ** 2   # Extraterrestrial radiation = Solar constant, scaled with distance
    DNI_se = I_extra_se / extFac               # DNI for a clear sky

    ghi = df_xl['globale_zonnestraling']  # global horizontal irradiance (pyranometer)
    dhi = df_xl['diffuse_zonnestraling']  # diffuse horizontal irradiance (calculated)
    bhi = df_xl['directe_zonnestraling']  # beam horizontal irradiance (calculated)
    bni = df_xl['directe_normale_zonnestraling'] # beam normal irradiance aka. "DNI" (calculated)

    cosTheta = cos_angle_sun_panels(spAz, spIncl, sunAz, sunAlt)  # cos of the angle with which Sun hits my panels
    theta = np.arccos(cosTheta)

    # direct irradiance on inclined surface
    I_direct = bni * cosTheta
    # diffuse radiation on inclined surface
    I_diffuse = diffuse_radiation_projection_perez87(doy, sunAlt, spIncl,
                                                        theta, bni, dhi)
    albedo = 0.2
    I_reflected = ghi * albedo

    I_total = I_direct + I_diffuse + I_reflected

    index = df_xl.index

    se_style = dict(linestyle='none', marker='o',
                    markerfacecolor='none', markeredgecolor='g', markersize=5)
    pv_style = dict(linestyle='none', marker='.',
                    markerfacecolor='r', markeredgecolor='r', markersize=3)

    fig, ax = plt.subplots(2, figsize=(15, 8), sharex=True)
    ax[0].plot(index, I_direct, label='direct', **pv_style)
    ax[0].plot(index, I_diffuse, label='diffuse', **se_style)
    ax[0].plot(index, I_reflected, label='reflected')
    ax[0].plot(index, I_total, label='total')
    ax[0].set_ylabel('Irradiance [$W/m^2$]')

    ax[1].plot(index, bni, label='BNI (NEN5060)')
    ax[1].set_ylabel('Beam Normal Irradiance [$W/m^2$]')
    plt.suptitle('Validation of Extraterrestrial radiation and Airmass \n '
                 'SE vs. PVLIB')
    plt.show()


if __name__ == "__main__":
    atest_5060_yield()
