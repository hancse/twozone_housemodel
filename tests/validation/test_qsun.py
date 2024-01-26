
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

# import context

import logging
logging.basicConfig()
logger = logging.getLogger('test_qsun')
logger.setLevel(logging.INFO)

from housemodel.sourcesink.NEN5060 import run_qsun, run_qsun_new
from housemodel.weather_solar.weatherdata import (read_nen_weather_from_xl,
                                                  NENdatehour2datetime)

matplotlib.use("Qt5Agg")


def my_assert(condition, fail_str, suc_str):
    assert condition, fail_str
    print(suc_str)


def test_qsun_old_vs_new():
    # read NEN5060 data from spreadsheet NEN5060-2018.xlsx "as is" into pandas DataFrame
    # use first header line as column names
    df_nen = read_nen_weather_from_xl()
    # generate and insert timezone-aware UTC and local timestamps (with DST)
    df_nen = NENdatehour2datetime(df_nen)

    Qsolar = np.zeros((8760, 1))  # 8760 rows x 2 cols
    az = 0
    tlt = 0
    logging.info(f"Azimuth {az} and Tilt {tlt}")

    df_irr_old = run_qsun(df_nen)

    # north_is_zero (niz), check azimuth 0(N), 90(E), 180(S), 270(W)
    # in qsun, azimuth = azimuth - 180: north = -180 internally (works!)
    df_irr_north_niz= run_qsun_new(df_nen, 0, 90, north_is_zero=True)
    Qsolar_north_niz = (df_irr_north_niz.total_irr).values

    df_irr_east_niz= run_qsun_new(df_nen, 90, 90, north_is_zero=True)
    Qsolar_east_niz = (df_irr_east_niz.total_irr).values

    df_irr_south_niz = run_qsun_new(df_nen, 180, 90, north_is_zero=True)
    Qsolar_south_niz = (df_irr_south_niz.total_irr).values

    df_irr_west_niz = run_qsun_new(df_nen, 270, 90, north_is_zero=True)
    Qsolar_west_niz = (df_irr_west_niz.total_irr).values

    df_irr_hor_niz = run_qsun_new(df_nen, 0, 0, north_is_zero=True)
    Qsolar_hor_niz = (df_irr_hor_niz.total_irr).values

    # south_is_zero, check azimuth -180(N), -90(E), 0(S), 90(W)
    df_irr_north= run_qsun_new(df_nen, -180, 90, north_is_zero=False)
    Qsolar_north = (df_irr_north.total_irr).values

    df_irr_east= run_qsun_new(df_nen, -90, 90, north_is_zero=False)
    Qsolar_east = (df_irr_east.total_irr).values

    df_irr_south = run_qsun_new(df_nen, 0, 90, north_is_zero=False)
    Qsolar_south = (df_irr_south.total_irr).values

    df_irr_west = run_qsun_new(df_nen, 90, 90, north_is_zero=False)
    Qsolar_west = (df_irr_west.total_irr).values

    df_irr_hor = run_qsun_new(df_nen, 0, 0, north_is_zero=False)
    Qsolar_hor = (df_irr_hor.total_irr).values

    df_irr_north_plus= run_qsun_new(df_nen, 180, 90, north_is_zero=False)
    Qsolar_north_plus = (df_irr_north_plus.total_irr).values

    time_old = df_irr_old.iloc[:, 0].values  # 8760 rows 1D
    time_new = df_irr_north.iloc[:, 0].values

    time_hr = time_new/3600

    assert_allclose(Qsolar_north, df_irr_old['total_N'].values)
    assert_allclose(Qsolar_north, df_irr_old['total_N'].values)
    assert_allclose(Qsolar_east, df_irr_old['total_E'].values)
    assert_allclose(Qsolar_south, df_irr_old['total_S'].values)
    assert_allclose(Qsolar_west, df_irr_old['total_W'].values)
    assert_allclose(Qsolar_hor, df_irr_old['total_hor'].values)

    assert_allclose(Qsolar_north, Qsolar_north_niz)
    assert_allclose(Qsolar_east, Qsolar_east_niz)
    assert_allclose(Qsolar_south, Qsolar_south_niz)
    assert_allclose(Qsolar_west, Qsolar_west_niz)
    assert_allclose(Qsolar_hor, Qsolar_hor_niz)

    assert_allclose(Qsolar_north, Qsolar_north_plus)

    assert_allclose(time_old, time_new)

    print(f"all assertions passed")

    old_style = dict(linestyle='none', marker='o',
                    markerfacecolor='none', markeredgecolor='g', markersize=5)
    new_style = dict(linestyle='none', marker='.',
                    markerfacecolor='r', markeredgecolor='r', markersize=3)
    # plot the results
    # plt.figure(figsize=(15, 5))         # key-value pair: no spaces
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True,
                                    figsize=(15, 12))  # 3 rijen en 1 kolom. x-axis will be shared among all subplots.
    ax1.plot(time_hr, Qsolar_north, label='North_new', **new_style)
    ax1.plot(time_hr, df_irr_old['total_N'].values, label='North_old', **old_style)
    ax1.set_ylabel('Irradiation ($W/m^2$)')
    ax1.set_ylim(-100, 1200)
    ax1.legend()

    ax2.plot(time_hr, Qsolar_east, label='East_new', **new_style)
    ax2.plot(time_hr, df_irr_old['total_E'].values, label='East_old', **old_style)
    ax2.set_ylabel('Irradiation ($W/m^2$)')
    ax2.set_ylim(-100, 1200)
    ax2.legend()

    ax3.plot(time_hr, Qsolar_south, label='South_new', **new_style)
    ax3.plot(time_hr, df_irr_old['total_S'].values, label='South_old', **old_style)
    ax3.set_ylabel('Irradiation ($W/m^2$)')
    ax3.set_ylim(-100, 1200)
    ax3.legend()

    ax4.plot(time_hr, Qsolar_west, label='West_new', **new_style)
    ax4.plot(time_hr, df_irr_old['total_W'].values, label='West_old', **old_style)
    ax4.set_ylabel('Irradiation ($W/m^2$)')
    ax4.set_ylim(-100, 1200)
    ax4.legend()

    ax5.plot(time_hr, Qsolar_hor, label='Horizontal_new', **new_style)
    ax5.plot(time_hr, df_irr_old['total_hor'].values, label='Horizontal_old', **old_style)
    ax5.set_ylabel('Irradiation ($W/m^2$)')
    ax5.set_ylim(-100, 1200)
    ax5.legend()

    ax1.set_title("Validation of run_qsun_new (arbitrary orientation) vs. run_qsun (8 fixed orientations)")
    plt.show()


if __name__ == "__main__":
    test_qsun_old_vs_new()
