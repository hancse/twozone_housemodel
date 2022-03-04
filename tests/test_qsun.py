
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import context

import logging
logging.basicConfig()
logger = logging.getLogger('test_qsun')
logger.setLevel(logging.INFO)

from housemodel.sourcesink.NEN5060 import nen5060_to_dataframe, run_qsun, run_qsun_new

def my_assert(condition, fail_str, suc_str):
    assert condition, fail_str
    print( suc_str )

df_nen = nen5060_to_dataframe()

Qsolar = np.zeros((8760, 1))  # 8760 rows x 2 cols
az = 0
tlt = 0
logging.info(f"Azimuth {az} and Tilt {tlt}")

df_irr_old = run_qsun(df_nen)

# north_is_zero (niz), check azimuth 0(N), 90(E), 180(S), 270(W)
# in qsun, azimuth = azimuth - 180: north = -180 internally (works!)
df_irr_north_niz= run_qsun_new(df_nen, 0, 90, north_is_zero=True)
Qsolar_north_niz = df_irr_north_niz.total_irr.values

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

np.testing.assert_allclose(Qsolar_north, df_irr_old['total_N'].values)
np.testing.assert_allclose(Qsolar_east, df_irr_old['total_E'].values)
np.testing.assert_allclose(Qsolar_south, df_irr_old['total_S'].values)
np.testing.assert_allclose(Qsolar_west, df_irr_old['total_W'].values)
np.testing.assert_allclose(Qsolar_hor, df_irr_old['total_hor'].values)

np.testing.assert_allclose(Qsolar_north, Qsolar_north_niz)
np.testing.assert_allclose(Qsolar_east, Qsolar_east_niz)
np.testing.assert_allclose(Qsolar_south, Qsolar_south_niz)
np.testing.assert_allclose(Qsolar_west, Qsolar_west_niz)
np.testing.assert_allclose(Qsolar_hor, Qsolar_hor_niz)

np.testing.assert_allclose(Qsolar_north, Qsolar_north_plus)

np.testing.assert_allclose(time_old, time_new)

# plot the results
# plt.figure(figsize=(15, 5))         # key-value pair: no spaces
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True,
                                    figsize=(15, 12))  # 3 rijen en 1 kolom. x-axis will be shared among all subplots.
ax1.plot(time_hr, Qsolar_north, label='North')
ax1.set_ylabel('Irradiation ($W/m^2$)')
ax1.set_ylim(-100, 1200)
ax1.legend()

ax2.plot(time_hr, Qsolar_east, label='East')
ax2.set_ylabel('Irradiation ($W/m^2$)')
ax2.set_ylim(-100, 1200)
ax2.legend()

ax3.plot(time_hr, Qsolar_south, label='South')
ax3.set_ylabel('Irradiation ($W/m^2$)')
ax3.set_ylim(-100, 1200)
ax3.legend()

ax4.plot(time_hr, Qsolar_west, label='West')
ax4.set_ylabel('Irradiation ($W/m^2$)')
ax4.set_ylim(-100, 1200)
ax4.legend()

ax5.plot(time_hr, Qsolar_hor, label='Horizontal')
ax5.set_ylabel('Irradiation ($W/m^2$)')
ax5.set_ylim(-100, 1200)
ax5.legend()
plt.show()