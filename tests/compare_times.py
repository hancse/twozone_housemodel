import numpy as np
import pandas as pd

import solarenergy as se

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")

# VALIDATION of input time ranges in solarenergy


# solarenergy
# Location of my solar panels:
geoLon = 5.0 * se.d2r  # Geographic longitude (>0 for eastern hemisphere; ° -> rad)
geoLat = 52.0 * se.d2r  # Geographic latitude  (>0 for northern hemisphere; ° -> rad)

times2020 = pd.date_range(start='2019-01-01 00:00:00', end='2019-12-31 23:00:00',
                          freq='H', tz='Europe/Amsterdam')  # range of Pandas Timestamps

times2020_dt = times2020.to_pydatetime()  # range of Python datetimes
sunAz_1, sunAlt_1, sunDist_1 = se.sun_position_from_datetime(geoLon, geoLat, times2020)
sunAz_2, sunAlt_2, sunDist_2 = se.sun_position_from_datetime(geoLon, geoLat, times2020_dt)

# calculate differences
diff_az1 = sunAz_2 * se.r2d - sunAz_1 * se.r2d
diff_elev1 = sunAlt_2 * se.r2d - sunAlt_1 * se.r2d

# plot
fig, ax = plt.subplots(3, figsize=(15, 8), sharex=True)
ax[0].set_ylabel('Azimuth [$\degree$]')
ax[0].plot(times2020, sunAz_1 * se.r2d, '-r', label='Pandas Timestamp')
ax[0].plot(times2020, sunAz_2 * se.r2d, '-g', label='Python datetime')

ax[1].set_ylabel('Altitude/elevation [$\degree$]')
ax[1].plot(times2020, sunAlt_1 * se.r2d, '-r', label='Pandas Timestamp')
ax[1].plot(times2020, sunAlt_2 * se.r2d, '-g', label='Python datetime')

# ax[2].plot(times2020, pos1['equation_of_time'])
ax[2].set_ylabel('Difference')
ax[2].plot(times2020_dt, diff_az1, '--g', label='diff azimuth')
ax[2].plot(times2020_dt, diff_elev1, '.m', label='diff alt')

ax[0].legend()
ax[2].legend()
ax[2].set_ylim(-1, 1)
ax[2].set_xlabel('Time')
plt.suptitle(
    'Validation of SE.sun_position_from_datetime (pandas DateTimeIndex) vs. SE.sun_position_from_datetime (python datetimes) ')
plt.show()
