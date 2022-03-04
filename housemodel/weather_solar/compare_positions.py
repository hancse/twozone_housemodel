import numpy as np
import pandas as pd

import solarenergy as se

from pvlib.location import Location
from pvlib.solarposition import get_solarposition
from pvlib.atmosphere import get_relative_airmass, alt2pres, get_absolute_airmass
from pvlib.clearsky import lookup_linke_turbidity, ineichen
from pvlib.irradiance import get_extra_radiation

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")

# VALIDATION of solar position calculations in PVLIB and solarenergy

loc = Location(52.0, 5.0, tz='Europe/Amsterdam', altitude=10, name='Nieuwegein')  # PVLIB Location object in degrees

times2020 = pd.date_range(start='2019-01-01 00:00:00', end='2019-12-31 23:00:00',
                          freq='H', tz='Europe/Amsterdam')                        # range of Pandas Timestamps
pos1 = loc.get_solarposition(times2020)                                           # built-in function of Location object
pos2 = get_solarposition(times2020, loc.latitude, loc.longitude, loc.altitude,
                         method='nrel_numpy')                                     # basic PVLIB function

# calculate differences
diff_az = pos1['azimuth'] - pos2['azimuth']
diff_elev = pos1['apparent_elevation'] - pos2['apparent_elevation']

# assert built-in and basic are equal
pd.testing.assert_series_equal(pos1['azimuth'], pos2['azimuth'])

# plot
fig, ax = plt.subplots(3, figsize=(15, 8), sharex=True)
ax[0].set_ylabel('Azimuth [$\degree$]')
ax[0].plot(times2020, pos1['azimuth'], '-b', label='PVLIB loc')
ax[0].plot(times2020, pos2['azimuth'], '--r', label='PVLIB basic')

ax[1].set_ylabel('Altitude/elevation [$\degree$]')
ax[1].plot(times2020, pos1['apparent_elevation'], '-b')
ax[1].plot(times2020, pos2['apparent_elevation'], '--r')

ax[2].set_ylabel('Difference')
ax[2].plot(times2020, diff_az, '-g')
ax[2].plot(times2020, diff_elev, '--m')

ax[0].legend()
ax[2].set_ylim(-0.001, 0.001)
ax[2].set_xlabel('Time')
plt.suptitle('Validation of PVLIB Location.get_solarposition vs. get_solarposition')
plt.show()

# CONCLUSION: loc.get_solarposition and get_solarposition(method='nrel_numpy or 'nrel_numba') are identical

# solarenergy
# Location of my solar panels:
geoLon = 5.0 * se.d2r  # Geographic longitude (>0 for eastern hemisphere; ° -> rad)
geoLat = 52.0 * se.d2r  # Geographic latitude  (>0 for northern hemisphere; ° -> rad)

times2020_dt = times2020.to_pydatetime()
sunAz, sunAlt, sunDist = se.sun_position_from_datetime(geoLon, geoLat, times2020_dt)

# calculate differences
diff_az1 = (sunAz*se.r2d + 180.0) - pos2['azimuth']
diff_elev1 = sunAlt*se.r2d - pos2['apparent_elevation']

# plot
fig, ax = plt.subplots(3, figsize=(15, 8), sharex=True)
ax[0].set_ylabel('Azimuth [$\degree$]')
ax[0].plot(times2020, pos2['azimuth'], '--r', label='PVLIB basic')
ax[0].plot(times2020, sunAz*se.r2d+180.0, '-g', label='SE')

ax[1].set_ylabel('Altitude/elevation [$\degree$]')
ax[1].plot(times2020, pos2['apparent_elevation'], '--r')
ax[1].plot(times2020, sunAlt*se.r2d, '-g')

# ax[2].plot(times2020, pos1['equation_of_time'])
ax[2].set_ylabel('Difference')
ax[2].plot(times2020_dt, diff_az1, '.g', label='diff azimuth')
ax[2].plot(times2020_dt, diff_elev1, '.m', label='diff alt')

ax[0].legend()
ax[2].legend()
ax[2].set_ylim(-1, 1)
ax[2].set_xlabel('Time')
plt.suptitle('Validation of SE.sun_position_from_datetime vs. PVLIB.get_solarposition (apparent_elevation)')
plt.show()

# calculate differences
diff_az2 = (sunAz*se.r2d + 180.0) - pos2['azimuth']
diff_elev2 = sunAlt*se.r2d - pos2['elevation']

# plot
fig, ax = plt.subplots(3, figsize=(15, 8), sharex=True)
ax[0].set_ylabel('Azimuth [$\degree$]')
ax[0].plot(times2020, pos2['azimuth'], '--r', label='PVLIB basic')
ax[0].plot(times2020, sunAz*se.r2d+180.0, '-g', label='SE')

ax[1].set_ylabel('Altitude/elevation [$\degree$]')
ax[1].plot(times2020, pos2['elevation'], '--r')
ax[1].plot(times2020, sunAlt*se.r2d, '-g')

ax[2].set_ylabel('Difference')
ax[2].plot(times2020_dt, diff_az2, '.g', label='diff azimuth')
ax[2].plot(times2020_dt, diff_elev2, '.m', label='diff alt')

ax[0].legend()
ax[2].legend()
ax[2].set_ylim(-1, 1)
ax[2].set_xlabel('Time')
plt.suptitle('Validation of SE.sun_position_from_datetime vs. PVLIB.get_solarposition (elevation)')
plt.show()









