import numpy as np
import pandas as pd

# import solarenergy as se
from solarenergy import *
from pvlib.location import Location
from pvlib.solarposition import get_solarposition
from pvlib.atmosphere import get_relative_airmass, alt2pres, get_absolute_airmass
from pvlib.clearsky import lookup_linke_turbidity, ineichen, bird

from pvlib.irradiance import get_extra_radiation

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")

# VALIDATION of solar intensity calculations in PVLIB and solarenergy
# PVLIB
loc = Location(52.0, 5.0, tz='Europe/Amsterdam', altitude=10, name='Nieuwegein')  # PVLIB Location object in degrees
times2020 = pd.date_range(start='2019-01-01 00:00:00', end='2019-12-31 23:00:00',
                          freq='H', tz='Europe/Amsterdam')  # range of Pandas Timestamps
solpos = get_solarposition(times2020, loc.latitude, loc.longitude, loc.altitude,
                           method='nrel_numpy')  # basic PVLIB function
apparent_zenith = solpos['apparent_zenith']
am_pv = get_relative_airmass(apparent_zenith, model='young1994')
cs = bird(apparent_zenith, am_pv, ozone=0.34, precipitable_water=1.42, aod500=0.2661, aod380=0.3538)
linke_turbidity = lookup_linke_turbidity(times2020, loc.latitude, loc.longitude)

dni_extra = get_extra_radiation(times2020, solar_constant=1361.5,
                                epoch_year=2020, method='asce')

# an input is a pandas Series, so solis is a DataFrame

# solarenergy
# Location of my solar panels:
geoLon = 5.0 * d2r  # Geographic longitude (>0 for eastern hemisphere; ° -> rad)
geoLat = 52.0 * d2r  # Geographic latitude  (>0 for northern hemisphere; ° -> rad)
times2020_dt = times2020.to_pydatetime()
# Compute Sun position (uses SolTrack behind the scenes):
sunAz, sunAlt, sunDist = sun_position_from_datetime(geoLon, geoLat, times2020_dt)
am_se = airmass(sunAlt)  # Air mass for this Sun altitude
extFac = extinction_factor(am_se)  # Extinction factor at sea level for this air mass
I_ext = sol_const / sunDist ** 2  # Extraterrestrial radiation = Solar constant, scaled with distance
DNI_cs = I_ext / extFac  # DNI for a clear sky

# plot
fig1, ax = plt.subplots(3, figsize=(15, 8), sharex='all')
ax[0].set_ylabel('Airmass')
ax[0].plot(times2020, am_pv, '.-r', label='PVLIB')
ax[0].plot(times2020, am_se, '.g', label='SE')

ax[1].set_ylabel('DNI [$W/m^2$]')
ax[1].plot(times2020, cs['dni'], '.r')
ax[1].plot(times2020, DNI_cs, '.g')

ax[2].plot(times2020, dni_extra, '-r')
ax[2].plot(times2020, I_ext, '--g')
ax[2].set_ylabel('Extra')

ax[0].legend()
# ax[2].legend()
# ax[2].set_ylim(-1, 1)
# ax[2].set_xlabel('Time')
plt.suptitle('Validation of Airmass and DNI SE vs. PVLIB')
plt.show()

plt.figure()
ax = cs.plot()
ax.set_ylabel('Irradiance $W/m^2$')
ax.set_title('Ineichen Clear Sky Model')
ax.legend(loc=2)

plt.figure()
cs.plot()
plt.ylabel('Irradiance $W/m^2$')
plt.title('Ineichen, climatological turbidity')
plt.show()

fig, ax = plt.subplots()
ax.plot(cs.index, cs['ghi'])
ax.plot(cs.index, cs['dni'])
ax.plot(cs.index, cs['dhi'])
ax.set_ylabel('Irradiance $W/m^2$')
plt.title('Ineichen, climatological turbidity')
plt.show()

# calculate differences
diff_az = (sunAz * r2d + 180.0) - solpos['azimuth']
diff_elev = sunAlt * r2d - solpos['apparent_elevation']

# plot
fig2, ax = plt.subplots(3, figsize=(15, 8), sharex='all')
ax[0].set_ylabel('Azimuth [$\degree$]')
ax[0].plot(times2020, solpos['azimuth'], '--r', label='PVLIB basic')
ax[0].plot(times2020, sunAz * r2d + 180.0, '-g', label='SE')

ax[1].set_ylabel('Altitude/elevation [$\degree$]')
ax[1].plot(times2020, solpos['apparent_elevation'], '--r')
ax[1].plot(times2020, sunAlt * r2d, '-g')

# ax[2].plot(times2020, pos1['equation_of_time'])
ax[2].set_ylabel('Difference')
ax[2].plot(times2020_dt, diff_az, '-g', label='diff azimuth')
ax[2].plot(times2020_dt, diff_elev, '--m', label='diff alt')

ax[0].legend()
ax[2].legend()
ax[2].set_ylim(-1, 1)
ax[2].set_xlabel('Time')
plt.suptitle('Validation of SE.sun_position_from_datetime vs. PVLIB.get_solarposition')
plt.show()
