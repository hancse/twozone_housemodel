"""Example Python script using the SolarEnergy module for a range of instances."""

import numpy as np
from solarenergy import (d2r, r2d, airmass,
                         sun_position_from_datetime,
                         sun_position_from_date_and_time,
                         extinction_factor,
                         cos_angle_sun_panels)
import pandas as pd

# Location of my solar panels:

geoLon = 5 * d2r  # Geographic longitude (>0 for eastern hemisphere; ° -> rad)
geoLat = 52 * d2r  # Geographic latitude  (>0 for northern hemisphere; ° -> rad)

# Orientation of my solar panels:
spAz = -2 * d2r  # Azimuth ('wind direction') of my panels are facing.
# Note: South=0, W=90° (pi/2 rad) in the northern hemisphere!  (rad)
spIncl = 28 * d2r  # Inclination of my panels w.r.t. the horizontal  (rad)

dates = pd.date_range('2022-03-21', pd.to_datetime('2022-03-22'), freq='1h',
                      tz='Europe/Amsterdam')  # DatetimeIndex 0-24h
df = pd.DataFrame(data=dates, columns=['dtm'])  # Create a Pandas DataFrame with the datetimes as first column

# Compute Sun positions (using SolTrack behind the scenes) and add them as three columns to the df:

df['sunAz'], df['sunAlt'], df['sunDist'] = sun_position_from_datetime(geoLon, geoLat, df['dtm'])

df['I_ext'] = 1361.5 / df.sunDist ** 2  # Extraterrestrial radiation (at the top of the atmosphere; AM0)

df['AM'] = airmass(df.sunAlt)  # Air mass for this Sun altitude
df['extFac'] = extinction_factor(df.AM)  # Extinction factor at sea level for this airmass
df['DNI'] = df.I_ext / df.extFac  # DNI for a clear sky

df['cosTheta'] = cos_angle_sun_panels(spAz, spIncl, df.sunAz,
                                      df.sunAlt)  # cos of the angle with which Sun hits my panels
df['dirRad'] = df.DNI * df.cosTheta  # Insolation of direct sunlight on my panels

df.sunAz *= r2d  # Convert azimuth and ...
df.sunAlt *= r2d  # ... altitude to degrees for printing

print(df[df.sunAlt > 0])  # Print the results for the hours when the Sun is above the horizon

# Code example for a single calculation
# Note that in most cases, the vector option is preferred
# (see the code example above, and see Performance for details).
# The code example below is given for completeness.

"""Example Python script using the SolarEnergy module for a single instance."""

# Location of my solar panels:

geoLon = 5 * d2r  # Geographic longitude (>0 for eastern hemisphere; ° -> rad)
geoLat = 52 * d2r  # Geographic latitude  (>0 for northern hemisphere; ° -> rad)

# Orientation of my solar panels:
spAz = -2 * d2r  # Azimuth ('wind direction') of my panels are facing.
# Note: South=0, W=90° (pi/2 rad) in the northern hemisphere!  (rad)
spIncl = 28 * d2r  # Inclination of my panels w.r.t. the horizontal  (rad)

# An hour past noon local time on 1 March 2020:
myTZ = 'Europe/Amsterdam'
year = 2020
month = 3
day = 1
hour = 13

# Compute Sun position (uses SolTrack behind the scenes):
sunAz, sunAlt, sunDist = sun_position_from_date_and_time(geoLon, geoLat, year, month, day, hour, timezone=myTZ)

I_ext = 1361.5 / sunDist ** 2  # Extraterrestrial radiation (at the top of the atmosphere; AM0)

AM = airmass(sunAlt)  # Air mass for this Sun altitude
extFac = extinction_factor(AM)  # Extinction factor at sea level for this airmass
cosTheta = cos_angle_sun_panels(spAz, spIncl, sunAz, sunAlt)  # cos of the angle with which Sun hits my panels

DNI = I_ext / extFac  # DNI for a clear sky
dirRad = DNI * cosTheta  # Insolation of direct sunlight on my panels

# Print input and output:

print("Location:           %0.3lf E, %0.3lf N" % (geoLon * r2d, geoLat * r2d))
print("Date:               %4d-%2.2d-%2.2d" % (year, month, day))
print("Time:               %2d:00" % hour)
print()
print("Sun azimuth:        %7.3lf°" % (sunAz * r2d))
print("Sun altitude:       %7.3lf°" % (sunAlt * r2d))
print("Sun distance:       %7.4lf AU" % sunDist)
print()
print("I_ext:              %7.1lf W/m²" % I_ext)
print()
print("Air mass:           %7.3lf" % AM)
print("Extinction factor:  %7.3lf" % extFac)
print("DNI:                %7.1lf W/m²" % DNI)
print()
print("Sun-panels angle:   %7.1lf°" % (np.arccos(cosTheta) * r2d))
print("Direct insolation:  %7.1lf W/m²" % dirRad)
