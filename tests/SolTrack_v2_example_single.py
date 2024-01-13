"""Example Python script to compute the position of the Sun and its rise and set times for a single instant
and demonstrate some other features."""

from soltrack import SolTrack
import datetime as dt
import pytz as tz

# Set the geographic location to Arnhem, the Netherlands:
geo_lon =  5.950270  # Positive -> east of Greenwich (degrees)
geo_lat = 51.987380  # Positive -> northern hemisphere (degrees)

st = SolTrack(geo_lon, geo_lat, use_degrees=True)  # Same as above, using default values for all but use_degrees.

# Set SolTrack date and time using separate (UTC!) year, month, day, hour, minute and second variables:
st.set_date_and_time(2023, 7, 16,  6, 2, 49.217348)  # Date: 2023-07-16, time: 06:02:49.217348 UTC

# Alternatively, use a (localised) datetime object:
cet     = tz.timezone('Europe/Amsterdam')
my_date = dt.datetime(2023, 7, 16,  8, 2, 49, 217348)  # Same as above, in local time for TZ=+2 (08:02:49.217348 LT)
my_date = cet.localize(my_date)
st.set_date_time(my_date)  # Set SolTrack date and time using a Python datetime object.

# Compute the Sun's position:
st.compute_position()

# Compute the rise and set times of the Sun:
st.compute_rise_set()


# Write results to standard output:
print("Location:   %0.3lf E, %0.3lf N"  % (st.geo_longitude, st.geo_latitude))
# print("Date/time:  %4d-%02d-%02d %02d:%02d:%09.6lf" % (st.year, st.month, st.day,  st.hour, st.minute, st.second))
print("Date/time:  %s"                  % my_date)
print("JD:         %0.11lf"             % (st.julian_day))
print()

print("Ecliptic longitude, latitude:        %10.6lf° %10.6lf°"     % (st.longitude, 0.0))  # Note: latitude is always 0 in this model
print("Distance:                            %10.6lf°"              % (st.distance))
print("Right ascension, declination:        %10.6lf° %10.6lf°"     % (st._right_ascension_uncorr, st._declination_uncorr))
print("Uncorrected altitude:                            %10.6lf°"  % (st._altitude_uncorr))
print("Corrected azimuth, altitude:         %10.6lf° %10.6lf°"     % (st.azimuth, st.altitude))
print("Corrected hour angle, declination:   %10.6lf° %10.6lf°"     % (st.hour_angle, st.declination))
print()

print("Rise time:      %s,    azimuth:   %11.5lf" % (*st.rise_time,     *st.rise_azimuth))
print("Transit time:   %s,    altitude:  %11.5lf" % (*st.transit_time,  *st.transit_altitude))
print("Set time:       %s,    azimuth:   %11.5lf" % (*st.set_time,      *st.set_azimuth))
print()


# Change the location whilst keeping the same SolTrack object:
st.set_location(geo_lon, -geo_lat)

# Compute the current position of the Sun for the new location:
st.now()
st.compute_position()

print("Location:   %0.3lf E, %0.3lf N"                         % (st.geo_longitude, st.geo_latitude))
print("Date (UT):  %4d-%02d-%02d"                              % (st.year, st.month, st.day))
print("Time (UT):  %02d:%02d:%09.6lf"                          % (st.hour, st.minute, st.second))
print("Corrected azimuth, altitude:         %10.6lf° %10.6lf°" % (st.azimuth, st.altitude))