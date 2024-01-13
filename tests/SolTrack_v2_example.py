# Create a Pandas DatetimeIndex range every 20 days 1hour and 10 minutes, in my timezone:
import pandas as pd

dti = pd.date_range('2022-01-01 01:00', '2022-12-31 23:00', freq='20d1h10min', tz='Europe/Amsterdam')

# Set the geographic location to Arnhem, the Netherlands (we'll use degrees in SolTrack):
geo_lon = 5.950270  # Positive -> east of Greenwich (degrees)
geo_lat = 51.987380  # Positive -> northern hemisphere (degrees)

# Create a SolTrack instance and specify preferences:
from soltrack import SolTrack

st = SolTrack(geo_lon, geo_lat, use_degrees=True)  # Use default values for all but use_degrees
st.set_date_time(dti)  # Pass my dates and times to SolTrack
st.compute_position()  # Compute the Sun's position
st.compute_rise_set()  # Compute the rise and set times of the Sun

# Print some selected results as arrays and create chaos:
if st.lt is not None:  # If local time was used
    print('Local time:     ', *st.lt)  # Asterisk (*) unpacks the DTI

print('UTC:            ', *st.utc)
print('azimuth:        ', *st.azimuth)
print('altitude:       ', *st.altitude)
print('distance:       ', *st.distance)
print('riseTime:       ', *st.rise_time)
print('transTime:      ', *st.transit_time)
print('setTime:        ', *st.set_time)

# Store selected results in a Pandas DataFrame and print that in a more orderly fashion:
st.create_df(utc=True, jd=True, ecl=True, rts_pos=True)
with pd.option_context('display.max_columns', None, 'display.width', None):  # Want all columns
    print(st.df)