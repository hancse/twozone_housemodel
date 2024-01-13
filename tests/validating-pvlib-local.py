#!usr/bin/bash

import os
from pathlib import Path

# for dirname, _, filenames in os.walk('NEN_data'):
# for filename in filenames:
# print(os.path.join(dirname, filename))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

import itertools
import pvlib
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
from pvlib.iotools import read_tmy3

import datetime as dt
import pytz as tz

from housemodel.sourcesink.qsun import qsun
from housemodel.sourcesink.solar_check import solar_check
from housemodel.tools.make_timeindex import make_timeindex

from solarenergy import *

import matplotlib

matplotlib.use("Qt5Agg")

"""
Use numpy.ndarray.item() to get scalars from single-element arrays
This fixes a DeprecationWarning about (implicit) array-to-scalar
conversion.
"""

# print(Path.cwd())
# data_dir = Path.cwd() / 'NEN_data'
data_dir = Path(__file__).parent.parent / "NEN_data"
# output_dir = Path.cwd()/'working'/'submit'
NENdata_path = data_dir / 'NEN5060-2018.xlsx'
print(NENdata_path)
xls = pd.ExcelFile(NENdata_path)
print(xls.sheet_names)  # Check sheet names

rground = 0  # ground reflection is ignored
# select sheet k=1 by NEN default
k = 1
if k == 1:
    # NUM = pd.read_excel(xls, 'nen5060 - energie')  # this file is part of NEN 5060 20018
    NUM = pd.read_excel(xls, 'nen5060 - energie', header=[0, 1])  # this file is part of NEN 5060 20018
    ind = NUM.index
    print(ind.values)
    print(NUM.head())
    print(NUM.columns)
elif k == 2:
    NUM = pd.read_excel(xls, 'ontwerp 1%')
elif k == 3:
    NUM = pd.read_excel(xls, 'ontwerp 5%')

# _________Convert data frame to array_____________
to_array = NUM.to_numpy()
# to_array = np.delete(to_array, 0, 0)

# ___________Read data_________________________
t = (np.array(list(range(1, 8761))) - 1) * 3600
# dom = to_array[:, 2]  # day of month
# hod = to_array[:, 3]  # hour of day
# qglob_hor = to_array[:, 4]
# qdiff_hor = to_array[:, 5]
# qdir_hor = to_array[:, 6]
# qdir_nor = to_array[:, 7]
# Toutdoor = to_array[:, 8] / 10
# phioutdoor = to_array[:, 9]
# xoutdoor = to_array[:, 10] / 10

# pdamp = to_array[:, 11]
# vwind = to_array[:, 12] / 10  # % at 10 m height
# dirwind = to_array[:, 13]
# cloud = to_array[:, 14] / 10
# rain = to_array[:, 15] / 10

dom = NUM.loc[:, 'DAY(datum)'].values
hod = NUM.loc[:, 'HOUR(uur)'].values
qglob_hor = NUM.loc[:, 'globale_zonnestraling'].values
qdiff_hor = NUM.loc[:, 'diffuse_zonnestraling'].values
qdir_hor = NUM.loc[:, 'directe_zonnestraling'].values
qdir_nor = NUM.loc[:, 'directe_normale_zonnestraling'].values
Toutdoor = NUM.loc[:, 'temperatuur'].values / 10.0
phioutdoor = NUM.loc[:, 'relatieve_vochtigheid'].values
xoutdoor = NUM.loc[:, 'absolute_vochtigheid'].values / 10.0

rain = NUM.loc[:, 'neerslaghoeveelheid'].values / 10.0
vwind = NUM.loc[:, 'windsnelheid'].values / 10.0
dirwind = NUM.loc[:, 'windrichting'].values
cloud = NUM.loc[:, 'bewolkingsgraad'].values / 8.0
sunduration = NUM.loc[:, 'zonneschijnduur'].values / 10.0
pdamp = NUM.loc[:, 'dampspanning'].values

# An hour past noon local time on 1 March 2020:
# myTZ = 'Europe/Amsterdam'
# year = to_array[:, 0]
# month = to_array[:, 1]
# day = to_array[:, 2]
# hour = to_array[:, 3]
# hour = hour - 1
# minute = 0
# second = 0

year = NUM.loc[:, 'jaar'].values
month = NUM.loc[:, 'MONTH(datum)'].values
day = NUM.loc[:, 'DAY(datum)'].values
hour = NUM.loc[:, 'HOUR(uur)'].values
hour = hour - 1  # running from 0-23
minute = 0
second = 0
# _______________________

myTZ = tz.timezone('Europe/Amsterdam')  # My timezone
myTime = dt.datetime(int(year[0]), int(month[0]), int(day[0]), int(hour[0]), int(minute),
                     int(second))  # Time w/o timezone
lt = myTZ.localize(myTime, is_dst=None)  # Mark as local time
utc = lt.astimezone(tz.utc)  # Convert to UTC
# pause()
"""
def make_timeindex(i):
    # temp = pd.DataFrame({"times": range(1)})
    temp1 = pd.DataFrame(columns=['times'], index=pd.to_datetime([]))

    myTZ = tz.timezone('Europe/Amsterdam')  # My timezone
    myTime = dt.datetime(int(year[i]), int(month[i]), int(day[i]), int(hour[i]), int(minute),
                         int(second))  # Time w/o timezone
    lt = myTZ.localize(myTime, is_dst=None)  # Mark as local time
    utc = lt.astimezone(tz.utc)  # Convert to UTC

    temp = pd.to_datetime(utc)
    temp1.loc[i] = temp

    df = pd.to_datetime(temp1['times'])
    time = pd.DatetimeIndex(df)

    return time
"""

# ======== PVlib calculation of EOT =================================================

PVlib_equation_of_time = np.zeros(365)
PVlib_declination = np.zeros(365)
PVlib_dni_extra = np.zeros(365)
# i: day of the year, i=i+1 correct for year start at day 1
j = 0
for i in range(365):
    # Equation of time from Duffie & Beckman and attributed to Spencer (1971) and Iqbal (1983).
    PVlib_equation_of_time[i] = pvlib.solarposition.equation_of_time_spencer71(i + 1)
    # Solar declination from Duffie & Beckman and attributed to Spencer (1971) and Iqbal (1983).
    PVlib_declination[i] = pvlib.solarposition.declination_spencer71(i + 1)
    # Determine extraterrestrial radiation from day of year.
    PVlib_dni_extra[i] = pvlib.irradiance.get_extra_radiation(i + 1,
                                                              solar_constant=1366.1,  # 1361.5 W/m^2
                                                              method='spencer')

# =========== QSUN calculation of EOT ===================================

t = (np.array(list(range(1, 8761))) - 1) * 3600
t2 = np.linspace(0, 8759 * 3600, 8760, dtype=int)  #
gamma = 180
beta = 90
Temp_Val = np.zeros((8760, 6))
ET_val = np.zeros(8760)
decl_val = np.zeros(8760)
ha_val = np.zeros(8760)
za_val = np.zeros(8760)
theta_val = np.zeros(8760)
Eon_val = np.zeros(8760)

for i in range(len(t)):
    Temp_Val[i] = solar_check(t[i])
    ET_val[i], decl_val[i], ha_val[i], dum1, dum2, Eon_val = solar_check(t2[i])

# __________
split = np.hsplit(Temp_Val[:, 0], 365)  # ET: equation of time
temp = np.ones(365)

# for i in range(len(split) - 1):
for i in range(len(split)):
    mask = np.isnan(split[i])
    a = split[i][~mask]
    unique = np.unique(a)
    # print(i)
    temp[i] = unique
# Equation_time = np.delete(temp, -1, 0)
Equation_time = temp

# ========= Plot Equation of time (EOT): compare PVlib and QSUN =================

plt.figure(figsize=(15, 5))
plt.plot(Equation_time, label="matlab_conv")
plt.plot(PVlib_equation_of_time, label="PVlib")
plt.plot(PVlib_equation_of_time - Equation_time, label='difference')
plt.legend(loc="upper left")
plt.suptitle("Equation of time", fontsize=20)
plt.show()
# input("Press enter")

# __________
split = np.hsplit(Temp_Val[:, 1], 365)  # decl: declination of the sun
temp = np.ones(365)

# for i in range(len(split) - 1):
for i in range(len(split)):
    mask = np.isnan(split[i])
    a = split[i][~mask]
    unique = np.unique(a)
    # print(i)
    temp[i] = unique
# decl = np.delete(temp, -1, 0)
decl = temp

# _____Plot EQ of time  compare to PV lib
plt.figure(figsize=(15, 5))
plt.plot(decl, label="matlab_conv")
plt.plot(PVlib_declination, label="PVlib")
plt.plot(PVlib_declination - decl, label='difference')
plt.legend(loc="upper left")
plt.suptitle("Declination", fontsize=20)
plt.show()

# __________
split = np.hsplit(Temp_Val[:, 5], 365)  # Eon: extraterrestial radiation
temp = np.ones(365)

# for i in range(len(split) - 1):
for i in range(len(split)):
    mask = np.isnan(split[i])
    a = split[i][~mask]
    unique = np.unique(a)
    # print(i)
    temp[i] = unique
# extra_i = np.delete(temp, -1, 0)
extra_i = temp

# _____Plot EQ of time  compare to PV lib
plt.figure(figsize=(15, 5))
plt.plot(extra_i, label="qsun (converted from matlab)")
plt.plot(PVlib_dni_extra, label="PVlib")
plt.plot((1366.1 - (PVlib_dni_extra - extra_i)), label="difference (%)")
plt.legend(loc="upper left")
plt.suptitle("Extraterrestial", fontsize=20)
plt.show()
# input("Press enter")

'''
Calculate hour angle in degree.
time: pd datetime index 
longitude: in degree
equation_of_time: in minutes
'''

time = make_timeindex(year, month, day, hour, offset=1)

longitude = 5.1  # in degree
hourangle = np.ones(8760)  # in degree , time must be localized to the timezone for the
j = 0
# n=0
k = 0
for i in range(8760):
    i = i + 1
    if k < 2066 or (
            2066 < k < 7274) or k > 7274:  # This 2 values of NEN5060 date time give error of none-existing
        if i == (24 * j):  # group time by 24 hour to create j : day of the year
            j = j + 1
            # n=n+1

        time = make_timeindex(year, month, day, hour)
        ha = pvlib.solarposition.hour_angle(time, longitude, PVlib_equation_of_time[j])
        hourangle[k] = ha  # in degree
    k = k + 1

PVlib_hourangle_rad = hourangle * np.pi / 180  # Co

hour_angle = np.delete(Temp_Val[:, 2], -1, 0)

# _____Plot EQ of time  compare to PV lib

plt.figure(figsize=(20, 5))
plt.plot(hour_angle[0:500], label="matlab_conv")
plt.plot(PVlib_hourangle_rad[0:500], label="PVlib")
plt.legend(loc="upper left")
plt.show()

'''
Check hour angle calculation with 7.1_Solar_Radiation_on_Inclined paper.

'''
LST = 9
L = 52.1
# LON = Local Longitude [graden] oost is positief
LON = 5.1
LSM = 15  # https://www.pveducation.org/pvcdrom/properties-of-sunlight/solar-time
et = 14.97
r = math.pi / 180
L = L * r
beta = beta * r
theta = 2 * math.pi * (1 - 1) / 365.25
el = 4.901 + 0.033 * math.sin(-0.031 + theta) + theta  # elevation
print('elavation', el)
delta = math.asin(math.sin(23.442 * r) * math.sin(el))
q1 = math.tan(4.901 + theta)
q2 = math.cos(23.442 * r) * math.tan(el)
AST = LST + et / 60 + (4 / 60) * (LSM - LON)  # change minus to plus
print('AST', AST)
h = (AST - 12) * 15
print('hour angle', h)
hai = math.cos(L) * math.cos(delta) * math.cos(h) + math.sin(L) * math.sin(delta)
salt = math.asin(hai)
print('salt', salt)
phi = math.acos((hai * math.sin(L) - math.sin(delta)) / (math.cos(salt) * math.cos(L))) * np.sign(h)
gam = phi - gamma * r
# cai=cos(teta)
cai = math.cos(salt) * math.cos(abs(gam)) * math.sin(beta) + hai * math.cos(beta)
# teta = incident angle on the tilted surface
teta = math.acos(cai)
# salts=solar altitude for an inclined surface

salts = math.pi / 2 - teta
print('salts', salts)

'''
Check hour angle calculation with 7.1_Solar_Radiation_on_Inclined paper.
'''
latitude = 52.1  # *np.pi/180
# LON = Local Longitude [graden] oost is positief
longitude = 5.1  # *np.pi/180

Eq_time = 14.97  # euqation of time
testtime = pd.date_range(start='2018-01-01 09:00:00', end='2018-01-01 09:00:00', freq='h')
print(testtime)
temp = pvlib.solarposition.hour_angle(testtime, 5.1, Eq_time)
print('hour_angle', temp)

# __________________check hour of angle calculation_______________
'''
copy code from PVlib and paste to the section
'''

NS_PER_HR = 1.e9 * 3600.  # nanoseconds per hour
naive_times = testtime.tz_localize(None)  # naive but still localized
# hours - timezone = (times - normalized_times) - (naive_times - times)
hrs_minus_tzs = 1 / NS_PER_HR * (
        2 * testtime.astype(np.int64) - testtime.normalize().astype(np.int64) - naive_times.astype(np.int64))
# ensure array return instead of a version-dependent pandas <T>Index

'The equation is slightly different.'

h_a = np.asarray(15. * (hrs_minus_tzs - 12.) + longitude + Eq_time / 4.)
print('hour_angle', h_a)

# _____________ AST____________
df = pvlib.solarposition.ephemeris(testtime, latitude, longitude, pressure=101325, temperature=12)
print('AST - solar time', df)  # SolarTime (AST) = (180 + HrAngle) / 15.

# Calculate Solar zenith angle using PVlib

# The latitude input is in Rad
# hour angle input in rad
# The declination in rad
# The return Solar Zenith and solar Azimuth in Rad

latitude = 52.1 * np.pi / 180  # convert to rads.
longitude = 5.1 * np.pi / 180

solar_zenith = np.ones(8760)  # Solar zenith angle in radians.
solar_azimuth = np.ones(8760)
# hourangle_rad = hourangle*np.pi/180
# declination_rad = declination*np.pi/180
j = 0
n = 0
for i in range(8760):
    i = i + 1
    if i == (24 * j):  # group time by 24 hour to create day of the year
        j = j + 1
        # print(n)
    temp = pvlib.solarposition.solar_zenith_analytical(latitude, PVlib_hourangle_rad[n], PVlib_declination[j])
    temp1 = pvlib.solarposition.solar_azimuth_analytical(latitude, PVlib_hourangle_rad[n], PVlib_declination[j],
                                                         temp)  # in rad

    solar_zenith[n] = temp  # solar zenith in rad
    solar_azimuth[n] = temp1  # Solar azimuth angle in radians
    n = n + 1

solar_zenith_z = (solar_zenith * 180) / np.pi
solar_azimuth_a = (solar_azimuth * 180) / np.pi
# _____
plt.figure(figsize=(20, 5))
plt.plot(solar_zenith_z[0:24], label="solar_zenith")
plt.legend(loc="upper left")
plt.show()

# %%

# _____Plot Solar Azimuth
plt.figure(figsize=(20, 5))
plt.plot(solar_azimuth_a[0:24], label="solar_azimuth")
plt.legend(loc="upper left")
plt.show()

# %%

solar_azimuth_a = solar_azimuth * 180 / np.pi
plt.plot(solar_azimuth_a[0:24])  # check with https://www.esrl.noaa.gov/gmd/grad/antuv/SolarCalc.jsp

# %%

solar_zenith_z = solar_zenith * 180 / np.pi

airmass = pvlib.atmosphere.get_relative_airmass(solar_zenith_z, model='kastenyoung1989')  # solar_zenith in degree
plt.figure(figsize=(20, 5))
plt.plot(airmass[0:500], label="airmass")
plt.legend(loc="upper left")
plt.show()

qglob_hor = to_array[:, 4]
qdiff_hor = to_array[:, 5]
qdir_hor = to_array[:, 6]
qdir_nor = to_array[:, 7]

plt.figure(figsize=(20, 5))
plt.plot(qdir_nor, label="direct normal irradiation")
plt.legend(loc="upper left")
plt.suptitle('Direct normal irradiance', fontsize=20)
plt.show()

'''
gamma = azimuth angle of the surface,

Surface azimuth angles in decimal degrees. surface_azimuth must
be >=0 and <=360. The Azimuth convention is defined as degrees
east of north (e.g. North = 0, South=180 East = 90, West = 270).

beta = inclination angle of the surface,
horizontal: beta=0, vertical: beta=90

Surface tilt angles in decimal degrees. The tilt angle is
defined as degrees from horizontal (e.g. surface facing up = 0,
surface facing horizon = 90)
'''

Total_in_plane_irradiance = np.ones(8760)
Total_in_plane_beam_irradiance = np.ones(8760)
Total_in_plane_diffuse_irradiance = np.ones(8760)
Inplane_diffuse_irradiance_from_sky = np.ones(8760)
In_plane_diffuse_irradiance_from_ground = np.ones(8760)

dni = qdir_nor
ghi = qglob_hor
dhi = qdiff_hor

solar_zenith_z = (solar_zenith * 180) / np.pi  # Convert rad in to degree
solar_azimuth_a = (solar_azimuth * 180) / np.pi  # Convert rad in to degree

j = 1
n = 0
k = 0

gamma = 0
beta = 90

for i in range(8760):
    i = i + 1
    if i == (24 * j):
        j = j + 1
        k = k + 1
    if k == 365:
        break
        # print(n)
    # gamma =gamma*np.pi/180
    # beta  =beta*np.pi/180
    '''Sky diffuse models include:
        * isotropic (default)
        * klucher
        * haydavies
        * reindl
        * king
        * perez'''
    temp = pvlib.irradiance.get_total_irradiance(beta, gamma,
                                                 solar_zenith_z[n], solar_azimuth_a[n],
                                                 dni[n], ghi[n], dhi[n], dni_extra=PVlib_dni_extra[k], airmass=None,
                                                 albedo=.2, surface_type=None,
                                                 model='haydavies',
                                                 model_perez='allsitescomposite1990')

    Total_in_plane_irradiance[n] = temp['poa_global']
    Total_in_plane_beam_irradiance[n] = temp['poa_direct']
    Total_in_plane_diffuse_irradiance[n] = temp['poa_diffuse']
    Inplane_diffuse_irradiance_from_sky[n] = temp['poa_sky_diffuse']
    In_plane_diffuse_irradiance_from_ground[n] = temp['poa_ground_diffuse']
    n = n + 1

'''
(scalar) gamma = azimuth angle of the surface,
    east:gamma = -90, west:gamma = 90
    south:gamma = 0, north:gamma = 180
    (scalar) beta = inclination angle of the surface,
    horizontal: beta=0, vertical: beta=90
'''

rground = 0  # ground reflection is ignored
t = (np.array(list(range(1, 8761))) - 1) * 3600

gamma = 180  # gamma 180 in qsun North =  gamma 0 in PVlib
beta = 90

# w, h = 9, 8760;
# E=[[0 for x in range(w)] for y in range(h)]
E = np.zeros((8760, 4))
for i in range(8759):
    E[i] = qsun(t[i], qdiff_hor[i], qdir_nor[i], gamma, beta, rground)
myarray = np.asarray(E)


def plot_irradiance(fig_x, fig_y, qsun_data, qsun_label, pvlib_data, pvlib_label, title):
    plt.figure(figsize=(fig_x, fig_y))
    plt.plot(qsun_data, label=qsun_label)
    plt.plot(pvlib_data, label=pvlib_label)
    plt.legend(loc='upper left')
    plt.suptitle(title, fontsize=20)
    plt.show()


# Compare Total irradiance on an inclined surface
plot_irradiance(20, 5, Total_in_plane_irradiance[0:8759], "PVlib_Total_ir",
                myarray[:, 2][0:8759], "Ir",
                'Total irradiance North')

# Compare diffuse irradiance on an inclined surface
plot_irradiance(20, 5, Total_in_plane_diffuse_irradiance[0:8000], "PVlib_Diffuse_ir",
                myarray[:, 0][0:8000], "diffuseIr",
                'Diffuse irradiance North')

# Compare beam irradiance on an inclined surface
plot_irradiance(20, 5, myarray[:, 1][0:8000], "directIr",
                Total_in_plane_beam_irradiance[0:8000], "PVlib_Direct_ir",
                'Beam irradiance North')

# Calculate with different Azimuth
'''
gamma = azimuth angle of the surface,

Surface azimuth angles in decimal degrees. surface_azimuth must
be >=0 and <=360. The Azimuth convention is defined as degrees
east of north (e.g. North = 0, South=180 East = 90, West = 270).

beta = inclination angle of the surface,
horizontal: beta=0, vertical: beta=90

Surface tilt angles in decimal degrees. The tilt angle is
defined as degrees from horizontal (e.g. surface facing up = 0,
surface facing horizon = 90)
'''
solar_zenith_z = (solar_zenith * 180) / np.pi  # Convert rad in to degree
solar_azimuth_a = (solar_azimuth * 180) / np.pi  # Convert rad in to degree

Q = np.zeros((9, 8760, 4))
g = 0

for j in range(9):

    j = j + 1

    if j < 9:

        gamma = 45 * (j - 1)  # N, NE, E, SE , S , SW , W, NW (j = 1,2,3,4,5,6,7)
        beta = 90

    else:

        gamma = 90
        beta = 0

    Total_in_plane_irradiance = np.ones(8760)
    Total_in_plane_beam_irradiance = np.ones(8760)
    Total_in_plane_diffuse_irradiance = np.ones(8760)
    Inplane_diffuse_irradiance_from_sky = np.ones(8760)
    In_plane_diffuse_irradiance_from_ground = np.ones(8760)

    k = 1
    e = 0
    n = 0

    for i in range(8760):

        i = i + 1

        if i == (24 * k):
            k = k + 1
            e = e + 1

        if e == 365:
            continue

        temp = pvlib.irradiance.get_total_irradiance(beta, gamma,
                                                     solar_zenith_z[n], solar_azimuth_a[n],
                                                     dni[n], ghi[n], dhi[n], dni_extra=PVlib_dni_extra[e], airmass=None,
                                                     albedo=.2, surface_type=None,
                                                     model='haydavies',
                                                     model_perez='allsitescomposite1990')

        Total_in_plane_irradiance[n] = temp['poa_global']
        Total_in_plane_beam_irradiance[n] = temp['poa_direct']
        Total_in_plane_diffuse_irradiance[n] = temp['poa_diffuse']
        Inplane_diffuse_irradiance_from_sky[n] = temp['poa_sky_diffuse']
        In_plane_diffuse_irradiance_from_ground[n] = temp['poa_ground_diffuse']
        n = n + 1
    temp1 = np.array([Total_in_plane_irradiance, Total_in_plane_beam_irradiance, Total_in_plane_diffuse_irradiance,
                      In_plane_diffuse_irradiance_from_ground])
    temp1 = np.transpose(temp1)
    Q[g][:, ][:] = temp1
    g = g + 1

# %%

'''
(scalar) gamma = azimuth angle of the surface,
    east:gamma = -90, west:gamma = 90
    south:gamma = 0, north:gamma = 180
    (scalar) beta = inclination angle of the surface,
    horizontal: beta=0, vertical: beta=90
'''

rground = 0  # ground reflection is ignored

t = (np.array(list(range(1, 8761))) - 1) * 3600
E = np.zeros((8760, 9, 4))

n = 0
k = -1

for j in range(9):
    # j=j+1
    if k < 6:
        gamma = 45 * (k - 1)  # gamma -90 (E), -45 (SE), 0 (S), 45 (SW), 90 (W), 135 (NW), 180 (N), 225 (NE)
        beta = 90
    else:
        gamma = 90
        beta = 0
    k = k + 1

    for i in range(8760):
        # print(i)
        # print(j)
        # E[i][j]=qsun(t[i],qdiff_hor[i],qdir_nor[i],gamma,beta,rground)
        E[:, n][i] = qsun(t[i], qdiff_hor[i], qdir_nor[i], gamma, beta, rground)
    n = n + 1
myarray = np.asarray(E)

# North surface Azimuth
plot_irradiance(20, 5, Q[0][:, 0], "PVlib_Total_ir",
                myarray[:, 6, 2], "qsun_ir",
                "Total irradiance North")

plot_irradiance(20, 5, myarray[:, 6, 1], "qsun_ir",
                Q[0][:, 1], "PVlib_beam_ir",
                "Total beam irradiance North")

# NE surface Azimuth
plot_irradiance(20, 5, Q[1][:, 0], "PVlib_Total_ir",
                myarray[:, 7, 2], "qsun_ir",
                "Total irradiance NE")

plot_irradiance(20, 5, myarray[:, 7, 1], "direct_ir",
                Q[1][:, 1], "PVlib_direct_ir",
                "Beam irradiance NE")

# E surface Azimuth
plot_irradiance(20, 5, Q[2][:, 0], "PVlib_Total_ir",
                myarray[:, 0, 2], "qsun_ir",
                "Total irradiance East")

plot_irradiance(20, 5, Q[2][:, 1], "PVlib_Beam_ir",
                myarray[:, 0, 1], "qsun_ir",
                "Beam irradiance East")

# Azimuth SE
plot_irradiance(20, 5, Q[3][:, 0], "PVlib_Total_ir",
                myarray[:, 1, 2], "qsun_ir",
                "Total irradiance SE")

plot_irradiance(20, 5, Q[3][:, 1], "PVlib_Beam_ir",
                myarray[:, 1, 1], "qsun_ir",
                "Beam irradiance SE")

# Azimuth S
plot_irradiance(20, 5, Q[4][:, 0], "PVlib_Total_ir",
                myarray[:, 2, 2], "qsun_ir",
                "Total irradiance South")

plot_irradiance(20, 5, Q[4][:, 1], "PVlib_Beam_ir",
                myarray[:, 2, 1], "qsun_ir",
                "Beam irradiance South")

# Azimuth SW
plot_irradiance(20, 5, Q[5][:, 0], "PVlib_Total_ir",
                myarray[:, 3, 2], "qsun_ir",
                "Total irradiance SW")

plot_irradiance(20, 5, Q[5][:, 1], "PVlib_Beam_ir",
                myarray[:, 3, 1], "qsun_ir",
                "Beam irradiance SW")

# Azimuth W
plot_irradiance(20, 5, Q[6][:, 0], "PVlib_Total_ir",
                myarray[:, 4, 2], "qsun_ir",
                "Total irradiance West")

plot_irradiance(20, 5, myarray[:, 4, 1], "qsun_ir",
                Q[6][:, 1], "PVlib_Beam_ir",
                "Beam irradiance West")

# Azimuth NW
plot_irradiance(20, 5, Q[7][:, 0], "PVlib_Total_ir",
                myarray[:, 5, 2], "qsun_ir",
                "Total irradiance NW")

plot_irradiance(20, 5, myarray[:, 5, 1], "qsun_ir",
                Q[7][:, 1], "PVlib_Beam_ir",
                "Beam irradiance NW")
