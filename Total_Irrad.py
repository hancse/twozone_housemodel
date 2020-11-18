# def irrad_any(rground,iday,LST,t):
"""
(scalar) gamma = azimuth angle of the surface,
 east:gamma = -90, west:gamma = 90
 south:gamma = 0, north:gamma = 180
(scalar) beta = inclination angle of the surface,
 horizontal: beta=0, vertical: beta=90
"""

import numpy as np  # removed tabs
from qsun import qsun
# import pandas as pd
# import read_NEN
from NEN5060 import nen5060_to_dataframe

# 1. reading of the NEN5060 and assignment to local variables
# assignment to local variables is not really necessary

NUM = nen5060_to_dataframe(xl_tab_name="nen5060 - energie")

# Convert data frame to array
# to_array=NUM.to_numpy()

# assignment to local numpy arrays directly from dataframe columns
dom = NUM.loc[:, 'DAY(datum)'].values  # day of month
hod = NUM.loc[:, 'HOUR(uur)'].values  # hour of day
qglob_hor = NUM.loc[:, 'globale_zonnestraling'].values  # global irradiation
qdiff_hor = NUM.loc[:, 'diffuse_zonnestraling'].values  # diffuse irradiation
qdir_hor = NUM.loc[:, 'directe_zonnestraling'].values  # direct irradiation
qdir_nor = NUM.loc[:, 'directe_normale_zonnestraling'].values  # DNI
Toutdoor = NUM.loc[:, 'temperatuur'].values / 10.0  # temperature
phioutdoor = NUM.loc[:, 'relatieve_vochtigheid'].values  # %RH
xoutdoor = NUM.loc[:, 'absolute_vochtigheid'].values / 10.0  # AH

rain = NUM.loc[:, 'neerslaghoeveelheid'].values / 10.0  # precipitation
vwind = NUM.loc[:, 'windsnelheid'].values / 10.0  # wind speed
dirwind = NUM.loc[:, 'windrichting'].values  # wind dir
cloud = NUM.loc[:, 'bewolkingsgraad'].values / 8.0  # cloud coverage
sunduration = NUM.loc[:, 'zonneschijnduur'].values / 10.0  # daily solar duration
pdamp = NUM.loc[:, 'dampspanning'].values  # vapour pressure

# 2. prepare 1D time arrays and 3 D array E for results from qsun

# prepare time arrays
# t = (np.array(list(range(1,8761)))-1)*3600
t = (np.array(list(range(0, 8760)))) * 3600  # hourly grid with one year timespan expressed in seconds
# changed to more readable expression
# t2 = NUM.index.values

iday = 1 + np.floor(t / (24 * 3600))  # day of the year from t array: qsun assumes year starts with day 1
LST = np.floor((t / 3600) % 24)  # local time in hour : from 0 to 23:00

# 3. run qsun nine times in a loop

# Define an empty matrix for the result of qsun
# this result is a numpy stack with
# 8760 rows (hours per year)
# 9 columns (compass directions -90(E), -45(SE), 0(S), 45(SW), 90(W), 135(NW), 180(N), 225 (NE) plus horizontal)
# 4 decks (diffuse, direct, global and total solar irradiation)
E = np.zeros((8760, 9, 4))

# ground albedo is ignored, hence the  input parameter for qsun is zero
ground_albedo = 0

"""
# (scalar) beta is the tilt (inclination) angle of the surface, horizontal is 0, vertical is +90
# beta is 90 for k = -1 to 7 (vertical walls of house) and 0 for the horizontal flat roof surface
# (scalar) gamma is the azimuth of the surface, starting with -90(E), loop over k = -1 to 7 to 225(NE)
# for k = 8 (horizontal surface) gamma is arbitrary (set to 90 degrees here), since beta = 0
"""



k = -1  # k starts at -1 because of "East comes First" convention
for j in range(9):
    print(j)
    if k < 7:
        gamma = 45 * (k - 1)  # gamma -90 (E), -45 (SE), 0 (S), 45 (SW), 90 (W), 135 (NW), 180 (N), 225 (NE)
        beta = 90
    else:
        gamma = 90
        beta = 0
    k = k + 1

    for i in range(8760):
        temp = qsun(qdiff_hor[i], qdir_nor[i], gamma, beta, ground_albedo, iday[i], LST[i])
        # E[i][j]=qsun(t[i],qdiff_hor[i],qdir_nor[i],gamma,beta,rground)
        E[:, j][i] = temp
    # n=n+1
# myarray = np.asarray(E)
myarray = E

# is it necessary to have an assignment from temp to E and then to myarray?

# 4. split 3d array in 9 2D arrays: actually, only column 2 (total_irradiation) is used

# add time and Total radiation together
# split up the 3D array in 9 2 arrays with two rows and 8760 columns, shape: (2, 8760)
# one 2D array for each wind direction and for the roof

qsunE = np.vstack((t, myarray[:, 0, 2]))
qsunSE = np.vstack((t, myarray[:, 1, 2]))
qsunS = np.vstack((t, myarray[:, 2, 2]))
qsunSW = np.vstack((t, myarray[:, 3, 2]))
qsunW = np.vstack((t, myarray[:, 4, 2]))
qsunNW = np.vstack((t, myarray[:, 5, 2]))
qsunN = np.vstack((t, myarray[:, 6, 2]))
qsunNE = np.vstack((t, myarray[:, 7, 2]))
qsunhor = np.vstack((t, myarray[:, 8, 2]))

# alternative: use hstack to create nine 2D arrays with 8760 rows and 2 columns
# does not work well, because it yields a 1D array with shape (2*8760, )
# qsunE = np.hstack((t, myarray[:,0,2]))
# qsunSE = np.hstack((t, myarray[:,1,2]))
# qsunS = np.hstack((t, myarray[:,2,2]))
# qsunSW = np.hstack((t, myarray[:,3,2]))
# qsunW = np.hstack((t, myarray[:,4,2]))
# qsunNW = np.hstack((t, myarray[:,5,2]))
# qsunN = np.hstack((t, myarray[:,6,2]))
# qsunNE = np.hstack((t, myarray[:,7,2]))
# qsunhor = np.hstack((t, myarray[:,8,2]))

# even better alternative: create a pandas Dataframe with 9 named datacolumns,
# and an optional time column, which would be identical to the dataframe index [ 0...8759]

# df = pd.DataFrame()
