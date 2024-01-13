#!/bin/env python
# copied from MMIP-HP-configurator-PythonCode-git commit 3c83d1a

# from soltrack import Position, Location, Time, computeSunPosition
from solarenergy import *

from housemodel.weather_solar.weatherdata import read_hourly_knmi_weather_from_csv
import numpy as np
import pytz as tz
import datetime as dt
import astrotool as at
import colored_traceback
colored_traceback.add_hook()

import logging
logging.basicConfig()
logger = logging.getLogger('solar_dummy_calculation')
logger.setLevel(logging.INFO)

# Location:
# get_location(s)
latitude = 52.0*d2r  # >0 = N
longitude = 5.0*d2r  # >0 = E

# Orientation panel:
# get_orientation(s)
panelAz   =  0*d2r   # S=0
panelIncl = 35*d2r   # Horizontal = 0

# Date and time:
yr = 2019
mon = 6
day = 1
hr = 12

# Get weather data from KNMI API:
# station_DeBilt = '260'
# vars_from_station = 'SUNR'
# dfsolar = get_weather(station_DeBilt, vars_from_station, yr, mon, day, yr+1, mon, day)

# Read weather data from KNMI file:
nltz = tz.timezone('Europe/Amsterdam')
start_date = nltz.localize(dt.datetime(2019, 1, 1))
end_date   = nltz.localize(dt.datetime(2019, 12, 31))
dfsolar    = read_hourly_knmi_weather_from_csv("uurgeg_260_2011-2020_Bilt.csv", start_date, end_date)
logger.info(dfsolar)

# using year, month, day, hour and timezone
azimuth, altitude, distance = sun_position_from_date_and_time(longitude, latitude, yr, mon, day, hr, timezone="Europe/Amsterdam")
# print("%4i %2i %2i %2i               %11.5f %11.5f %11.7f" % (yr, mon, day, hr, azimuth*r2d, altitude*r2d, distance))
logger.info(" %4i %2i %2i %2i               %11.5f %11.5f %11.7f" % (yr, mon, day, hr, azimuth*r2d, altitude*r2d, distance))

# using timezone-naive datetime object
myDatetime = dt.datetime(yr, mon, day, hr, 0, 0)  # tz-naive
azimuth, altitude, distance = sun_position_from_datetime(longitude, latitude, myDatetime)
# print("%s         %11.5f %11.5f %11.7f " % (myDatetime, azimuth*r2d, altitude*r2d, distance))
logger.info(" %s         %11.5f %11.5f %11.7f " % (myDatetime, azimuth*r2d, altitude*r2d, distance))

# using timezone-aware datetime object created with localize
myDatetime = nltz.localize(dt.datetime(yr, mon, day, hr, 0, 0))  # tz-aware
azimuth, altitude, distance = sun_position_from_datetime(longitude, latitude, myDatetime)
# print("%s   %11.5f %11.5f %11.7f " % (myDatetime, azimuth*r2d, altitude*r2d, distance))
logger.info(" %s   %11.5f %11.5f %11.7f " % (myDatetime, azimuth*r2d, altitude*r2d, distance))

# print(type(dfsolar['datetime']))
myDatetime = dfsolar['datetime'].values
dfsolar['sunAz'], dfsolar['sunAlt'], dfsolar['sunDist'] = sun_position_from_datetime(longitude, latitude, myDatetime)

dfsolar['Iext'] = sol_const/np.square(dfsolar['sunDist'])    # Extraterrestrial radiation [W/m^2]
dfsolar['AM']   = airmass(dfsolar['sunAlt'])                # Air mass [-]
dfsolar['EF']   = extinction_factor(dfsolar['AM'])          # Extinction factor [-]

dfsolar['DHR'], dfsolar['BHR'], dfsolar['DNI'] = \
    diffuse_radiation_from_global_radiation_and_sunshine(dfsolar['Q'], dfsolar['SQ'], dfsolar['sunAlt'], dfsolar['Iext'])

# Projection of direct sunlight on panel:
dfsolar['cosTheta'] = cos_angle_sun_panels(panelAz, panelIncl, dfsolar['sunAz'], dfsolar['sunAlt'])
dfsolar['panelDir'] = dfsolar['DNI'] * dfsolar['cosTheta']

# Projection of diffuse daylight on panel:
DoY = at.doy_from_datetime(myDatetime)  # Day of year
dfsolar['panelDif'] = diffuse_radiation_projection_perez87(DoY, dfsolar['sunAlt'], panelIncl,
                                                              np.arccos(dfsolar['cosTheta']), dfsolar['DNI'], dfsolar['DHR'])
#Cloudiness
Profileswitch = 2
corr = 0.375        # Emperical correction for effect of cloudiness/tranparency on method 1 and 2
df_cloud    = read_hourly_knmi_weather_from_csv("uurgeg_260_2011-2020_Bilt.csv", start_date, end_date)

if Profileswitch == 1:
    # Compute cloudiness to reduce direct insolation:
    #    myDatetime = df_cloud['datetime'].values  # Get the datetimes to compute the Sun position for from the df
    cloudiness = df_cloud['N']/8*10/10  # cloudiness in octans converted to decimals and then percentages
    brightness = 1-cloudiness*corr
    # print(cloudiness)
    # print(brightness)
elif Profileswitch == 2: # Method of https://www.researchgate.net/publication/288669348_EMPIRICAL_MODEL_FOR_PV-BATTERY_INSOLATION_WITH_CLOUDINESS_ACCOUNTING
    Bmax = 8 # according to research it should be: 10 - cloudiness, but it gives high negative values
    Pcld = 6
    Al = 0.7
    Kmin = 0.1
    B = df_cloud['N']
    Kcld = 1-(1-Kmin)*(B/Bmax)*((B/Bmax)**(Pcld-1)+Al)*(1/(1+Al))
    print('Avr Kcld', sum(Kcld)/len(Kcld))
    brightness = Kcld / (1-corr)

elif Profileswitch == 3:
    # no cloudiness is taken into account
    brightness = 1

# Total light on panel:
dfsolar['insolTot'] = dfsolar['panelDir'] + dfsolar['panelDif']    
dfsolar['panelTot'] = dfsolar['panelDir']*brightness + dfsolar['panelDif']
Solareff = sum(dfsolar['panelTot'])/sum(dfsolar['insolTot'])*100
print('insolTot', sum(dfsolar['insolTot']))
print('panelTot', sum(dfsolar['panelTot']))
print('Solarefficiency', Solareff, '%')



# ===  computation Pv performance===
Tamb        = 20        # [°C] 
Vwind       = 1         # [m/s]
NOCT        = 44        # [°C]
Pmaxcorr    = -0.3      # [%/°C] powerloss by temperature
Apanel      = 1.7272    # [m²]
PpanelNom   = 375       # [ Wp @ STC] 
nPanels     = 10         # number of panels in system (asummed to be equally installed)

Effpanel    =   PpanelNom / Apanel / 1000       # Divided by 1.000 because the STC for determining WattPeaks use 1000 W/m² solar irradiation.
Effsystem   = 0.78 # According to ISSO 78 (2005)
mounting    = 1   #  Way the panels have been mounted 0 = in roof systeem, 0.5 = moderately ventilated, 1 = well ventilated  ===> CGI's moderation, no fundamental research

dfsolar['Sirradiance'] = dfsolar['panelTot'] / 10000 * 1000  # Divided by 10.000 and multiplied by 1000 to convert W/m² to mW/cm², used for NOCT.
dfsolar['Tamb'] = dfsolar['T'] 
dfsolar['Tcell'] = dfsolar['T']  + (NOCT-dfsolar['T'])/80 * dfsolar['Sirradiance']  # Computation cell temperature based on NOCT, given by manufacturer.



# Article: Wind Effect on PV Module Temperature: Analysis of Different Techniques for an Accurate Estimation by Schwingshackl, Petitta a.o.
# https://www.researchgate.net/publication/258779655_Wind_Effect_on_PV_Module_Temperature_Analysis_of_Different_Techniques_for_an_Accurate_Estimation
# Based on Skoplaki method 1 and 2 for windspeed correction on PV modules

hwNOCT1  = 8.91 + 2 * 1                                 # Based on Skoplaki 1 method, Skoplaki 2 would have 5.7 + 2.8 * Vwind (STC)
hwV1     = 8.91 + 2 * dfsolar['FF']/10                  # Vf, actual Vwind measured 10m above ground --> suitable for KNMI data
hwNOCT2  = 5.7 + 2.8 * 1                                # Based on Skoplaki 1 method, Skoplaki 2 would have 5.7 + 2.8 * Vwind (STC)
hwV2     = 5.7 + 2.8 * (0.68*(dfsolar['FF']/10)-0.5)    # Vw, actual Vwind (Vw = 0.68*Vf-0.5)
TauA     = 0.9                                          # Constant determined by Skoplaki (tau * alpha)

TcellSkoplaki1 = dfsolar['Tcell'] * (((hwNOCT1/hwV1) * 1-(Effpanel/TauA)*(1-Pmaxcorr/100*25)))  # 25°C for STC temperature
dfsolar['TcellSkoplaki1'] = np.where(TcellSkoplaki1 < dfsolar['Tamb'], dfsolar['Tamb'], TcellSkoplaki1)  # Limits the computated cell temperature lower than Tamb to Tamb. Sometimes Skoplaki has lower Tcell than Tamb.

TcellSkoplaki2 = dfsolar['Tcell'] * (((hwNOCT2/hwV2) * 1-(Effpanel/TauA)*(1-Pmaxcorr/100*25)))  # 25°C for STC temperature
TcellSkoplaki2 = np.where(TcellSkoplaki2 < dfsolar['Tamb'], dfsolar['Tamb'], TcellSkoplaki2) # Limits the computated cell temperature lower than Tamb to Tamb. Sometimes Skoplaki has lower Tcell than Tamb.
#dfsolar['Vwind'] = dfsolar['FF']/10
dfsolar['Ppanel'] = PpanelNom / 1000 * dfsolar['panelTot'] * (1 + Pmaxcorr/100 * (dfsolar['Tcell'] - 25))   # Computation panel power based on power correction by temperature difference between Tcell and 25°C from the STC.
PpanelCheck = Apanel * Effpanel * dfsolar['panelTot'] * (1 + Pmaxcorr/100 * (dfsolar['Tcell'] - 25))   # 2nd Computation panel power based on power correction by temperature difference between Tcell and 25°C from the STC.


Profileswitch = 2

if Profileswitch == 1:
    # Computating elementary
    dfsolar['Ppanel'] = Apanel * Effpanel * dfsolar['panelTot']   # Computation panel power based on WattPeak
    dfsolar['Psystem'] = dfsolar['Ppanel'] * nPanels * Effsystem
    print("Computating elementary")
    Esystem = sum(dfsolar['Psystem'])/1000 # [kWh]

elif Profileswitch == 2:
    # Computating basic
    dfsolar['Ppanel'] = PpanelNom / 1000 * dfsolar['panelTot'] * (1 + Pmaxcorr/100 * (dfsolar['Tcell'] - 25))   # Computation panel power based on power correction by temperature difference between Tcell and 25°C from the STC.
    PpanelCheck = Apanel * Effpanel * dfsolar['panelTot'] * (1 + Pmaxcorr/100 * (dfsolar['Tcell'] - 25))   # 2nd Computation panel power based on power correction by temperature difference between Tcell and 25°C from the STC.
    dfsolar['Psystem'] = dfsolar['Ppanel'] * nPanels * Effsystem
    print("Computating basic")
    Esystem = sum(dfsolar['Psystem'])/1000 # [kWh]

elif Profileswitch == 3:
    # Computating Skoplaki mt 1
    dfsolar['PpanelSKO1'] = PpanelNom / 1000 * dfsolar['panelTot'] * (1 + Pmaxcorr/100 * (dfsolar['TcellSkoplaki1'] - 25))
    dfsolar['PsystemSKO1'] = dfsolar['PpanelSKO1'] * nPanels * Effsystem
    print("Computating Skoplaki mt 1")
    Esystem = sum(dfsolar['PsystemSKO1'])/1000 # [kWh]

elif Profileswitch == 4:
    # Computating Skoplaki mt 2
    dfsolar['PpanelSKO2'] = PpanelNom / 1000 * dfsolar['panelTot'] * (1 + Pmaxcorr/100 * (dfsolar['TcellSkoplaki1'] - 25))
    dfsolar['PsystemSKO2'] = dfsolar['PpanelSKO2'] * nPanels * Effsystem
    print("Computating Skoplaki mt 2") 
    Esystem = sum(dfsolar['PsystemSKO2'])/1000 # [kWh]

print(dfsolar)
print("Esystem", Esystem, "[kWh]")
print("Iglo", sum(dfsolar['panelTot']*nPanels)/1000, "[kWh]")
print("Rough efficiency",Esystem/(sum(dfsolar['panelTot']*nPanels*Apanel)/1000)*100,"%")
# print(dfsolar['panelTot'][4000:4048])
# print(dfsolar['Tcell'][4000:4048])


if __name__ == "__main__":
    pass
