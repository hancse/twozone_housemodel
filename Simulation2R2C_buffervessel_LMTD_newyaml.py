# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:05:19 2020

@author: TrungNguyen, PvK, MvdB
"""
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

from house_buffervessel_LMTD import house_buffervessel  # exposed function "house" in house module
# function "model" in module house is private

from new_configurator import load_config, calculateRC
from NEN5060 import nen5060_to_dataframe, run_qsun_new

from internal_heat_gain import internal_heat_gain
from Temperature_SP import simple_thermostat


def main():
    house_param = load_config("Tussenwoning_alt.yaml")
    days_sim = house_param['timing']['days_sim']
    print('Simulation days:', days_sim)

    Rair_wall, Cwall, Rair_outdoor, Cair = calculateRC(house_param)

    # Loading the radiator and buffervessel parameters
    # Heat transfer coefficient of the radiator and het capacity
    cpwater = house_param['radiator']['cpwater']
    rhowater = house_param['radiator']['rhowater']
    Urad = house_param['radiator']['Urad']
    Arad = house_param['radiator']['Arad']
    volumeRadiator = house_param['radiator']['volume_rad']
    UAradiator = Urad * Arad
    Crad = cpwater * volumeRadiator * rhowater

    # Heat capacity of the buffervessel
    volumeBuffervessel = house_param['radiator']['volume_buffervessel']
    Cbuffervessel = cpwater * volumeBuffervessel * rhowater

    df_nen = nen5060_to_dataframe()

    Qsolar = np.zeros(365*24)
    for s in house_param['chains'][0]['Solar_irradiation']:
        descr = s['Designation']
        az = s['azimuth']
        tlt = s['tilt']
        print("Azimuth {0} and Tilt {1} for {2}".format(az, tlt, descr))
        df_irr = run_qsun_new(df_nen, az, tlt)
        Qsolar += (df_irr.total_irr).values
        time_sim = df_irr.iloc[0:days_sim * 24, 0].values

    CF = house_param['ventilation']['CF']
    Qsolar_sim = Qsolar[0:days_sim * 24]

    Q_internal = internal_heat_gain(house_param['internal']['Q_day'],
                              house_param['internal']['delta_Q'],
                              house_param['internal']['t1'],
                              house_param['internal']['t2'])
    Qinternal_sim = Q_internal[0:days_sim * 24]

    Toutdoor = df_nen.loc[:, 'temperatuur'].values / 10.0  # temperature
    T_outdoor_sim = Toutdoor[0:days_sim * 24]

    t_on = house_param['chains'][0]['Controller'][0]['Set_time'][0]
    t_off = house_param['chains'][0]['Controller'][0]['Set_time'][1]
    T_day = house_param['chains'][0]['Controller'][0]['Set_temp'][0]
    T_night = house_param['chains'][0]['Controller'][0]['Set_temp'][1]
    SP = simple_thermostat(t_on, t_off, T_day, T_night)
    SP_sim = SP[0:days_sim * 24]

    # solve ODE
    data = house_buffervessel(T_outdoor_sim, Qinternal_sim, Qsolar_sim, SP_sim, time_sim,
                              CF, Rair_outdoor, Rair_wall, Cair, Cwall, UAradiator, Crad, Cbuffervessel, cpwater)

    # plot the results
    plt.figure(figsize=(15, 5))  # key-value pair: no spaces
    plt.plot(data[4], data[0], label='Tair')
    plt.plot(data[4], data[1], label='Twall')
    plt.plot(data[4], data[2], label='Treturn')
    plt.plot(data[4], data[3], label='Tbuffervessel')
    plt.plot(time_sim, SP_sim, label='SP_Temperature')
    plt.plot(time_sim, T_outdoor_sim, label='Toutdoor')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    main()  # temporary solution, recommended syntax
