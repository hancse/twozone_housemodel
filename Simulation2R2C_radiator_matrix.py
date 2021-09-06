# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:05:19 2020

@author: TrungNguyen, PvK, MvdB
"""
from housemodel.solvers.house_radiator_matrix import house_radiator  # exposed function "house" in house module
# function "model" in module house is private

# from housemodel.tools.configurator import load_config, calculateRCOne
from housemodel.tools.new_configurator import load_config, calculateRC
from housemodel.sourcesink.NEN5060 import nen5060_to_dataframe, run_qsun

from housemodel.sourcesink.internal_heat_gain import internal_heat_gain
from housemodel.controls.Temperature_SP import simple_thermostat

import matplotlib.pyplot as plt

from pathlib import Path
CONFIGDIR = Path(__file__).parent.absolute()

def main(show=False):
    house_param = load_config(str(CONFIGDIR / "config2R2Ctrans.yml"))
    days_sim = 365 # house_param['timing']['days_sim']
    CF = house_param['ventilation']['CF']
    Rair_wall, Cwall, Rair_outdoor, Cair = calculateRC(house_param)
    print(days_sim)
    
    #Loading the radiator and buffervessel parameters
    #Heat transfer coefficient of the radiator and het capacity
    UAradiator = house_param["chains"][0]["links"][2]["Conductance"]
    Crad =  house_param["chains"][0]["links"][2]["Capacity"]
    
    #Heat capacity of the buffervessel
    # volumeBuffervessel = house_param['radiator']['volume_buffervessel']
    # Cbuffervessel = house_param["chains"][0]["links"][3]["Capacity"]

    df_nen = nen5060_to_dataframe()
    df_irr = run_qsun(df_nen)
    print(df_irr.head())

    time_sim = df_irr.iloc[0:days_sim*24, 0].values

    Qsolar = (df_irr.total_E * house_param['solar_irradiation']['E'] +
              df_irr.total_SE * house_param['solar_irradiation']['SE'] +
              df_irr.total_S * house_param['solar_irradiation']['S'] +
              df_irr.total_SW * house_param['solar_irradiation']['SW'] +
              df_irr.total_W * house_param['solar_irradiation']['W'] +
              df_irr.total_NW * house_param['solar_irradiation']['NW'] +
              df_irr.total_N * house_param['solar_irradiation']['N'] +
              df_irr.total_NE * house_param['solar_irradiation']['NE']).values
    Qsolar *= house_param['solar_irradiation']['g_value']
    Qsolar_sim = Qsolar[0:days_sim*24]

    Qint = internal_heat_gain(house_param['internal']['Q_day'],
                              house_param['internal']['delta_Q'],
                              house_param['internal']['t1'],
                              house_param['internal']['t2'])
    Qinternal_sim = Qint[0:days_sim*24]

    Toutdoor = df_nen.loc[:, 'temperatuur'].values / 10.0  # temperature
    T_outdoor_sim = Toutdoor[0:days_sim*24]

    SP = simple_thermostat(8, 23, 20, 17)
    SP_sim = SP[0:days_sim * 24]
    # solve ODE
    data = house_radiator(T_outdoor_sim, Qinternal_sim, Qsolar_sim, SP_sim, time_sim,
                 CF, Rair_outdoor, Rair_wall, Cair, Cwall, UAradiator, Crad)

    # if show=True, plot the results
    if show:
        plt.figure(figsize=(15, 5))         # key-value pair: no spaces
        plt.plot(data [3],data[0], label='Tair')
        plt.plot(data [3],data[1], label='Twall')
        plt.plot(data [3],data[2], label='Tradiator')
        plt.plot(time_sim, SP_sim, label='SP_Temperature')
        plt.plot(time_sim,T_outdoor_sim,label='Toutdoor')
        plt.legend(loc='best')
        plt.show()

    return time_sim, SP_sim, T_outdoor_sim, data

    
if __name__ == "__main__":
    main(show=True)  # temporary solution, recommended syntax
