# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:05:19 2020

@author: TrungNguyen, PvK, MvdB
"""
from house_buffervessel import house_buffervessel  # exposed function "house" in house module
# function "model" in module house is private

from configurator import load_config, calculateRC
from NEN5060 import nen5060_to_dataframe, run_qsun

from internal_heat_gain import internal_heat_gain
from Temperature_SP     import thermostat_sp, SP_profile

import matplotlib.pyplot as plt

def main():
    house_param = load_config("config2R2C.yml")
    days_sim = house_param['timing']['days_sim']
    CF = house_param['ventilation']['CF']
    Rair_wall, Cwall, Rair_outdoor, Cair = calculateRC(house_param)
    print(days_sim)
    
    #Loading the radiator and buffervessel parameters
    #Heat transfer coefficient of the radiator and het capacity
    cpwater = house_param['radiator']['cpwater']
    rhowater = house_param['radiator']['rhowater']
    Urad = house_param['radiator']['Urad']
    Arad = house_param['radiator']['Arad']
    volumeRadiator = house_param['radiator']['volume_rad']
    UAradiator = Urad * Arad
    Crad =  cpwater*volumeRadiator*rhowater
    
    #Heat capacity of the buffervessel
    volumeBuffervessel = house_param['radiator']['volume_buffervessel']
    Cbuffervessel = cpwater*volumeBuffervessel*rhowater
    

    df_nen = nen5060_to_dataframe()
    df_irr = run_qsun(df_nen)
    print(df_irr.head())

    time_sim = df_irr.iloc[0:days_sim*24, 0].values

    Qsolar = (df_irr.total_E * house_param['glass']['E'] +
              df_irr.total_SE * house_param['glass']['SE'] +
              df_irr.total_S * house_param['glass']['S'] +
              df_irr.total_SW * house_param['glass']['SW'] +
              df_irr.total_W * house_param['glass']['W'] +
              df_irr.total_NW * house_param['glass']['NW'] +
              df_irr.total_N * house_param['glass']['N'] +
              df_irr.total_NE * house_param['glass']['NE']).values
    Qsolar *= house_param['glass']['g_value']
    Qsolar_sim = Qsolar[0:days_sim*24]

    Qint = internal_heat_gain(house_param['internal']['Q_day'],
                              house_param['internal']['delta_Q'],
                              house_param['internal']['t1'],
                              house_param['internal']['t2'])
    Qinternal_sim = Qint[0:days_sim*24]

    Toutdoor = df_nen.loc[:, 'temperatuur'].values / 10.0  # temperature
    T_outdoor_sim = Toutdoor[0:days_sim*24]

    week_day_setpoint = thermostat_sp(house_param['setpoint']['t1'],
                                         house_param['setpoint']['t2'],
                                         house_param['setpoint']['Night_T_SP'],
                                         house_param['setpoint']['Day_T_SP'],
                                         house_param['setpoint']['Flex_T_SP_workday'],
                                         house_param['setpoint']['Wu_time'],
                                         house_param['setpoint']['Work_time'],
                                         house_param['setpoint']['back_home_from_work'])
    
    day_off_setpoint  = thermostat_sp(house_param['setpoint']['t1'],
                                         house_param['setpoint']['t2'],
                                         house_param['setpoint']['Night_T_SP'],
                                         house_param['setpoint']['Day_T_SP'],
                                         house_param['setpoint']['Flex_T_SP_dayoff'],
                                         house_param['setpoint']['Wu_time'],
                                         house_param['setpoint']['shopping_time'],
                                         house_param['setpoint']['back_home'])
    
    SP =SP_profile(week_day_setpoint,day_off_setpoint)
    
    SP_sim = SP[0:days_sim * 24]
    # solve ODE
    data = house_buffervessel(T_outdoor_sim, Qinternal_sim, Qsolar_sim, SP_sim, time_sim,
                 CF, Rair_outdoor, Rair_wall, Cair, Cwall, UAradiator, Crad, Cbuffervessel, cpwater)

    # plot the results
    plt.figure(figsize=(15, 5))         # key-value pair: no spaces
    plt.plot(data [4],data[0], label='Tair')
    plt.plot(data [4],data[1], label='Twall')
    plt.plot(data [4],data[2], label='Treturn')
    plt.plot(data [4],data[3], label='Tbuffervessel')
    plt.plot(time_sim, SP_sim, label='SP_Temperature')
    plt.plot(time_sim,T_outdoor_sim,label='Toutdoor')
    plt.legend(loc='best')
    plt.show()

    
if __name__ == "__main__":
    main()  # temporary solution, recommended syntax
