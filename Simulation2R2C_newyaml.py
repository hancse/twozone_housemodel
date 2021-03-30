# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:05:19 2020

@author: TrungNguyen, PvK, MvdB
"""
from house import house  # exposed function "house" in house module
# function "model" in module house is private

# from configurator import load_config, calculateRCOne
from new_configurator import load_config, calculateRC
from NEN5060      import nen5060_to_dataframe, run_qsun_new

from internal_heat_gain import internal_heat_gain
#from Temperature_SP     import temp_sp
from Temperature_SP     import thermostat_sp, SP_profile

import matplotlib.pyplot as plt
import numpy as np


def main():
    house_param = load_config("Tussenwoning_alt.yaml")
    days_sim = house_param['timing']['days_sim']
    print('Simulation days:', days_sim)

    Rair_wall, Cwall, Rair_outdoor, Cair = calculateRC(house_param)

    df_nen = nen5060_to_dataframe()

    Qsolar = np.zeros(days_sim*24)
    for s in house_param['chains'][0]['Solar_irradiation']:
        descr = s['Designation']
        az = s['azimuth']
        tlt = s['tilt']
        print("Azimuth {0} and Tilt {1} for {2}".format(az, tlt, descr))
        df_irr = run_qsun_new(df_nen, az, tlt)
        Qsolar += (df_irr.total_irr).values
        time_sim = df_irr.iloc[0:days_sim * 24, 0].values
        CF = house_param['ventilation']['CF']

    # Q_internal = np.zeros(days_sim*24)
    Q_internal = internal_heat_gain(house_param['internal']['Q_day'],
                              house_param['internal']['delta_Q'],
                              house_param['internal']['t1'],
                              house_param['internal']['t2'])
    Q_internal = Q_internal.flatten()

    T_outdoor = df_nen.loc[:, 'temperatuur'].values / 10.0  # temperature
    # T_outdoor_sim = Toutdoor[0:days_sim*24]
    # plt.plot(T_outdoor_sim)
    
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
    
    SP = SP_profile(week_day_setpoint,day_off_setpoint)
    
    #SP = temp_sp(house_param['setpoint']['t1'],
    #             house_param['setpoint']['t2'],
    #             house_param['setpoint']['Night_T_SP'],
    #             house_param['setpoint']['Day_T_SP'],
    #             house_param['setpoint']['Wu_time'],
    #             house_param['setpoint']['Work_time'],
    #             house_param['setpoint']['back_home'])
    
    
    # SP_sim = SP[0:days_sim * 24]


    # addition NTA8800 house model
    
    # Controller value
    
    kp = house_param['chains'][0]['Control']['kp']

    # solve ODE
    data = house(T_outdoor, Q_internal, Qsolar, SP, time_sim,
                 CF, Rair_outdoor, Rair_wall, Cair, Cwall, kp)

    # plot the results
    plt.figure(figsize=(15, 5))         # key-value pair: no spaces
    plt.plot(data[0], label='Tair')
    plt.plot(data[1], label='Twall')
    plt.plot(SP, label='SP_Temperature')
    # plt.plot(T_outdoor_sim,label='Toutdoor')
    plt.legend(loc='best')
    plt.show()
    
    '''
    The fluctuation : resolution is 1 hour, use kp only, A_internalmass is small
    '''


if __name__ == "__main__":
    main()  # temporary solution, recommended syntax
