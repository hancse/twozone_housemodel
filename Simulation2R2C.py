# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:05:19 2020

@author: TrungNguyen, PvK
"""
from house import house  # exposed function "house" in house module
# function "model" in module house is private

from configurator import load_config, calculateRC
from NEN5060      import nen5060_to_dataframe, run_qsun

from internal_heat_gain import internal_heat_gain
from Temperature_SP     import temp_sp
from Setpoint_profileV1     import thermostat_sp, SP_profile

import matplotlib.pyplot as plt


def main():
    house_param = load_config("config2R2C.yml")
    days_sim = house_param['timing']['days_sim']
    CF = house_param['ventilation']['CF']
    
    [Rair_wall_z1, Rair_wall_z2, Rair_cc, 
    Rair_outdoor_z1, Rair_outdoor_z2, Rair_z12, 
    Rair_z21, Cair_z1, Cair_z2, Cwall_z1, 
    Cwall_z2, Cwall_cc, Rair_wall] = calculateRC(house_param)
    
    print('Simulation days:', days_sim)

    df_nen = nen5060_to_dataframe()
    df_irr = run_qsun(df_nen)
    #df_weeks = read_week('NEN_data')
    print(df_irr.head())

    time_sim = df_irr.iloc[0:days_sim*24, 0].values

    Qsolar = (df_irr.total_E * house_param['glass_z1']['E'] +
              df_irr.total_SE * house_param['glass_z1']['SE'] +
              df_irr.total_S * house_param['glass_z1']['S'] +
              df_irr.total_SW * house_param['glass_z1']['SW'] +
              df_irr.total_W * house_param['glass_z1']['W'] +
              df_irr.total_NW * house_param['glass_z1']['NW'] +
              df_irr.total_N * house_param['glass_z1']['N'] +
              df_irr.total_NE * house_param['glass_z1']['NE']).values
    Qsolar *= house_param['glass_z1']['g_value']
    Qsolar *= 2
    Qsolar_sim = Qsolar[0:days_sim*24]
    #print(len(Qsolar_sim))

    Qint = internal_heat_gain(house_param['internal']['Q_day'],
                              house_param['internal']['delta_Q'],
                              house_param['internal']['t1'],
                              house_param['internal']['t2'])
    Qinternal_sim = Qint[0:days_sim*24]

    Toutdoor = df_nen.loc[:, 'temperatuur'].values / 10.0  # temperature
    T_outdoor_sim = Toutdoor[0:days_sim*24]
    #plt.plot(T_outdoor_sim)
    
    week_day_setpoint = thermostat_sp(house_param['setpoint']['t1'],
                                         house_param['setpoint']['t2'],
                                         house_param['setpoint']['Night_T_SP'],
                                         house_param['setpoint']['Day_T_SP'],
                                         house_param['setpoint']['Wu_time'],
                                         house_param['setpoint']['Work_time'],
                                         house_param['setpoint']['back_home_from_work'])
    
    day_off_setpoint  = thermostat_sp(house_param['setpoint']['t1'],
                                         house_param['setpoint']['t2'],
                                         house_param['setpoint']['Night_T_SP'],
                                         house_param['setpoint']['Day_T_SP'],
                                         house_param['setpoint']['Wu_time'],
                                         house_param['setpoint']['shopping_time'],
                                         house_param['setpoint']['back_home'])
    
    SP =SP_profile(week_day_setpoint,day_off_setpoint)
    
    #SP = temp_sp(house_param['setpoint']['t1'],
    #             house_param['setpoint']['t2'],
    #             house_param['setpoint']['Night_T_SP'],
    #             house_param['setpoint']['Day_T_SP'],
    #             house_param['setpoint']['Wu_time'],
    #             house_param['setpoint']['Work_time'],
    #             house_param['setpoint']['back_home'])
    
    
    SP_sim = SP[0:days_sim * 24]
    
    # Controller value
    
    kp = house_param['controller']['kp']

    # solve ODE
    data = house(T_outdoor_sim,Qinternal_sim,Qsolar_sim,SP_sim,time_sim,
                 CF,Rair_outdoor_z1,Rair_wall_z1,Cair_z1,
                 Cwall_z1,Rair_z12,Rair_z21,Rair_cc,Cwall_cc,kp)

    # plot the results
    plt.figure(figsize=(15, 5))         # key-value pair: no spaces
    plt.plot(data[0], label='Tair_zone1')
    plt.plot(data[1], label='Twall')
    plt.plot(data[2], label='Tair_zone2')
    plt.plot(SP_sim, label='SP_Temperature')
    # plt.plot(T_outdoor_sim,label='Toutdoor')
    plt.legend(loc='best')
    plt.show()
    
    '''
    The fluctuation : resolution is 1 hour, use kp only, A_internalmass is small
    '''


if __name__ == "__main__":
    main()  # temporary solution, recommended syntax
