# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:05:19 2020

@author: TrungNguyen, PvK, MvdB
"""
from housemodel.solvers.house_radiator_matrix_for_loop_discrete_controller import house_radiator_m  # exposed function "house" in house module
# function "model" in module house is private

from housemodel.tools.new_configurator import (load_config,
                                               add_chain_to_k, make_c_inv_matrix)
from housemodel.sourcesink.NEN5060 import nen5060_to_dataframe, run_qsun

from housemodel.sourcesink.internal_heat_gain import internal_heat_gain
from housemodel.controls.Temperature_SP import simple_thermostat

# import matplotlib
# matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

import numpy as np
from scipy.interpolate import interp1d

from pathlib import Path
CONFIGDIR = Path(__file__).parent.absolute()

def main(show=False):
    house_param = load_config(str(CONFIGDIR / "config2R2Ctrans.yml"))
    days_sim = 365 # house_param['timing']['days_sim']
    CF = house_param['ventilation']['CF']

    num_links = len(house_param["chains"][0]["links"])
    cap_list = []
    for n in range(num_links):
        cap_list.append(house_param["chains"][0]["links"][n]["Capacity"])
    cap_mat_inv = make_c_inv_matrix(cap_list)

    cond_list = []
    for n in range(num_links):
        cond_list.append(house_param["chains"][0]["links"][n]["Conductance"])

    cond_mat = add_chain_to_k(np.array([cond_list[0]]), cond_list[1], 0)
    cond_mat = add_chain_to_k(cond_mat, cond_list[2], 0)
    print(days_sim)

    #Loading the radiator and buffervessel parameters
    #Heat transfer coefficient of the radiator and het capacity
    UAradiator = house_param["chains"][0]["links"][2]["Conductance"]
    Crad =  house_param["chains"][0]["links"][2]["Capacity"]

    df_nen = nen5060_to_dataframe()
    df_irr = run_qsun(df_nen)
    print(df_irr.head())

    time_sim = df_irr.iloc[0:days_sim*24, 0].values

    # Interval in seconds the control algorithm
    control_interval = 600

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
    Qint = Qint.flatten()
    Qinternal_sim = Qint[0:days_sim*24]

    Toutdoor = df_nen.loc[:, 'temperatuur'].values / 10.0
    Toutdoor = Toutdoor.flatten()   # temperature
    T_outdoor_sim = Toutdoor[0:days_sim*24]

    SP = simple_thermostat(8, 23, 20, 17)
    SP_sim = SP[0:days_sim * 24].flatten()

    # make predictable part of q_dot vector
    q_vector = np.zeros((num_links,days_sim*24))
    leak_to_amb = house_param["chains"][0]["links"][0]["Conductance"]
    q_vector[0,:] = (T_outdoor_sim * leak_to_amb) + Qinternal_sim + CF * Qsolar_sim
    q_vector[1,:] = (1 - CF) * Qsolar_sim

    # Interpolation of data
    interp_func = interp1d(time_sim, q_vector)
    interp_func_SP = interp1d(time_sim, SP_sim)
    interp_func_Toutdoor = interp1d(time_sim, T_outdoor_sim)
    q_vector = interp_func(np.arange(0, time_sim[-1], control_interval))
    SP_sim = interp_func_SP(np.arange(0, time_sim[-1], control_interval))
    T_outdoor_sim = interp_func_Toutdoor(np.arange(0, time_sim[-1], control_interval))
    time_sim = np.arange(0, time_sim[-1], control_interval)

    # solve ODE
    data = house_radiator_m(cap_mat_inv, cond_mat, q_vector,
                            SP_sim, time_sim, control_interval)

    # if show=True, plot the results
    if show:
        plt.figure(figsize=(15, 5))         # key-value pair: no spaces
        plt.plot(data [3],data[0], label='Tair')
        plt.plot(data [3],data[1], label='Twall')
        plt.plot(data [3],data[2], label='Tradiator')
        plt.plot(time_sim, SP_sim, label='SP_Temperature')
        plt.plot(time_sim,T_outdoor_sim,label='Toutdoor')
        plt.legend(loc='best')
        plt.title("Simulation2R2C_radiator_matrix_loop")
        plt.show()

    return time_sim, SP_sim, T_outdoor_sim, data

    
if __name__ == "__main__":
    main(show=True)  # temporary solution, recommended syntax
