# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:05:19 2020

@author: TrungNguyen, PvK, MvdB
"""
from housemodel.solvers.house_buffervessel import house_buffervessel  # exposed function "house" in house module
# function "model" in module house is private

# from housemodel.tools.configurator import load_config, calculateRCOne
from housemodel.tools.new_configurator import (load_config,
                                               make_c_inv_matrix,
                                               make_k_minus_matrix)

from housemodel.sourcesink.NEN5060 import nen5060_to_dataframe, run_qsun, run_qsun_new

from housemodel.sourcesink.internal_heat_gain import internal_heat_gain
from housemodel.controls.Temperature_SP import simple_thermostat

import matplotlib.pyplot as plt
import numpy as np

import logging
logging.basicConfig()
logger = logging.getLogger('buffervessel_matrix')
logger.setLevel(logging.INFO)

from pathlib import Path
CONFIGDIR = Path(__file__).parent.absolute()

def main(show=False):
    house_param = load_config(str(CONFIGDIR / "Tussenwoning2R2C_simple.yml"))
    days_sim = 365 # house_param['timing']['days_sim']
    num_nodes = 2
    CF = house_param['ventilation']['CF']

    Cair = house_param['thermal']['capacity'][0]
    Cwall = house_param['thermal']['capacity'][1]

    Rair_outdoor = 1.0/house_param['thermal']['conductance'][0]
    Rair_wall = 1.0/house_param['thermal']['conductance'][1]

    print(1.0/Rair_outdoor, Cair, 1.0/Rair_wall, Cwall)
    print(days_sim)

    c_matrix = make_c_inv_matrix(house_param['thermal']['capacity'])
    logger.info(f"C matrix: \n {c_matrix}")

    k_matrix = make_k_minus_matrix(house_param['thermal']['conductance'])
    logger.info(f"K matrix: \n {k_matrix}")

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

    """
    # old way: Qsolar as sum of transparent windows in 8 directions and horizontal
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
    """

    # new way: Qsolar2 as loop over transparent surfaces with:
    # effective area = area * ZTA
    # orientation in azimuth and tilt (inclination)
    # partition factor over nodes
    # inner loop over node divides energy of each window over nodes
    # result: array with num_nodes rows
    # containing 8760 values for heat delivered to each node
    time_secs = np.zeros(8760)
    Qsolar2 = np.zeros((num_nodes, 8760))  # num_nodes rows x 8760 cols
    for s in house_param['Solar_irradiation']:
        descr = s['Designation']
        az = s['Azimuth']
        tlt = s['Tilt']
        area = s['Effective Area']
        df_irr2 = run_qsun_new(df_nen, az, tlt, north_is_zero=True)
        area =  s['Effective Area']
        partfactor = s['Node_partition'] # list of num_nodes elements 0<x<1
        logger.info(f"Window area {area} @ Azimuth {az} and Tilt {tlt} for {descr}, divided {partfactor[0]} {partfactor[1]}")
        for n in range(num_nodes):
            Qsolar2[n, :] += (df_irr2.total_irr * area * partfactor[n]).values
        time_secs = df_irr2.iloc[:, 0].values  # 8760 rows 1D

    time_sim = time_secs[0:days_sim*24]
    Qsolar2 *= house_param['glass']['g_value']
    Qsolar2_sum = np.sum(Qsolar2, axis=0)
    # logger.info(f"Testing if Qsolar == Qsolar2_sum")
    # np.testing.assert_allclose(Qsolar, Qsolar2_sum)
    # tested to be true!

    # Qsolar_sim = Qsolar[0:days_sim * 24]
    Qsolar_sim = Qsolar2_sum[0:days_sim*24]

    Qint = internal_heat_gain(house_param['internal']['Q_day'],
                              house_param['internal']['delta_Q'],
                              house_param['internal']['t1'],
                              house_param['internal']['t2'])
    Qinternal_sim = Qint[0:days_sim*24]

    Toutdoor = df_nen.loc[:, 'temperatuur'].values / 10.0  # temperature
    T_outdoor_sim = Toutdoor[0:days_sim*24]

    t_on = house_param['control']['set_time'][0]
    t_off = house_param['control']['set_time'][1]
    T_day = house_param['control']['set_temp'][0]
    T_night = house_param['control']['set_temp'][1]
    SP = simple_thermostat(t_on, t_off, T_day, T_night)
    
    SP_sim = SP[0:days_sim * 24]


    # solve ODE
    data = house_buffervessel(T_outdoor_sim, Qinternal_sim, Qsolar_sim, SP_sim, time_sim,
                 CF, Rair_outdoor, Rair_wall, Cair, Cwall, UAradiator, Crad, Cbuffervessel, cpwater)

    # if show=True, plot the results
    if show:
        plt.figure(figsize=(15, 5))         # key-value pair: no spaces
        plt.plot(data [4],data[0], label='Tair')
        plt.plot(data [4],data[1], label='Twall')
        plt.plot(data [4],data[2], label='Treturn')
        plt.plot(data [4],data[3], label='Tbuffervessel')
        plt.plot(time_sim, SP_sim, label='SP_Temperature')
        plt.plot(time_sim,T_outdoor_sim,label='Toutdoor')
        plt.legend(loc='best')
        plt.show()

    return time_sim, SP_sim, T_outdoor_sim, data

    
if __name__ == "__main__":
    main(show=True)  # temporary solution, recommended syntax
