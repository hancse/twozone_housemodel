# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:05:19 2020

@author: TrungNguyen, PvK, MvdB
"""
from housemodel.solvers.house_buffervessel_newyaml import house_buffervessel  # exposed function "house" in house module
# function "model" in module house is private

from housemodel.tools.new_configurator import (load_config,
                                            make_c_matrix,
                                            make_k_matrix)
from housemodel.sourcesink.NEN5060 import nen5060_to_dataframe, run_qsun_new

from housemodel.sourcesink.internal_heat_gain import internal_heat_gain
from housemodel.controls.Temperature_SP import simple_thermostat

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

import logging
logging.basicConfig()
logger = logging.getLogger('matrix')
logger.setLevel(logging.INFO)

def main():

    # make 8760 Q vectors
    q_vector = np.zeros((num_nodes, 8760))
    # T_ambient
    T_outdoor = df_nen.loc[:, 'temperatuur'].values / 10.0  # temperature
    q_vector[0,:] = conductance[0] * T_outdoor.T

    Qsolar = np.zeros((8760, num_nodes))  # 8760 rows x 2 cols
    for s in chain['Solar_irradiation']:
        descr = s['Designation']
        az = s['Azimuth']
        tlt = s['Tilt']

        df_irr = run_qsun_new(df_nen, az, tlt, north_is_zero=True)
        # area =  s['Effective Area']
        # partfactor = s['Node_partition'] # list of num_nodes elements 0<x<1
        # logger.info(f"Window area {area} @ Azimuth {az} and Tilt {tlt} for {descr}, divided {partfactor[0]} {partfactor[1]}")
        for n in range(num_nodes):
            Qsolar[:, n] += (df_irr.total_irr).values * area * partfactor[n]

        time_year = df_irr.iloc[:, 0].values  # 8760 rows 1D

    time_hr = time_year / 3600
    # plot the results
    # plt.figure(figsize=(15, 5))         # key-value pair: no spaces
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                                  figsize=(15,
                                                           12))  # 3 rijen en 1 kolom. x-axis will be shared among all subplots.
    ax1.plot(time_hr, Qsolar[:,0], label='0')
    ax1.set_ylabel('Irradiation ($W/m^2$)')
    # ax1.set_ylim(-100, 1200)
    ax1.legend()

    ax2.plot(time_hr, Qsolar[:,1], label='1')
    ax2.set_ylabel('Irradiation ($W/m^2$)')
    # ax2.set_ylim(-100, 1200)
    ax2.legend()
    plt.show()

    """
    sources = house_param['chains'][0]['Sources']
    for src in sources:
        if src['name'] == 'Internal_load':
            # Q_internal = np.zeros(8760)  # 8760 rows x 1 cols
            Q_internal = internal_heat_gain(src['Set_load'][0],
                                                  src['Set_load'][1],
                                                  src['Set_time'][0],
                                                  src['Set_time'][1])



    t_on = house_param['chains'][0]['Control']['Set_time'][0]
    t_off = house_param['chains'][0]['Control']['Set_time'][1]
    T_day = house_param['chains'][0]['Control']['Set_temp'][0]
    T_night = house_param['chains'][0]['Control']['Set_temp'][1]
    SP = simple_thermostat(t_on, t_off, T_day, T_night)

    num_sim = 8760
    time_sim = time_year[0:num_sim] # 480 cols 1D
    Qsolar_sim = Qsolar[0:num_sim, :].T   # 2 rows x 480 cols
    Qinternal_sim = Q_internal[0:num_sim].flatten()  # 480 cols 1D
    T_outdoor_sim = T_outdoor[0:num_sim].flatten()  # 480 cols 1D
    SP_sim = SP[0:num_sim]  # 480 cols 1D


    # solve ODE
    data = house_buffervessel(c_matrix, k_matrix, q_vector,
                              SP_sim, time_sim,
                              UAradiator, Crad, Cbuffervessel, cpwater)

    # df_out = pd.DataFrame(data[4], columns=['Timestep'])
    df_out = pd.DataFrame({'Timestep': data[4]})
    df_out['Outdoor temperature'] = T_outdoor_sim
    for n in range(num_nodes):
        nodename = house_param['chains'][0]['links'][n]['Name']
        df_out["T_{}".format(n)] = data[0].tolist()
        df_out["Solar_{}".format(n)] = Qsolar_sim[n, :]
        if nodename == 'Internals':
            df_out["Internal_{}".format(n)] = Qinternal_sim
        df_out["Heating_{}".format(n)] = 0

    df_out['Treturn'] = data[2].tolist()
    df_out['Tbuffervessel'] = data[3]

    wb = Workbook()
    ws = wb.active
    ws.append(['DESCRIPTION',
                'Resultaten HAN Dynamic Model Heat Built Environment'])
    ws.append(['Chain number', 0])
    ws.append(['Designation', None, '2R-2C-1-zone',
               None, None, None, '2R-2C-1-zone'])
    ws.append(['Node number', None, 0, None, None, None, 1])
    ws.append(['Designation', None,
               house_param['chains'][0]['links'][0]['Name'], None, None, None,
               house_param['chains'][0]['links'][1]['Name']])
    for r in dataframe_to_rows(df_out, index=False):
        ws.append(r)
    # df_out.to_excel('tst.xlsx', index=False, startrow=10)
    wb.save('tst.xlsx')

    timeaxis = data[4]/3600

    # plot the results
    # plt.figure(figsize=(15, 5))         # key-value pair: no spaces
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(15, 8))  # 3 rijen en 1 kolom. x-axis will be shared among all subplots.
    ax1.plot(timeaxis, data[0], label='Tair')
    ax1.plot(timeaxis, data[1], label='Twall')
    ax1.plot(time_sim/3600, SP_sim, label='SP_Temperature')
    ax1.set_ylabel('T ($\degree C$)')
    ax1.set_ylim(15, 40)

    ax2.plot(timeaxis, data[2], label='Treturn')
    ax2.plot(timeaxis, data[3], label='Tbuffervessel')
    ax2.set_ylabel('T ($\degree C$)')
    ax2.set_ylim(10, 90)

    ax3.plot(time_sim/3600,T_outdoor_sim,label='Toutdoor')
    ax3.set_ylabel('T ($\degree C$)')
    ax3.set_ylim(-10, 30)
    ax1.set_title('Simulation2R2C_buffervessel_newyaml_with_xl')
    ax3.set_xlabel('time (h)')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()
    """

if __name__ == "__main__":
    main()  # temporary solution, recommended syntax

