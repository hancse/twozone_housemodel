# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:05:19 2020

@author: TrungNguyen, PvK, MvdB
"""
from house_model_with_HeatingSystemClass import house_radiator_m  # exposed function "house" in house module
# function "model" in module house is private

from housemodel.tools.new_configurator import (load_config,
                                               add_chain_to_k, make_c_inv_matrix)
from housemodel.sourcesink.NEN5060 import run_qsun
# from housemodel.sourcesink.NEN5060 import nen5060_to_dataframe
from housemodel.weather_solar.weatherdata import read_nen_weather_from_xl

from housemodel.sourcesink.internal_heat_gain import internal_heat_gain
from housemodel.controls.Temperature_SP import simple_thermostat

# import matplotlib
# matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

import logging
logging.basicConfig()
logger = logging.getLogger('matrix')
logger.setLevel(logging.INFO)

from pathlib import Path
CONFIGDIR = Path(__file__).parent.absolute()

def main(show=False, xl=False):
    house_param = load_config(str(CONFIGDIR / "excel_for_companies.yaml"))
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

    # df_nen = nen5060_to_dataframe()
    df_nen = read_nen_weather_from_xl()

    df_irr = run_qsun(df_nen)
    print(df_irr.head())

    time_sim = df_irr.iloc[0:days_sim*24, 0].values

    # Interval in seconds the control algorithm
    control_interval = house_param["Timescale"]*60

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

    Toutdoor = df_nen.loc[:, 'temperatuur'].values
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
    interp_func = interp1d(time_sim, q_vector, fill_value='extrapolate')
    interp_func_SP = interp1d(time_sim, SP_sim, fill_value='extrapolate')
    interp_func_Q_internal = interp1d(time_sim, Qinternal_sim, fill_value='extrapolate')
    interp_func_Toutdoor = interp1d(time_sim, T_outdoor_sim, fill_value='extrapolate')
    q_vector = interp_func(np.arange(0, time_sim[-1]+(6*600), control_interval))
    SP_sim = interp_func_SP(np.arange(0, time_sim[-1]+(6*600), control_interval))
    T_outdoor_sim = interp_func_Toutdoor(np.arange(0, time_sim[-1]+(6*600), control_interval))
    Qinternal_sim = interp_func_Q_internal(np.arange(0, time_sim[-1]+(6*600), control_interval))
    time_sim = np.arange(0, time_sim[-1]+(6*600), control_interval)

    # time_sim = np.linspace(0, time_sim[-1], (8760-1)*6, endpoint=False)


    # Input PID values in to control
    control_parameters = np.zeros(3)
    control_parameters[0] = house_param['controller']['kp']
    control_parameters[1] = house_param['controller']['ki']
    control_parameters[2] = house_param['controller']['kd']

    # solve ODE
    data = house_radiator_m(cap_mat_inv, cond_mat, q_vector,
                            SP_sim, time_sim, control_interval, control_parameters)

    # if show=True, plot the results
    if show:
        plt.figure(figsize=(15, 5))         # key-value pair: no spaces
        plt.plot(data[0],data[1], label='Tair')
        plt.plot(data[0],data[2], label='Twall')
        plt.plot(data[0],data[3], label='Tradiator')
        plt.plot(time_sim, SP_sim, label='SP_Temperature')
        plt.plot(time_sim,T_outdoor_sim,label='Toutdoor')
        plt.plot(data[0], data[4], label='Qinst')
        plt.legend(loc='best')
        plt.title("Simulation2R2C_companies")
        plt.show()

    if xl:
        # df_out = pd.DataFrame(data[0], columns=['Timestep'])
        df_out = pd.DataFrame({'Timestep': data[0]})
        df_out['Outdoor temperature'] = T_outdoor_sim
        for n in range(num_links):
            nodename = house_param['chains'][0]['links'][n]['Name']
            df_out["T_{}".format(n)] = data[n+1].tolist()
            # df_out["Solar_{}".format(n)] = Qsolar_sim[n, :]
            if nodename == 'Internals':
                df_out["Internal_{}".format(n)] = Qinternal_sim

        df_out['Tradiator'] = data[3].tolist()
        df_out["Heating"] = data[4].tolist()

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

if __name__ == "__main__":
    main(show=True, xl=True)  # temporary solution, recommended syntax
