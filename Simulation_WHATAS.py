# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:05:19 2020

@author: TrungNguyen, PvK, MvdB
"""
from housemodel.solvers.Apartment_model_WHATAS import house_radiator_m  # exposed function "house" in house module
# function "model" in module house is private

from housemodel.tools.new_configurator import (load_config,
                                               add_chain_to_k, make_c_inv_matrix)
from housemodel.sourcesink.NEN5060 import run_qsun

from housemodel.sourcesink.internal_heat_gain import internal_heat_gain
from housemodel.controls.Temperature_SP import simple_thermostat

from housemodel.weather_solar.weatherdata import (read_nen_weather_from_xl,
                                                  NENdatehour2datetime)

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
    house_param = load_config(str(CONFIGDIR / "Simulation_WHATAS.yaml"))
    days_sim = 5
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

    leak_construction = (1/3) * house_param["chains"][0]["links"][0]["Conductance"]

    cond_mat[1, 1] += leak_construction

    # read NEN5060 data from spreadsheet NEN5060-2018.xlsx into pandas DataFrame
    df_nen = read_nen_weather_from_xl()
    # generate and insert timezone-aware UTC and local timestamps (with DST)
    df_nen = NENdatehour2datetime(df_nen)
    df_irr = run_qsun(df_nen)
    time_sim = df_irr.iloc[0:days_sim*24, 0].values

    # Interval in seconds the control algorithm
    control_interval = house_param["Timescale"]*60

    #Read the waterflow demand
    df_water = pd.read_excel(open(r'appartementencomplex.xlsx', 'rb'),
                       sheet_name='Tapwater', skiprows=[0, 1, 2, 3])

    time_water = df_water.iloc[:]["Seconden"]
    water_demand = df_water.iloc[:]["(l/min)"]
    water_time_sim = np.arange(0, 86400, 600)
    interp_func_water = interp1d(time_water, water_demand, fill_value='extrapolate')
    waterflow = interp_func_water(np.arange(0, water_time_sim[-1] + 600, control_interval))
    waterflow_sim = np.tile(waterflow, days_sim)

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

    SP = simple_thermostat(8, 23, 20, 18)
    SP_sim = SP[0:days_sim * 24].flatten()

    # make predictable part of q_dot vector
    q_vector = np.zeros((num_links,days_sim*24))
    leak_to_amb = house_param["chains"][0]["links"][0]["Conductance"]
    q_vector[0,:] = (T_outdoor_sim * leak_to_amb) + Qinternal_sim + CF * Qsolar_sim
    q_vector[1,:] = (T_outdoor_sim * leak_construction) + (1 - CF) * Qsolar_sim

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

    #interpolation of radiator data
    df_radiator = pd.read_excel(open(r'appartementencomplex.xlsx', 'rb'),
                             sheet_name='Radiator', skiprows=[0, 1, 2,3])
    Power_normalized = df_radiator.iloc[:]["P/Pref radiator"]
    Flow_normalized = df_radiator.iloc[:]["Flow"]
    Return_temp = df_radiator.iloc[:]["Tretour"]
    interp_func_retour = interp1d(Power_normalized, Return_temp, fill_value='extrapolate')
    interp_func_flow = interp1d(Power_normalized, Flow_normalized, fill_value='extrapolate')

    # Input PID values in to control
    control_parameters = np.zeros(4)
    control_parameters[0] = house_param['controller']['kp']
    control_parameters[1] = house_param['controller']['ki']
    control_parameters[2] = house_param['controller']['kd']
    control_parameters[3] = house_param['controller']['Maximum_power']

    # solve ODE
    data = house_radiator_m(cap_mat_inv, cond_mat, q_vector,
                            SP_sim, time_sim, control_interval, control_parameters, T_outdoor_sim, waterflow_sim, interp_func_retour, interp_func_flow)

    # if show=True, plot the results
    if show:
        fig, ax = plt.subplots(2, 2, sharex=True)
        ax[0, 0].plot(data[0],data[1], label='Tair')
        ax[0, 0].plot(data[0],data[2], label='Twall')
        ax[0, 0].plot(data[0], SP_sim, label='SP_Temperature')
        ax[0, 0].plot(data[0], T_outdoor_sim, label='Toutdoor')
        ax[0, 0].legend(loc='upper right')
        ax[0, 0].set_title('Nodal Temperatures')
        ax[0, 0].set_xlabel(('Time (s)'))
        ax[0, 0].set_ylabel(('Temperature (°C)'))

        ax[1, 0].plot(data[0], data[3]/1000, label='Power', color='c')
        ax[1, 0].legend(loc='upper right')
        ax[1, 0].set_title('Power')
        ax[1, 0].set_xlabel(('Time (s)'))
        ax[1, 0].set_ylabel(('Power (kW)'))

        ax[1, 1].plot(data[0], data[7]/1000, label='Power Buffervessel',color='b')
        ax[1, 1].plot(data[0], data[9]/1000, label='Power Buffervessel', color='g')
        ax[1, 1].legend(loc='upper right')
        ax[1, 1].set_title('Thermal power heat pump')
        ax[1, 1].set_xlabel(('Time (s)'))
        ax[1, 1].set_ylabel(('Power [kW]'))

        ax[0, 1].plot(data[0], data[5], label='Top')
        ax[0, 1].plot(data[0], data[6], label='Bottom')


        #ax[0, 1].legend(loc='upper right')
        #ax[0, 1].set_title('Buffervessel')
        #ax[0, 1].set_xlabel(('Time (s)'))
        #ax[0, 1].set_ylabel(('Temperature (°C)'))


        plt.tight_layout()
        plt.suptitle(Path(__file__).stem)
        plt.show()

        energy = np.trapz(data[3], data[0])
        energykWh = energy/3600000
        energyGasM3 = energykWh/10.2

        electrical_energy_compressor = np.trapz(data[9], data[0])
        energykWh_compressor = electrical_energy_compressor / 3600000

        energy_buffervessel = np.trapz(data[7], data[0])
        energykWh_buffervessel = energy_buffervessel / 3600000
        energyGasM3_buffervessel = energykWh_buffervessel / 10.2

        COP = energy_buffervessel/electrical_energy_compressor

        print(f'COP: {COP}')
        print(f'Elektrische energie: {energykWh_compressor}')
        print(f'Thermische energie: {energykWh_buffervessel}')


        print(f'Totaal aan gas voor appartementencomplex in [m3]: {energyGasM3_buffervessel}')
        print(f'Totaal aan gas voor per woning voor verwarming in [m3]: {energyGasM3/48}')
        print(f'Totaal aan gas voor per woning in [m3]: {energyGasM3_buffervessel/48}')
        print(f'Totaal aan gas voor per woning voor tapwater in [m3]: {energyGasM3_buffervessel / 48 - energyGasM3/48}')

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

        #df_out['Tradiator'] = data[3].tolist()
        df_out["Heating"] = data[3].tolist()

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
    main(show=True, xl=False)  # temporary solution, recommended syntax
