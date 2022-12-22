# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:05:19 2020

@author: TrungNguyen, PvK, MvdB
"""
from housemodel.solvers.Office_model_WHATAS import house_radiator_m  # exposed function "house" in house module
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
from scipy.interpolate import interp1d, interp2d
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
    house_param = load_config(str(CONFIGDIR / "Simulation_WHATAS_office.yaml"))
    days_sim = 365
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

    # read NEN5060 data from spreadsheet NEN5060-2018.xlsx into pandas DataFrame
    df_nen = read_nen_weather_from_xl()
    # generate and insert timezone-aware UTC and local timestamps (with DST)
    df_nen = NENdatehour2datetime(df_nen)
    df_irr = run_qsun(df_nen)
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
    #Qinternal_sim = Qint[0:days_sim*24]

    time = np.arange(0, 24*3600, 3600)
    internal_heat_daily = np.zeros(24)
    internal_heat_daily[8] = 61875/2
    internal_heat_daily[9:16] = 61875
    internal_heat_daily[16] = 61875 / (4/3)
    internal_heat_daily[17] = 61875 / (2)
    internal_heat_daily[18] = 61875 / (4)
    internal_heat = []
    internal_heat_time = []

    for i in range(days_sim):
        internal_heat_time = np.append(internal_heat_time, time + (24 * 3600 * i))
        if ((i % 6 == 0) | (i % 6 == 7)):
            internal_heat = np.append(internal_heat, np.zeros(24))
        else:
            internal_heat = np.append(internal_heat, internal_heat_daily)



    Toutdoor = df_nen.loc[:, 'temperatuur'].values
    Toutdoor = Toutdoor.flatten()   # temperature
    T_outdoor_sim = Toutdoor[0:days_sim*24]

    #SP = simple_thermostat(8, 23, 20, 18)
    #SP_sim = SP[0:days_sim * 24].flatten()

    time = np.arange(0, 24 * 3600, 3600)
    temperature_setpoint_daily = np.ones(24) * 18
    temperature_setpoint_daily[7:19] = 21
    setpoint_daily = []
    setpoint_time = []

    for i in range(days_sim):
        setpoint_time = np.append(setpoint_time, time + (24 * 3600 * i))
        if ((i % 6 == 0) | (i % 6 == 7)):
            setpoint_daily = np.append(setpoint_daily, np.ones(24) * 16)
        else:
            setpoint_daily = np.append(setpoint_daily, temperature_setpoint_daily)

    SP_sim = setpoint_daily

    # make predictable part of q_dot vector
    q_vector = np.zeros((num_links,days_sim*24))
    leak_to_amb = house_param["chains"][0]["links"][0]["Conductance"]
    q_vector[0,:] = (T_outdoor_sim * leak_to_amb) + internal_heat + CF * Qsolar_sim
    q_vector[1,:] = (1 - CF) * Qsolar_sim

    # Interpolation of data
    interp_func = interp1d(time_sim, q_vector, fill_value='extrapolate')
    interp_func_SP = interp1d(time_sim, SP_sim, fill_value='extrapolate')
    #interp_func_Q_internal = interp1d(time_sim, Qinternal_sim, fill_value='extrapolate')
    interp_func_Toutdoor = interp1d(time_sim, T_outdoor_sim, fill_value='extrapolate')
    q_vector = interp_func(np.arange(0, time_sim[-1]+(6*600), control_interval))
    SP_sim = interp_func_SP(np.arange(0, time_sim[-1]+(6*600), control_interval))
    T_outdoor_sim = interp_func_Toutdoor(np.arange(0, time_sim[-1]+(6*600), control_interval))
    #Qinternal_sim = interp_func_Q_internal(np.arange(0, time_sim[-1]+(6*600), control_interval))
    time_sim = np.arange(0, time_sim[-1]+(6*600), control_interval)

    #interpolation of radiator flow and return temperatures based on the Kantoor.xlsx file
    Power_normalized = [1, 0.5, 0.25]
    feed_temperature = [70, 65, 60, 55, 50, 45, 40, 35, 30, 25]
    flow_normalized = np.array([(1, 0.702557893, 0.396497482), (0.905899997, 0.63320233, 0.355969377), (0.811165629, 0.56257539, 0.314633404),
                                (0.714450715, 0.492885594, 0.273749801), (0.616490056, 0.421596778, 0.231844745), (0.516371149, 0.349476187, 0.191514044)
                                , (0.413793002, 0.276365433, 0.148539848), (0.307681479, 0.202125193, 0.108315729), (0.197892826, 0.126510596, 0.066595755), (0.080209624, 0.048733622, 0.0253955)])
    return_temp_normalized = np.array([(40, 27.84804791, 22.42795917), (37.79754795, 27.01347543, 22.34815261), (35.66475247, 26.19250546, 22.27173163),
                                       (33.54890815, 25.45037251, 22.20628052), (31.50308426, 24.7253933, 22.14520503), (29.50836187, 24.04759663, 22.09971643),
                                       (27.58542565, 23.42540724, 22.05539995), (25.7423552, 22.87304353, 22.02895827), (24.05942478, 22.41250118, 22.00942387), (22.58915095, 22.08359809, 22.00095818)])

    interp_func_flow = interp2d(Power_normalized, feed_temperature, flow_normalized, fill_value='0')
    interp_func_retour = interp2d(Power_normalized, feed_temperature, return_temp_normalized, fill_value='0')

    # Input PID values in to control
    control_parameters = np.zeros(4)
    control_parameters[0] = house_param['controller']['kp']
    control_parameters[1] = house_param['controller']['ki']
    control_parameters[2] = house_param['controller']['kd']
    control_parameters[3] = house_param['controller']['Maximum_power']

    # solve ODE
    data = house_radiator_m(cap_mat_inv, cond_mat, q_vector,
                            SP_sim, time_sim, control_interval, control_parameters, T_outdoor_sim, interp_func_retour, interp_func_flow)

    # if show=True, plot the results
    if show:
        fig, ax = plt.subplots(2, 2, sharex=True)
        ax[0, 0].plot(data[0],data[1], label='Tair')
        #ax[0, 0].plot(data[0],data[2], label='Twall')
        ax[0, 0].plot(data[0], SP_sim, label='SP_Temperature')
        ax[0, 0].plot(data[0], T_outdoor_sim, label='Toutdoor')
        ax[0, 0].legend(loc='upper right')
        ax[0, 0].set_title('House Temperatures')
        ax[0, 0].set_xlabel(('Time (s)'))
        ax[0, 0].set_ylabel(('Temperature (°C)'))

        ax[1, 0].plot(data[0], data[3]/1000, label='Power', color='c')
        ax[1, 0].legend(loc='upper right')
        ax[1, 0].set_title('Thermal power demand')
        ax[1, 0].set_xlabel(('Time (s)'))
        ax[1, 0].set_ylabel(('Power (kW)'))

        ax[1, 1].plot(data[0], data[7]/1000, label='Thermal power output compressor',color='b')
        ax[1, 1].plot(data[0], data[8]/1000, label='Electric power compressor', color='g')
        ax[1, 1].plot(internal_heat_time, internal_heat / 1000, label='Internal heat', color='r')

        ax[1, 1].legend(loc='upper right')
        ax[1, 1].set_title('Compressor data')
        ax[1, 1].set_xlabel(('Time (s)'))
        ax[1, 1].set_ylabel(('Power [kW]'))

        ax[0, 1].plot(data[0], data[5], label='Temperature of top level buffervessel')
        ax[0, 1].plot(data[0], data[6], label='Temperature of bottom level buffervessel')
        ax[0, 1].plot(data[0], data[9], label='Setpoint of top level buffervessel')



        plt.tight_layout()
        plt.suptitle(Path(__file__).stem)
        plt.show()

        energy = np.trapz(data[3], data[0])
        energykWh = energy/3600000
        energyGasM3 = energykWh/10.2

        electrical_energy_compressor = np.trapz(data[8], data[0])
        energykWh_compressor = electrical_energy_compressor / 3600000

        energy_buffervessel = np.trapz(data[7], data[0])
        energykWh_buffervessel = energy_buffervessel / 3600000
        energyGasM3_buffervessel = energykWh_buffervessel / 10.2

        COP = energy_buffervessel/electrical_energy_compressor

        print(f'COP: {COP}')
        print(f'Elektrische energie: {energykWh_compressor}')
        print(f'Thermische energie: {energykWh_buffervessel}')
        print(f'Totaal aan gas voor kantoorcomplex in [m3]: {energyGasM3_buffervessel}')

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
