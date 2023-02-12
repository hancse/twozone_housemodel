# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:05:19 2020

@author: PvK, MvdB
"""

import numpy as np
import math
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from tqdm import tqdm, trange

from housemodel.solvers.house_model_heat_pump_NTA8800_with_buffervessel import house_radiator_m  # exposed function "house" in house module
# function "model" in module house is private

from housemodel.tools.new_configurator import load_config

from housemodel.sourcesink.NEN5060 import run_qsun
from housemodel.sourcesink.internal_heat_gain import internal_heat_gain
from housemodel.controls.Temperature_SP import simple_thermostat
from housemodel.weather_solar.weatherdata import (read_nen_weather_from_xl,
                                                  NENdatehour2datetime)

from housemodel.buildings.building import Building
from housemodel.sourcesink.radiators.linear_radiator import LinearRadiator
from housemodel.basics.powersource import PowerSource
from housemodel.basics.totalsystem import TotalSystem

import matplotlib
import matplotlib.pyplot as plt

from pathlib import  Path

import logging

logging.basicConfig()
logger = logging.getLogger('BUF')
logger.setLevel(logging.INFO)

CONFIGDIR = Path(__file__).parent.absolute()

matplotlib.use('Qt5Agg')

def model_radiator_ne(t, x,
                      tot_sys, Q_vectors, control_interval):
    """model function for scipy.integrate.solve_ivp.

    Args:
        t:                (array):   time
        x:                (float):   row vector with temperature nodes
        tot_sys:          (object):  model system
        Q_vectors:        (ndarray): set of Q_vectors for all input times
        control_interval: (int)    : in seconds

    Returns:
        (list): vector elements of dx/dt
    """
    index = int(t / control_interval)
    local_q_vector = Q_vectors[:, [index]]

    # tot_sys.add_ambient_to_q()
    # tot_sys.add_source_to_q(Q_solar, index)

    # Conversion of 1D array to a 2D array
    # https://stackoverflow.com/questions/5954603/transposing-a-1d-numpy-array
    x = np.array(x)[np.newaxis]

    dTdt = (-tot_sys.k_mat @ x.T) + local_q_vector
    dTdt = np.dot(tot_sys.c_inv_mat, dTdt)

    return dTdt.flatten().tolist()


def main(show=False, xl=False):
    # read configuration file into dictionary
    param = load_config(str(CONFIGDIR / "Simulation_heat_pump_NTA8800_with_buffervessel.yaml"))
    # convert duration in hours to days and round up with ceil function
    days_sim = math.ceil(param['timing']['Duration'] / 24)
    print(days_sim)

    # create House object
    h = Building("MyHouse")
    section = param["Building"]
    # read nodes attribute from dictionary and create capacity matrix
    h.nodes_from_dict(section["nodes"])
    h.fill_c_inv()
    # read FixedNode objects (external nodes);
    h.boundaries_from_dict(param["boundaries"])  # function selects "outdoor" as ambient
    h.make_k_ext_and_add_ambient()  # initialize k_ext_mat and add diagonal elements
    logger.info(f" \n\n C^-1: \n {h.c_inv_mat} \n K_ext: \n {h.k_ext_mat}, \n q_vec: \n {h.q_vec} \n")

    #Loading the radiator and buffervessel parameters
    #Heat transfer coefficient of the radiator and het capacity
    #UAradiator = house_param["chains"][0]["links"][2]["Conductance"]
    #Crad =  house_param["chains"][0]["links"][2]["Capacity"]

    # read NEN5060 data from spreadsheet NEN5060-2018.xlsx into pandas DataFrame
    df_nen = read_nen_weather_from_xl()
    # generate and insert timezone-aware UTC and local timestamps (with DST)
    df_nen = NENdatehour2datetime(df_nen)

    df_irr = run_qsun(df_nen)
    print(df_irr.head())

    time_sim = df_irr.iloc[0:days_sim*24, 0].values

    # Interval in seconds the control algorithm
    # Timestep is in minutes
    control_interval = param["timing"]["Timestep"] * 60

    Qsolar = PowerSource("Solar")
    Qsolar.connected_to = param['solar_irradiation']['distribution']
    Qsolar.values = (df_irr.total_E * param['solar_irradiation']['E'] +
                     df_irr.total_SE * param['solar_irradiation']['SE'] +
                     df_irr.total_S * param['solar_irradiation']['S'] +
                     df_irr.total_SW * param['solar_irradiation']['SW'] +
                     df_irr.total_W * param['solar_irradiation']['W'] +
                     df_irr.total_NW * param['solar_irradiation']['NW'] +
                     df_irr.total_N * param['solar_irradiation']['N'] +
                     df_irr.total_NE * param['solar_irradiation']['NE']).values
    Qsolar.values *= param['solar_irradiation']['g_value']
    Qsolar.values = Qsolar.values[0:days_sim*24]

    Qint = PowerSource("Q_internal")
    Qint.connected_to = param['internal']['distribution']
    Qint.values = internal_heat_gain(param['internal']['Q_day'],
                                     param['internal']['delta_Q'],
                                     param['internal']['t1'],
                                     param['internal']['t2'])
    Qint.values = Qint.values.flatten()
    Qint.values = Qint.values[0:days_sim*24]

    Toutdoor = PowerSource("T_outdoor")
    Toutdoor.values = df_nen.loc[:, 'temperatuur'].values
    Toutdoor.values = Toutdoor.values.flatten()
    Toutdoor.values = Toutdoor.values[0:days_sim * 24]

    SP = PowerSource("SetPoint")
    SP.values = simple_thermostat(8, 23, 20, 17)
    SP.values = SP.values[0:days_sim * 24].flatten()

    source_list = [Qsolar, Qint]
    Q_vectors = np.zeros((total.num_nodes, days_sim * 24))
    for n in range(days_sim * 24):
        total.parts[0].ambient.update(Toutdoor.values[n])
        total.add_ambient_to_q()
        for s in source_list:
            total.add_source_to_q(s, n)
        # logging.info(f" q_vector: \n {total.q_vec}")
        Q_vectors[:, [n]] = total.q_vec
        total.make_empty_q_vec()

    interp_func = interp1d(time_sim, Q_vectors, fill_value='extrapolate')
    Q_vectors = interp_func(np.arange(0, time_sim[-1] + (6 * 600), control_interval))

    # Interpolation of data
    Qint.interpolate_power(time_sim, control_interval)
    Toutdoor.interpolate_power(time_sim, control_interval)
    SP.interpolate_power(time_sim, control_interval)

    glob = PowerSource("Global")
    glob.values = df_nen['globale_zonnestraling'].values
    glob.values = glob.values[0:days_sim * 24].flatten()
    glob.interpolate_power(time_sim, control_interval)

    cloud = PowerSource('cloud')
    cloud.values = df_nen['bewolkingsgraad'].values
    cloud.values = cloud.values[0:days_sim * 24].flatten()
    cloud.interpolate_power(time_sim, control_interval)

    # interpolate time_sim itself (after all arrays are interpolated)
    time_sim = np.arange(0, time_sim[-1] + (6 * 600), control_interval)
    # time_sim = np.linspace(0, time_sim[-1], (8760-1)*6, endpoint=False)

    # Input PID values in to control
    controllers = []
    for n in range(len(param['controllers'])):
        c = param['controllers'][n]
        controllers.append(c)

    # solve ODE
    """
    data = house_radiator_m(cap_mat_inv, cond_mat, q_vector,
                            SP_sim, time_sim, control_interval, control_parameters, T_outdoor_sim)
    """

    """
    house model base on 2R2C model with a buffervessel and a radiator
    """

    from housemodel.controls.ivPID.PID import PID
    from housemodel.sourcesink.heatpumps.Heatpump_HM import Heatpump_NTA
    from housemodel.controls.heating_curves import hyst, outdoor_reset
    from housemodel.sourcesink.heatpumps.NTA8800_Q.HPQ9 import calc_WP_general
    import housemodel.tools.ReturnTemperature as Tr

    q_o = 1  # [kW] Desing (peak) heat demand rate
    Ts_o = 50  # [°C] Design radiator supply temperature
    Tr_o = 30  # [°C] Design radiator return temperature
    Ti_o = 20  # [°C] Design indoor (set) temperature
    n = 1.3  # [-]  Emprical radiator constant


    def model_radiator_m(t, x, cap_mat_inv, cond_mat, q_vector,
                         control_interval):
        """model function for scipy.integrate.odeint.

        Args:
            t:           (array):   variable array dependent on time with the vairable Air temperature, Wall temperature Radiator
            x:           (float):
            cap_mat_inv: (float):  diagonal heat capacity matrix
            cond_mat:    (float):  symmetric conductance matrix
            q_vector:    (float):  external heat sources and sinks
            SP_T:        (float): thermostat setpoints

        Returns:
            (list): vector elements of dx/dt
        """
        # States :
        # Tair = x[0]

        # Parameters :
        index = int(t / control_interval)

        # Equations :
        local_q_vector = np.zeros((2, 1))
        local_q_vector[0, 0] = q_vector[0, index]
        local_q_vector[1, 0] = q_vector[1, index]
        # local_q_vector[2,0] = q_vector[2,index]

        # Conversion of 1D array to a 2D array
        # https://stackoverflow.com/questions/5954603/transposing-a-1d-numpy-array
        x = np.array(x)[np.newaxis]

        dTdt = (-cond_mat @ x.T) + local_q_vector
        dTdt = np.dot(cap_mat_inv, dTdt)

        return dTdt.flatten().tolist()

    def model_stratified_buffervessel(t, x, U, As, Aq, Tamb, Tsupply, Treturn, cpwater, lamb, mdots, mdotd, mass_water,
                                      z):
        """model function for scipy.integrate.odeint.

        :param x:            (array):   variable array dependent on time with the vairable Air temperature, Wall temperature Return water temperature and buffervessel temperature
        :param t:            (float):
        :param Pin:          (float):  Power input in [W]
        :param U:            (float):
        :param A:            (float):  Area of
        :param T_amb:        (float):
        :param rho:          (float):
        :param volume:       (float):
        :param cp: (float):  Thermal resistance from indoor air to outdoor air [K/W]

        x,t: ode input function func : callable(x, t, ...) or callable(t, x, ...)
        Computes the derivative of y at t.
        If the signature is ``callable(t, y, ...)``, then the argument tfirst` must be set ``True``.
        """

        # Water supply
        mdote = mdots - mdotd

        if mdote > 0:
            deltaPlus = 1
        else:
            deltaPlus = 0

        if mdote < 0:
            deltaMinus = 1
        else:
            deltaMinus = 0

        dT1 = ((mdots * cpwater * (Tsupply - x[0])) + (mdote * cpwater * (x[0] - x[1]) * deltaMinus) - (
                    U * As * (x[0] - Tamb)) + ((Aq * lamb) / z) * (x[0] - x[1])) / (mass_water * cpwater)
        dT2 = ((mdote * cpwater * (x[0] - x[1]) * deltaPlus) + (mdote * cpwater * (x[1] - x[2]) * deltaMinus) - (
                    U * As * (x[1] - Tamb)) + ((Aq * lamb) / z) * (x[0] + x[2] - (2 * x[1]))) / (mass_water * cpwater)
        dT3 = ((mdote * cpwater * (x[1] - x[2]) * deltaPlus) + (mdote * cpwater * (x[2] - x[3]) * deltaMinus) - (
                    U * As * (x[2] - Tamb)) + ((Aq * lamb) / z) * (x[1] + x[3] - (2 * x[2]))) / (mass_water * cpwater)
        dT4 = ((mdote * cpwater * (x[2] - x[3]) * deltaPlus) + (mdote * cpwater * (x[3] - x[4]) * deltaMinus) - (
                    U * As * (x[3] - Tamb)) + ((Aq * lamb) / z) * (x[2] + x[4] - (2 * x[3]))) / (mass_water * cpwater)
        dT5 = ((mdote * cpwater * (x[3] - x[4]) * deltaPlus) + (mdote * cpwater * (x[4] - x[5]) * deltaMinus) - (
                    U * As * (x[4] - Tamb)) + ((Aq * lamb) / z) * (x[3] + x[5] - (2 * x[4]))) / (mass_water * cpwater)
        dT6 = ((mdote * cpwater * (x[4] - x[5]) * deltaPlus) + (mdote * cpwater * (x[5] - x[6]) * deltaMinus) - (
                    U * As * (x[5] - Tamb)) + ((Aq * lamb) / z) * (x[4] + x[6] - (2 * x[5]))) / (mass_water * cpwater)
        dT7 = ((mdote * cpwater * (x[5] - x[6]) * deltaPlus) + (mdote * cpwater * (x[6] - x[7]) * deltaMinus) - (
                    U * As * (x[6] - Tamb)) + ((Aq * lamb) / z) * (x[5] + x[7] - (2 * x[6]))) / (mass_water * cpwater)
        dT8 = ((mdotd * cpwater * (Treturn - x[7])) + (mdote * cpwater * (x[6] - x[7]) * deltaPlus) - (
                    U * As * (x[7] - Tamb)) + ((Aq * lamb) / z) * (x[6] - x[7])) / (mass_water * cpwater)

        return [dT1, dT2, dT3, dT4, dT5, dT6, dT7, dT8]

    def house_radiator_m(cap_mat_inv, cond_mat, q_vector,
                         SP_T, time_sim, control_interval, control_parameters, T_outdoor_sim):
        """Compute air and wall temperature inside the house.

        Args:
            cap_mat:    (float):  diagonal heat capacity matrix
            cond_mat:   (float):  symmetric conductance matrix
            q_vector:   (float):  external heat sources and sinks
            SP_T:       (array):  Setpoint temperature from thermostat.
            time_sim:   (array)  :  simulation time

        Returns:
            tuple :  (array) containing Tair, Twall, Tradiator and evaluation time:

        Note:
            - Tair (float):   air temperature inside the house in degree C.
            - Twall (float):  wall temperature inside the house in degree C

            - Qinst ?	  (array):  instant heat from heat source such as HP or boiler [W].
        """
    # initial values for solve_ivp
    # make a list of all nodes in total_system
    yn = [n for node in [p.nodes for p in total.parts] for n in node]
    # make a list of the (initial) temperatures of all nodes
    y0 = [cn.temp for cn in yn]
    # in one statement
    # y0 = [cn.temp for cn in [n for node in [p.nodes for p in tot_sys.parts] for n in node]]

    t = time_sim  # Define Simulation time with sampling time
    Tair = np.ones(len(t)) * y0[0]
    Twall = np.ones(len(t)) * y0[1]
    Tradiator = np.ones(len(t)) * y0[2]

    TBuffervessel0 = 80
    y0buffervessel = [TBuffervessel0, TBuffervessel0, TBuffervessel0, TBuffervessel0, TBuffervessel0,
                      TBuffervessel0, TBuffervessel0, TBuffervessel0]

    Treturn = np.ones(len(t)) * Twall0
    Power = np.ones(len(t)) * Twall0
    TBuffervessel1 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel2 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel3 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel4 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel5 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel6 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel7 = np.ones(len(t)) * TBuffervessel0
    TBuffervessel8 = np.ones(len(t)) * TBuffervessel0

    # Controller initialization
    # heatingPID = PID(Kp=5000, Ki=0, Kd=0, beta=1, MVrange=(0, 12000), DirectAction=False)
    # heating = 0
    # kp = control_parameters[0]
    # ki = control_parameters[1]
    # kd = control_parameters[2]

    pid = PID(controllers[0]['kp'],
                  controllers[0]['ki'],
                  controllers[0]['kd'],
                  t[0])

    pid.SetPoint = 17.0
    pid.setSampleTime(0)
    pid.setBounds(0, controllers[0]["maximum"])
    pid.setWindup(controllers[0]["maximum"] / control_interval)

    # Heat pump initialization
    nta = Heatpump_NTA()
    nta.Pmax = 8
    nta.set_cal_val([4.0, 3.0, 2.5], [6.0, 2.0, 3.0])

    nta.c_coeff = calc_WP_general(nta.cal_T_evap, nta.cal_T_cond,
                                      nta.cal_COP_val, order=1)

    nta.p_coeff = calc_WP_general(nta.cal_T_evap, nta.cal_T_cond,
                                      nta.cal_Pmax_val, order=1)

    water_temp = np.zeros_like(T_outdoor_sim)
    cop_hp = np.zeros_like(T_outdoor_sim)

    # define hysteresis object for heat pump
    hp_hyst = hyst(dead_band=0.5, state=True)

    # inputs = (cap_mat_inv, cond_mat, q_vector, control_interval)
    inputs = (total, Q_vectors, control_interval)

        # Note: the algorithm can take an initial step
        # larger than the time between two elements of the "t" array
        # this leads to an "index-out-of-range" error in the last evaluation
        # of the model function, where e.g SP_T[8760] is called.
        # Therefore set "first_step" equal or smaller than the spacing of "t".
        # https://github.com/scipy/scipy/issues/9198

        for i in tqdm(range(len(t) - 1)):
            # here comes the "arduino style" controller
            pid.SetPoint = SP.values[i]
            pid.update(Tair[i], t[i])
            Qinst = pid.output

            # Simple PID controller
            # Qinst = (SP_T[i] - Tair[i]) * 5000
            # Qinst = np.clip(Qinst, 0, 5000)
            # q_vector[2, i] = Qinst

            # Velocity PID controller (not working properly)
            # heating  = heatingPID.update(t[i], SP_T[i], Tair[i], heating)
            # print(f"{heating}")
            # heating  = heatingPID.update(t[i], SP_T[i], Tair[i], heating)
            # print(f"{heating}")
            # q_vector[2, i] = heating

            # Heat pump NTA800
            # p_hp = 0
            # determine new setting for COP and heat pump power
            water_temp[i] = outdoor_reset(T_outdoor_sim[i], 0.7, 20)
            cop_hp[i], p_hp = nta.update(T_outdoor_sim[i], water_temp[i])

            # incorporate hysteresis to control
            p_hp = hp_hyst.update(Tair[i], SP_T[i], p_hp)
            if Qinst < 2500:
                Qinst = 0

            Tr_GMTD = Tr.Tr_GMTD(Qinst / 1000, 80, 20, 5, 80, 60, 20, 1.33)

            # update q_vector
            q_vector[0, i] = q_vector[0, i] + Qinst

            ts = [t[i], t[i + 1]]
            result = solve_ivp(model_radiator_m, ts, y0,
                               method='RK45', args=inputs,
                               first_step=control_interval)

            toplevel = TBuffervessel1[i]
            mdots = np.clip((80 - toplevel) * 0.001, 0, 0.05)
            mdotd = np.clip(Qinst / ((TBuffervessel1[i] - Tr_GMTD) * 4180), 0, 0.05)
            inputs_buffervessel = (0.12, 0.196, 0.196, 10, 80, Tr_GMTD, 4190, 0.644, mdots, mdotd, 150 / 8, 1 / 8)
            result_buffervessel = solve_ivp(model_stratified_buffervessel, ts, y0buffervessel, method='RK45',
                                            args=inputs_buffervessel,
                                            first_step=control_interval)

            Tair[i + 1] = result.y[0, -1]
            Twall[i + 1] = result.y[1, -1]
            Treturn[i] = Tr_GMTD
            Power[i] = Qinst
            TBuffervessel1[i + 1] = result_buffervessel.y[0, -1]
            TBuffervessel2[i + 1] = result_buffervessel.y[1, -1]
            TBuffervessel3[i + 1] = result_buffervessel.y[2, -1]
            TBuffervessel4[i + 1] = result_buffervessel.y[3, -1]
            TBuffervessel5[i + 1] = result_buffervessel.y[4, -1]
            TBuffervessel6[i + 1] = result_buffervessel.y[5, -1]
            TBuffervessel7[i + 1] = result_buffervessel.y[6, -1]
            TBuffervessel8[i + 1] = result_buffervessel.y[7, -1]

            y0 = result.y[:, -1]
            y0buffervessel = result_buffervessel.y[:, -1]

        # return t, Tair, Twall, Treturn, Power, water_temp, cop_hp, TBuffervessel1, TBuffervessel2, TBuffervessel3, TBuffervessel4, TBuffervessel5, TBuffervessel6, TBuffervessel7, TBuffervessel8
        data = (t, Tair, Twall, Treturn, Power, water_temp, cop_hp,
                TBuffervessel1, TBuffervessel2, TBuffervessel3, TBuffervessel4,
                TBuffervessel5, TBuffervessel6, TBuffervessel7, TBuffervessel8)

    # if show=True, plot the results
    if show:
        """
        plt.figure(figsize=(15, 5))         # key-value pair: no spaces
        plt.plot(data[0],data[1], label='Tair')
        plt.plot(data[0],data[2], label='Twall')
        plt.plot(data[0],data[3], label='Tradiator')
        plt.plot(time_sim, SP_sim, label='SP_Temperature')
        plt.plot(time_sim,T_outdoor_sim,label='Toutdoor')
        plt.plot(data[0], data[4], label='Qinst')
        #plt.plot(data[0], data[5], label='Water temp')
        #plt.plot(data[0], data[6], label='COP')
        plt.legend(loc='best')
        plt.title(Path(__file__).stem)
        plt.show()
        """
        time_d = data[0] / (3600 * 24)
        fig, ax = plt.subplots(3, 2, sharex='all')
        ax[0, 0].plot(time_d, data[1], label='Tair')
        ax[0, 0].plot(time_d, data[2], label='Twall')
        ax[0, 0].plot(time_sim / (3600 * 24), SP.values, label='SP_Temperature')
        ax[0, 0].plot(time_sim / (3600 * 24), Toutdoor.values, label='Toutdoor')
        ax[0, 0].legend(loc='upper right')
        ax[0, 0].set_title('Nodal Temperatures')
        ax[0, 0].set_xlabel(('Time (s)'))
        ax[0, 0].set_ylabel(('Temperature (°C)'))

        ax[0, 1].plot(time_d, data[6], label='COP', color='r')
        ax[0, 1].legend(loc='upper right')
        ax[0, 1].set_title('COP')
        ax[0, 1].set_xlabel(('Time (s)'))
        ax[0, 1].set_ylabel(('COP'))

        ax[1, 0].plot(time_d, data[4], label='Power', color='c')
        ax[1, 0].legend(loc='upper right')
        ax[1, 0].set_title('Power')
        ax[1, 0].set_xlabel(('Time (s)'))
        ax[1, 0].set_ylabel(('Power (kW)'))

        ax[1, 1].plot(time_d, data[3], label='Return temp',color='b')
        ax[1, 1].legend(loc='upper right')
        ax[1, 1].set_title('Return Temperature')
        ax[1, 1].set_xlabel(('Time (s)'))
        ax[1, 1].set_ylabel(('Temperature (°C)'))

        ax[2, 1].plot(time_d, data[7], label='Top')
        ax[2, 1].plot(time_d, data[8], label='T2')
        ax[2, 1].plot(time_d, data[9], label='T3')
        ax[2, 1].plot(time_d, data[10], label='T4')
        ax[2, 1].plot(time_d, data[11], label='T5')
        ax[2, 1].plot(time_d, data[12], label='T6')
        ax[2, 1].plot(time_d, data[13], label='T7')
        ax[2, 1].plot(time_d, data[14], label='Bottom')
        ax[2, 1].legend(loc='upper right')
        ax[2, 1].set_title('Buffervessel')
        ax[2, 1].set_xlabel(('Time (s)'))
        ax[2, 1].set_ylabel(('Temperature (°C)'))

        plt.tight_layout()
        plt.suptitle(Path(__file__).stem)
        plt.show()

    if xl:
        # df_out = pd.DataFrame(data[0], columns=['Timestep'])
        df_out = pd.DataFrame({'Timestep': data[0]})
        df_out['Outdoor temperature'] = T_outdoor_sim
        for n in range(num_links):
            nodename = param['chains'][0]['links'][n]['Name']
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
                   param['chains'][0]['links'][0]['Name'], None, None, None,
                   param['chains'][0]['links'][1]['Name']])
        for r in dataframe_to_rows(df_out, index=False):
            ws.append(r)
        # df_out.to_excel('tst.xlsx', index=False, startrow=10)
        wb.save('tst.xlsx')


if __name__ == "__main__":
    main(show=True, xl=True)  # temporary solution, recommended syntax
