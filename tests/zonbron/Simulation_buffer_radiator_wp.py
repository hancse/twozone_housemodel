# -*- coding: utf-8 -*-
"""
@author: PvK, MvdB
"""

import numpy as np
import math
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from tqdm import tqdm

from housemodel.tools.new_configurator import load_config

from housemodel.sourcesink.NEN5060 import run_qsun
from housemodel.sourcesink.internal_heat_gain import internal_heat_gain
from housemodel.controls.Temperature_SP import simple_thermostat
from housemodel.weather_solar.weatherdata import (read_nen_weather_from_xl,
                                                  NENdatehour2datetime)

from housemodel.buildings.building import Building
from housemodel.sourcesink.buffervessels.stratified import StratifiedBufferNew
from housemodel.basics.powersource import PowerSource
from housemodel.basics.flows import Flow
from housemodel.basics.totalsystem import TotalSystem

from housemodel.controls.ivPID.PID import PID
from housemodel.sourcesink.heatpumps.heatpumpnew import HeatpumpNTANew
from housemodel.controls.heating_curves import hyst, outdoor_reset
from housemodel.sourcesink.heatpumps.NTA8800_Q.HPQ9 import calc_WP_general

import housemodel.tools.radiator_performance.ReturnTemperature as Tr
# import housemodel.tools.ReturnTemperature as Tr
from housemodel.sourcesink.radiators.radiators import Radiator

import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path

import logging

logging.basicConfig()
logger = logging.getLogger('ZB')
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

CONFIGDIR = Path(__file__).parent.absolute()

matplotlib.use('Qt5Agg')


def model_radiator_ne(t, x, tot_sys, control_interval):
    """model function for scipy.integrate.solve_ivp.

    Args:
        t:                (array):   time
        x:                (float):   row vector with temperature nodes
        tot_sys:          (object):  model system, including q_vec
        control_interval: (int)    : in seconds

    Returns:
        (list): vector elements of dx/dt
    """
    # Conversion of 1D array to a 2D array
    # https://stackoverflow.com/questions/5954603/transposing-a-1d-numpy-array
    x = np.array(x)[np.newaxis]

    dTdt = (-tot_sys.k_mat @ x.T) - (tot_sys.f_mat @ x.T) + tot_sys.q_vec
    dTdt = np.dot(tot_sys.c_inv_mat, dTdt)

    return dTdt.flatten().tolist()


def main(show=False, xl=False):
    # read configuration file into dictionary
    param = load_config(str(CONFIGDIR / "for_buffer_radiator_wp.yaml"))
    # convert duration in hours to days and round up with ceil function
    days_sim = math.ceil(param['timing']['Duration'] / 24)
    print(days_sim)

   # create Building object and read nodes, internal edges and boundaries
    h = Building.from_yaml(str(CONFIGDIR / "for_buffer_radiator_wp.yaml"))
    # make C_inv matrix and K_ext matrix, print result
    h.fill_c_inv()
    h.make_k_ext_and_add_ambient()
    logger.info(f" \n\n C^-1: \n {h.c_inv_mat} \n\n "
                f"K_ext: \n {h.k_ext_mat}, \n\n "
                f"q_vec: \n {h.q_vec} \n")

    b = StratifiedBufferNew.from_yaml(str(CONFIGDIR / "for_buffer_radiator_wp.yaml"))
    b.fill_c_inv()
    b.make_k_ext_and_add_ambient()

    total = TotalSystem("HouseBufferRadWP", [h, b])
    total.sort_parts()
    # compose c-1-matrix from parts and merge tag_lists
    total.merge_c_inv()
    total.merge_tag_lists()

    # compose k-matrix from edges
    total.edges_between_from_yaml(str(CONFIGDIR / "for_buffer_radiator_wp.yaml"))
    total.merge_edge_lists_from_parts_and_between()

    total.fill_k(total.edge_list)
    total.merge_k_ext()
    total.k_mat += total.k_ext_mat
    total.merge_ambients()  # assignment by reference, no copy!
    total.make_empty_q_vec()
    logger.info(f" \n\n {total.c_inv_mat} \n\n {total.k_mat}, \n\n {total.q_vec} \n")

    # calculate flow matrices
    total.flows_from_yaml(str(CONFIGDIR / "for_buffer_radiator_wp.yaml"))

    # read NEN5060 data from spreadsheet NEN5060-2018.xlsx into pandas DataFrame
    df_nen = read_nen_weather_from_xl()
    # generate and insert timezone-aware UTC and local timestamps (with DST)
    df_nen = NENdatehour2datetime(df_nen)

    df_irr = run_qsun(df_nen)
    print(df_irr.head())

    time_sim = df_irr.iloc[0:days_sim * 24, 0].values

    # Interval in seconds the control algorithm
    # Timestep is in minutes
    control_interval = param["timing"]["Timestep"] * 60

    # skip additional source terms from solar irradiation and human presence
    Toutdoor = PowerSource("T_outdoor")
    Toutdoor.values = df_nen.loc[:, 'temperatuur'].values
    Toutdoor.values = Toutdoor.values.flatten()
    Toutdoor.values = Toutdoor.values[0:days_sim * 24]

    SP = PowerSource("SetPoint")
    SP.values = simple_thermostat(8, 23, 20, 17)
    SP.values = SP.values[0:days_sim * 24].flatten()

    # Interpolation of data
    Toutdoor.interpolate_power(time_sim, control_interval)
    SP.interpolate_power(time_sim, control_interval)

    # interpolate time_sim itself (after all arrays are interpolated)
    time_sim = np.arange(0, time_sim[-1] + (6 * 600), control_interval)
    # time_sim = np.linspace(0, time_sim[-1], (8760-1)*6, endpoint=False)

    # Input PID values in to control
    controllers = []
    for n in range(len(param['controllers'])):
        c = param['controllers'][n]
        controllers.append(c)

    # solve ODE
    # initial values for solve_ivp
    # make a list of all nodes in total_system
    yn = [n for node in [p.nodes for p in total.parts] for n in node]
    # make a list of the (initial) temperatures of all nodes
    y0 = [cn.temp for cn in yn]
    # in one statement
    # y0 = [cn.temp for cn in [n for node in [p.nodes for p in total.parts] for n in node]]

    t = time_sim  # Define Simulation time with sampling time
    Tair = np.ones(len(t)) * y0[0]
    Twall = np.ones(len(t)) * y0[1]

    Treturn = np.ones(len(t)) * y0[1]
    Power_gb = np.zeros(len(t))
    Power_hp = np.zeros(len(t))

    TBuffervessel0 = np.ones(len(t)) * y0[2]
    TBuffervessel1 = np.ones(len(t)) * y0[3]
    TBuffervessel2 = np.ones(len(t)) * y0[4]
    TBuffervessel3 = np.ones(len(t)) * y0[5]
    TBuffervessel4 = np.ones(len(t)) * y0[6]
    TBuffervessel5 = np.ones(len(t)) * y0[7]
    TBuffervessel6 = np.ones(len(t)) * y0[8]
    TBuffervessel7 = np.ones(len(t)) * y0[9]

    pid = PID(controllers[0]['kp'],
              controllers[0]['ki'],
              controllers[0]['kd'],
              t[0])

    pid.SetPoint = 17.0
    pid.setSampleTime(0)
    pid.setBounds(0, controllers[0]["maximum"])
    pid.setWindup(controllers[0]["maximum"] / control_interval)

    # Heat pump initialization
    nta = HeatpumpNTANew()
    nta.Pmax_kW = 8.0         # in kW
    nta.set_cal_val([4.0, 3.0, 2.5], [6.0, 2.0, 3.0])

    nta.c_coeff = calc_WP_general(nta.cal_T_evap, nta.cal_T_cond,
                                  nta.cal_COP_val, order=1)

    nta.p_coeff = calc_WP_general(nta.cal_T_evap, nta.cal_T_cond,
                                  nta.cal_Pmax_val, order=1)
    water_temp = np.zeros(len(t))
    cop_hp = np.zeros(len(t))

    nta.set_flow(total.flows[1])
    print(f"Heat rate: {nta.flow.heat_rate} [W/K] \n")

    # define hysteresis object for heat pump
    # hp_hyst = hyst(dead_band=0.5, state=True)

    # Radiator object
    deg = u"\u00b0"     # degree sign
    radiator = Radiator(name="Rad", exp_rad=1.3)
    radiator.T_supply = TBuffervessel0[0]
    radiator.T_amb = Tair[0]
    radiator.T_return = (radiator.T_supply + radiator.T_amb) / 2.0  # crude quess

    radiator.set_flow(total.flows[0])
    print(f"Heat rate: {radiator.flow.heat_rate} [W/K] \n")

    radiator.update(radiator.func_rad_lmtd)
    print(f"Q_dot: {radiator.q_dot}, T_return: {radiator.T_return}")
    print(f"radiator to room: {radiator.flow.heat_rate * (TBuffervessel0[0] - radiator.T_return)} [W]")
    print(f"radiator to room: {radiator.Km * np.power(radiator.get_lmtd(), radiator.exp_rad)} [W] with Delta T_LMTD = {radiator.get_lmtd()}")
    print(f"top-bottom: {radiator.flow.heat_rate * (TBuffervessel0[0] - TBuffervessel7[0])} [W]")
    print(f"back-to-bottom: {radiator.flow.heat_rate * (radiator.T_return - TBuffervessel7[0])} [W]")

    # inputs = (cap_mat_inv, cond_mat, q_vector, control_interval)
    #inputs = (total, Q_vectors, control_interval)
    inputs = (total, control_interval)

    # Note: the algorithm can take an initial step
    # larger than the time between two elements of the "t" array
    # this leads to an "index-out-of-range" error in the last evaluation
    # of the model function, where e.g SP_T[8760] is called.
    # Therefore, set "first_step" equal or smaller than the spacing of "t".
    # https://github.com/scipy/scipy/issues/9198

    source_list = []
    for i in tqdm(range(len(t) - 1)):

        # first reset and update total.q_vec
        total.make_empty_q_vec()    # reset q_vec!!!!!
        total.parts[0].ambient.update(Toutdoor.values[i])  # assuming Building is parts[0]
        total.add_ambient_to_q()
        if source_list:
            for s in source_list:
                total.add_source_to_q(s, i)
            # logging.info(f" q_vector: \n {total.q_vec}")

        # here comes the "arduino style" controller
        pid.SetPoint = SP.values[i]
        pid.update(Tair[i], t[i])
        Qinst = pid.output

        # Heat pump NTA800
        # p_hp = 0
        # determine new setting for COP and heat pump power
        nta.T_cond = outdoor_reset(Toutdoor.values[i], 1.2, 20)  # stooklijn klopt niet helemaal!
        water_temp[i] = nta.T_cond
        nta.T_evap = Toutdoor.values[i]
        nta.update()
        cop_hp[i] = nta.COP

        # update q_vector: add heat source for Buffer vessel
        upper_layer = total.find_tag_from_node_label("DictBuffer0")
        total.q_vec[upper_layer] += nta.P_HP_W

        radiator.T_supply = TBuffervessel0[i]
        radiator.T_amb = Tair[i]
        radiator.T_return = (radiator.T_supply + radiator.T_amb) / 2.0  # crude quess
        radiator.update(radiator.func_rad_lmtd)

        # update q_vector: add heat source for Building
        air_node = total.find_tag_from_node_label("air")
        # total.q_vec[air_node] += Qinst          # GasBoiler via PID controller
        total.q_vec[air_node] += radiator.q_dot   # via radiator

        # supply flow
        # total.flows[1].set_flow_rate(nta.flow.flow_rate)  # Flow object????
        # demand flow
        # total.flows[0].set_flow_rate(radiator.flow.flow_rate)
        # total.flows[0].set_flow_rate(radiator.flow_rate)

        total.combine_flows()

        ts = [t[i], t[i + 1]]
        result = solve_ivp(model_radiator_ne, ts, y0,
                           method='RK45', args=inputs,
                           first_step=control_interval)

        Tair[i + 1] = result.y[0, -1]
        Twall[i + 1] = result.y[1, -1]
        Treturn[i] = radiator.T_return
        Power_gb[i] = Qinst
        Power_hp[i] = nta.P_HP_W

        TBuffervessel0[i + 1] = result.y[2, -1]
        TBuffervessel1[i + 1] = result.y[3, -1]
        TBuffervessel2[i + 1] = result.y[4, -1]
        TBuffervessel3[i + 1] = result.y[5, -1]
        TBuffervessel4[i + 1] = result.y[6, -1]
        TBuffervessel5[i + 1] = result.y[7, -1]
        TBuffervessel6[i + 1] = result.y[8, -1]
        TBuffervessel7[i + 1] = result.y[9, -1]

        y0 = result.y[:, -1]
        # y0buffervessel = result_buffervessel.y[:, -1]  # remove

        # return t, Tair, Twall, Treturn, Power, water_temp, cop_hp, TBuffervessel1, TBuffervessel2, TBuffervessel3, TBuffervessel4, TBuffervessel5, TBuffervessel6, TBuffervessel7, TBuffervessel8
        data = (t, Tair, Twall, Treturn, Power_gb, Power_hp, water_temp, cop_hp,
                TBuffervessel0, TBuffervessel1, TBuffervessel2, TBuffervessel3,
                TBuffervessel4, TBuffervessel5, TBuffervessel6, TBuffervessel7)

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

        ax[0, 1].plot(time_d, data[7], label='COP', color='r')
        ax[0, 1].legend(loc='upper right')
        ax[0, 1].set_title('COP')
        ax[0, 1].set_xlabel(('Time (s)'))
        ax[0, 1].set_ylabel(('COP'))

        ax[1, 0].plot(time_d, data[4], label='Power_gb', color='c')
        ax[1, 0].legend(loc='upper right')
        ax[1, 0].set_title('Power_gb')
        ax[1, 0].set_xlabel(('Time (s)'))
        ax[1, 0].set_ylabel(('Power (W)'))

        ax[1, 1].plot(time_d, data[3], label='Return temp', color='b')
        ax[1, 1].legend(loc='upper right')
        ax[1, 1].set_title('Return Temperature')
        ax[1, 1].set_xlabel(('Time (s)'))
        ax[1, 1].set_ylabel(('Temperature (°C)'))

        ax[2, 0].plot(time_d, data[5], label='Power HP', color = 'r')
        ax[2, 0].legend(loc='upper right')
        ax[2, 0].set_title('Power_hp')
        ax[2, 0].set_xlabel(('Time (s)'))
        ax[2, 0].set_ylabel(('Power (W)'))

        ax[2, 1].plot(time_d, data[8], label='Top')
        # ax[2, 1].plot(time_d, data[8], label='T1')
        ax[2, 1].plot(time_d, data[10], label='T2')
        # ax[2, 1].plot(time_d, data[10], label='T3')
        ax[2, 1].plot(time_d, data[12], label='T4')
        # ax[2, 1].plot(time_d, data[12], label='T5')
        # ax[2, 1].plot(time_d, data[13], label='T6')
        ax[2, 1].plot(time_d, data[15], label='Bottom')
        ax[2, 1].legend(loc='upper right')
        ax[2, 1].set_title('Buffervessel')
        ax[2, 1].set_xlabel(('Time (s)'))
        ax[2, 1].set_ylabel(('Temperature (°C)'))

        plt.tight_layout()
        plt.suptitle(Path(__file__).stem)
        plt.show()

    if xl:
        xlname = 'tst_8800_buffer.xlsx'
        logger.info(f"writing Excel file {xlname}...")
        # df_out = pd.DataFrame(data[0], columns=['Timestep'])
        df_out = pd.DataFrame({'Timestep': data[0]})
        df_out['Outdoor temperature'] = Toutdoor.values

        for n in total.tag_list:
            lbl = total.find_node_label_from_tag(n)
            df_out["T_{}".format(lbl)] = data[n + 1].tolist()
            # df_out["Solar_{}".format(n)] = Qsolar_sim[n, :]
            if lbl == 'air':
                df_out["Internal_{}".format(lbl)] = Qint.values

        df_out['Tradiator'] = data[3].tolist()
        df_out["Heating"] = data[4].tolist()

        wb = Workbook()
        ws = wb.active
        # ws.append(['DESCRIPTION',
        #           'Resultaten HAN Dynamic Model Heat Built Environment'])
        # ws.append(['Chain number', 0])
        # ws.append(['Designation', None, '2R-2C-1-zone',
        #           None, None, None, '2R-2C-1-zone'])
        # ws.append(['Node number', None, 0, None, None, None, 1])
        # ws.append(['Designation', None,
        #           param['chains'][0]['links'][0]['Name'], None, None, None,
        #           param['chains'][0]['links'][1]['Name']])
        for r in dataframe_to_rows(df_out, index=False):
            ws.append(r)
        # df_out.to_excel('tst.xlsx', index=False, startrow=10)
        wb.save(xlname)
        logger.info(f"Excel file {xlname} written")


if __name__ == "__main__":
    main(show=True, xl=False)  # temporary solution, recommended syntax
