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
from housemodel.sourcesink.heatpumps.Heatpump_HM import Heatpump_NTA
from housemodel.controls.heating_curves import hyst, outdoor_reset
from housemodel.sourcesink.heatpumps.NTA8800_Q.HPQ9 import calc_WP_general
import housemodel.tools.ReturnTemperature as Tr

import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path

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

    dTdt = (-tot_sys.k_mat @ x.T) - (tot_sys.f_mat @ x.T) + local_q_vector
    dTdt = np.dot(tot_sys.c_inv_mat, dTdt)

    return dTdt.flatten().tolist()


def main(show=False, xl=False):
    # read configuration file into dictionary
    param = load_config(str(CONFIGDIR / "for_heat_pump_NTA8800_with_buffervessel_nodes_edges.yaml"))
    # convert duration in hours to days and round up with ceil function
    days_sim = math.ceil(param['timing']['Duration'] / 24)
    print(days_sim)

    # create House object
    h = Building("MyHouse")
    section = param["Building"]
    # read nodes attribute from dictionary and create capacity matrix
    h.nodes_from_dict(section["nodes"])
    h.fill_c_inv()
    h.edges_from_dict(section["edges"])
    # read FixedNode objects (external nodes);
    h.boundaries_from_dict(param["boundaries"])  # function selects "outdoor" as ambient
    h.make_k_ext_and_add_ambient()  # initialize k_ext_mat and add diagonal elements
    logger.info(f" \n\n C^-1: \n {h.c_inv_mat} \n K_ext: \n {h.k_ext_mat}, \n q_vec: \n {h.q_vec} \n")

    b = StratifiedBufferNew.from_dict(param["Buffer"])
    # inputs_buffervessel = (U=0.12, As=0.196, Aq=0.196, Tamb=10,
    # Tsupply=80, Treturn=Tr_GMTD,
    # cpwater=4190, lamb=0.644, mdots=mdots, mdotd=mdotd, mass_water=150/8, z=1/8)

    b.generate_nodes()
    b.fill_c_inv()
    b.generate_edges()
    b.generate_ambient()
    b.make_k_ext_and_add_ambient()

    total = TotalSystem("HouseWithBuffervessel", [h, b])
    total.sort_parts()
    # compose c-1-matrix from parts and merge tag_lists
    total.merge_c_inv()
    total.merge_tag_lists()

    # compose k-matrix from edges
    total.edges_between_from_dict(param["edges"])
    total.merge_edge_lists_from_parts_and_between()

    total.fill_k(total.edge_list)
    total.merge_k_ext()
    total.k_mat += total.k_ext_mat
    total.merge_ambients()  # assignment by reference, no copy!
    total.make_empty_q_vec()
    logger.info(f" \n\n {total.c_inv_mat} \n\n {total.k_mat}, \n\n {total.q_vec} \n")

    # calculate flow matrices and combine into f_mat_all
    if total.flows:
        total.flows = []
    for n in range(len(param['flows'])):
        total.flows.append(Flow.from_dict(param['flows'][n]))
        # total.flows[n].flow_from_dict(param['flows'][n])
        total.flows[n].make_df_matrix(rank=total.k_mat.shape[0])

    """
    # combine F-matrices into matrix total.f_mat
    total.f_mat = np.zeros_like(total.flows[0].df_mat)
    for n in range(len(total.flows)):
        total.f_mat += np.multiply(total.flows[n].df_mat, total.flows[n].heat_rate)
    print(total.f_mat, "\n")

    # remove matrix elements > 0 from Fall
    total.f_mat = np.where(total.f_mat <= 0, total.f_mat, 0)
    print(total.f_mat, "\n")

    # create diagonal elements in Fall, so that som over each row is zero
    row_sums = np.sum(total.f_mat, axis=1).tolist()
    total.f_mat = total.f_mat - np.diag(np.array(row_sums), k=0)
    print(total.f_mat, "\n")
    """

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
    Qsolar.values = Qsolar.values[0:days_sim * 24]

    Qint = PowerSource("Q_internal")
    Qint.connected_to = param['internal']['distribution']
    Qint.values = internal_heat_gain(param['internal']['Q_day'],
                                     param['internal']['delta_Q'],
                                     param['internal']['t1'],
                                     param['internal']['t2'])
    Qint.values = Qint.values.flatten()
    Qint.values = Qint.values[0:days_sim * 24]

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
    Treturn = np.ones(len(t)) * y0[1]
    Power = np.ones(len(t)) * y0[1]

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
    nta = Heatpump_NTA()
    nta.Pmax = 8
    nta.set_cal_val([4.0, 3.0, 2.5], [6.0, 2.0, 3.0])

    nta.c_coeff = calc_WP_general(nta.cal_T_evap, nta.cal_T_cond,
                                  nta.cal_COP_val, order=1)

    nta.p_coeff = calc_WP_general(nta.cal_T_evap, nta.cal_T_cond,
                                  nta.cal_Pmax_val, order=1)
    water_temp = np.zeros(len(t))
    cop_hp = np.zeros(len(t))

    # define hysteresis object for heat pump
    hp_hyst = hyst(dead_band=0.5, state=True)

    # inputs = (cap_mat_inv, cond_mat, q_vector, control_interval)
    inputs = (total, Q_vectors, control_interval)

    # Note: the algorithm can take an initial step
    # larger than the time between two elements of the "t" array
    # this leads to an "index-out-of-range" error in the last evaluation
    # of the model function, where e.g SP_T[8760] is called.
    # Therefore, set "first_step" equal or smaller than the spacing of "t".
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
        water_temp[i] = outdoor_reset(Toutdoor.values[i], 0.7, 20)
        cop_hp[i], p_hp = nta.update(Toutdoor.values[i], water_temp[i])

        # incorporate hysteresis to control
        p_hp = hp_hyst.update(Tair[i], SP.values[i], p_hp)
        if Qinst < 2500:
            Qinst = 0

        Tr_GMTD = Tr.Tr_GMTD(Qinst / 1000, 80, 20, 5, 80, 60, 20, 1.33)

        # update q_vector
        air_node = total.find_tag_from_node_label("air")
        Q_vectors[air_node, i] += Qinst

        toplevel = TBuffervessel0[i]
        total.flows[1].set_flow_rate(np.clip((80 - toplevel) * 1.0e-6, 0, 50.0e-6))
        total.flows[0].set_flow_rate(np.clip(Qinst / ((toplevel - Tr_GMTD) * total.flows[0].cp), 0, 50.0e-6))

        # mdots = np.clip((80 - toplevel) * 0.001, 0, 0.05)
        # mdotd = np.clip(Qinst / ((TBuffervessel0[i] - Tr_GMTD) * 4180), 0, 0.05)

        # combine F-matrices into matrix total.f_mat
        total.f_mat = np.zeros_like(total.flows[0].df_mat)
        for n in range(len(total.flows)):
            total.f_mat += np.multiply(total.flows[n].df_mat, total.flows[n].heat_rate)

        # remove matrix elements > 0 from total.f_mat
        total.f_mat = np.where(total.f_mat <= 0, total.f_mat, 0)

        # create diagonal elements in total.f_mat, so that som over each row is zero
        row_sums = np.sum(total.f_mat, axis=1).tolist()
        total.f_mat = total.f_mat - np.diag(np.array(row_sums), k=0)

        ts = [t[i], t[i + 1]]
        result = solve_ivp(model_radiator_ne, ts, y0,
                           method='RK45', args=inputs,
                           first_step=control_interval)

        Tair[i + 1] = result.y[0, -1]
        Twall[i + 1] = result.y[1, -1]
        Treturn[i] = Tr_GMTD
        Power[i] = Qinst
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
        data = (t, Tair, Twall, Treturn, Power, water_temp, cop_hp,
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

        ax[1, 1].plot(time_d, data[3], label='Return temp', color='b')
        ax[1, 1].legend(loc='upper right')
        ax[1, 1].set_title('Return Temperature')
        ax[1, 1].set_xlabel(('Time (s)'))
        ax[1, 1].set_ylabel(('Temperature (°C)'))

        ax[2, 1].plot(time_d, data[7], label='Top')
        # ax[2, 1].plot(time_d, data[8], label='T1')
        ax[2, 1].plot(time_d, data[9], label='T2')
        # ax[2, 1].plot(time_d, data[10], label='T3')
        ax[2, 1].plot(time_d, data[11], label='T4')
        # ax[2, 1].plot(time_d, data[12], label='T5')
        # ax[2, 1].plot(time_d, data[13], label='T6')
        ax[2, 1].plot(time_d, data[14], label='Bottom')
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
    main(show=True, xl=True)  # temporary solution, recommended syntax
