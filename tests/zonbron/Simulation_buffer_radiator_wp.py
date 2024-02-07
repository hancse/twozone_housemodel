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

from housemodel.sourcesink.NEN5060 import run_qsun, run_qsun_new
from housemodel.sourcesink.internal_heat_gain import internal_heat_gain
from housemodel.controls.Temperature_SP import simple_thermostat
from housemodel.weather_solar.weatherdata import (read_nen_weather_from_xl,
                                                  NENdatehour2datetime)

from housemodel.buildings.building import Building
from housemodel.sourcesink.buffervessels.stratified import StratifiedBufferNew
from housemodel.basics.source_term import SourceTerm
from housemodel.basics.flows import Flow
from housemodel.basics.totalsystem import TotalSystem

from housemodel.controls.ivPID.PID import PID
from housemodel.sourcesink.heatpumps.heatpumpnew import HeatpumpNTANew
from housemodel.controls.heating_curves import hyst, outdoor_reset
from housemodel.sourcesink.heatpumps.NTA8800_Q.HPQ9 import calc_WP_general

import housemodel.tools.radiator_performance.ReturnTemperature as Tr
# import housemodel.tools.ReturnTemperature as Tr
from housemodel.sourcesink.radiators.radiators import Radiator
from housemodel.panels.pvt_panel import PVTPanel

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

    pvt = PVTPanel.from_yaml(str(CONFIGDIR / "for_buffer_radiator_wp.yaml"))

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

    Qsolar = SourceTerm("Solar")
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
    Qsolar.values.flatten()

    Qint = SourceTerm("Q_internal")
    Qint.connected_to = param['internal']['distribution']
    Qint.values = internal_heat_gain(param['internal']['Q_day'],
                                     param['internal']['delta_Q'],
                                     param['internal']['t1'],
                                     param['internal']['t2'])
    Qint.values = Qint.values.flatten()
    Qint.values = Qint.values[0:days_sim * 24]

    Q_PVT = SourceTerm("PVT Solar irradiation")
    df_irr_PVT = run_qsun_new(df_nen, pvt.azimuth, pvt.inclination)
    Q_PVT.values = (df_irr_PVT.total_irr * pvt.area).values
    Q_PVT.values = Q_PVT.values[0:days_sim * 24]
    Q_PVT.values.flatten()

    # skip additional source terms from solar irradiation and human presence
    Toutdoor = SourceTerm("T_outdoor")
    Toutdoor.values = df_nen.loc[:, 'temperatuur'].values
    Toutdoor.values = Toutdoor.values.flatten()
    Toutdoor.values = Toutdoor.values[0:days_sim * 24]

    # skip additional source terms from solar irradiation and human presence
    RH = SourceTerm("Relative Humidity")
    RH.values = df_nen.loc[:, 'relatieve_vochtigheid'].values * 0.01
    RH.values = RH.values.flatten()
    RH.values = RH.values[0:days_sim * 24]

    SP = SourceTerm("SetPoint")
    SP.values = simple_thermostat(8, 23, 20, 17)
    SP.values = SP.values[0:days_sim * 24].flatten()

    # Interpolation of data
    Toutdoor.interpolate_power(time_sim, control_interval)
    SP.interpolate_power(time_sim, control_interval)
    Qsolar.interpolate_power(time_sim, control_interval)
    Qint.interpolate_power(time_sim, control_interval)
    Q_PVT.interpolate_power(time_sim, control_interval)
    RH.interpolate_power(time_sim, control_interval)

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
    P_supply = np.zeros(len(t))
    Frost_evaporator = np.zeros(len(t))
    Power_evaporator = np.zeros(len(t))
    Tevaporator = np.zeros(len(t))

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
    nta = HeatpumpNTANew(name="HP")
    # Using the calibratoin values from the datasheet of a Nibe S2125-12 heat pump
    # Location: Documentation\Literature\References_Zon_als_Bron\Nibe_warmtepomp_referentie
    nta.cal_T_evap = np.array([-7, 7, 7])
    nta.cal_T_cond = np.array([35, 35, 45])
    nta.set_cal_val([2.65, 5.21, 3.91], [7.23, 3.67, 3.35])
    nta.Pmax_kW = 10  # in kW
    nta.Pmin_kW = 2
    nta.T_evap = Toutdoor.values[0]
    nta.T_cond_or = 45.0    #initial value
    nta.T_cond_out = nta.T_cond_or
    nta.update()

    water_temp = np.zeros(len(t))
    cop_hp = np.zeros(len(t))
    cop_carnot = np.zeros(len(t))


    nta.set_flow(total.flows[1])
    print(f"Heat rate: {nta.flow.heat_rate} [W/K] \n")

    # define hysteresis object for heat pump
    # hp_hyst = hyst(dead_band=0.5, state=True)

    # Radiator object
    deg = u"\u00b0"     # degree sign
    r = Radiator(name="Rad", exp_rad=1.3)
    r.T_supply = TBuffervessel0[0]
    r.T_amb = Tair[0]
    r.T_return = (r.T_supply + r.T_amb) / 2.0  # crude quess
    r.set_qzero(8000)

    r.set_flow(total.flows[0])
    print(f"Heat rate: {r.flow.heat_rate} [W/K] \n")

    r.update(r.func_rad_lmtd)
    print(f"Q_dot: {r.q_dot}, T_return: {r.T_return}")
    print(f"radiator to room: {r.flow.heat_rate * (TBuffervessel0[0] - r.T_return)} [W]")
    print(f"radiator to room: {r.Km * np.power(r.get_lmtd(), r.exp_rad)} [W] with Delta T_LMTD = {r.get_lmtd()}")
    print(f"top-bottom: {r.flow.heat_rate * (TBuffervessel0[0] - TBuffervessel7[0])} [W]")
    print(f"back-to-bottom: {r.flow.heat_rate * (r.T_return - TBuffervessel7[0])} [W]")

    # inputs = (cap_mat_inv, cond_mat, q_vector, control_interval)
    #inputs = (total, Q_vectors, control_interval)
    inputs = (total, control_interval)

    # Note: the algorithm can take an initial step
    # larger than the time between two elements of the "t" array
    # this leads to an "index-out-of-range" error in the last evaluation
    # of the model function, where e.g SP_T[8760] is called.
    # Therefore, set "first_step" equal or smaller than the spacing of "t".
    # https://github.com/scipy/scipy/issues/9198

    source_list = [Qsolar, Qint]
    for i in tqdm(range(len(t) - 1)):

        # first reset and update total.q_vec
        total.make_empty_q_vec()    # reset q_vec!!!!!
        total.parts[0].ambient.update(Toutdoor.values[i])  # assuming Building is parts[0]
        total.add_ambient_to_q()
        if source_list:
            for s in source_list:
                total.add_source_to_q(s, i)
            # logging.info(f" q_vector: \n {total.q_vec}")

        total.combine_flows()

        # here comes the "arduino style" controller
        pid.SetPoint = SP.values[i]
        pid.update(Tair[i], t[i])
        Qinst = pid.output

        bottom_layer = total.find_tag_from_node_label("DictBuffer7")
        upper_layer = total.find_tag_from_node_label("DictBuffer0")
        # Heat pump NTA800
        # p_hp = 0
        # determine new setting for COP and heat pump power

        nta.T_cond_or = outdoor_reset(Toutdoor.values[i], 2, 20)  # stooklijn klopt niet helemaal!
        water_temp[i] = nta.T_cond_out
        nta.T_evap = Toutdoor.values[i]
        nta.T_cond_in = TBuffervessel7[i]
        nta.adjust()

        # Calcuate the expected het flow if the heating is running
        nta.flow.set_flow_rate(150.0e-6)
        P_expected = nta.flow.heat_rate * (nta.T_cond_out - nta.T_cond_in)/1000
        if (P_expected > nta.Pmin_kW) & (SP.values[i]>16.5):
            nta.flow.set_flow_rate(150.0e-6)
            P_supply[i] = nta.flow.heat_rate * (nta.T_cond_out - nta.T_cond_in)
            cop_hp[i] = nta.COP
            cop_carnot[i] = ((outdoor_reset(Toutdoor.values[i], 2, 20) + 273.15) / (
                        (outdoor_reset(Toutdoor.values[i], 2, 20) + 273.15) - (Toutdoor.values[i] + 273.15))) * 0.45
        else:
            nta.flow.set_flow_rate(0)
            P_supply[i] = 0
            cop_hp[i] = 0
        # update q_vector: add heat source for Buffer vessel
        if Tair[i] < 20:
            r.flow.set_flow_rate(150.0e-6)
        else:
            r.flow.set_flow_rate(0)

        if(P_supply[i] == 0):
            P_hp_electric = 0
            P_PVT_thermal = 0
        else:
            P_hp_electric = P_supply[i] / cop_carnot[i]
            P_PVT_thermal = P_hp_electric * float(cop_carnot[i] - 1.0)

        P_PVT_thermal -= max(0.0, min(100000.0, float(Q_PVT.values[i])))
        P_PVT_thermal_kW = P_PVT_thermal/1000
        #P_PVT_thermal_kW = max(-10.0, min(10.0, P_PVT_thermal_kW))

        pvt.model.update(Toutdoor.values[i]+273.15, RH.values[i], P_PVT_thermal_kW, 0.5)

        r.T_supply = TBuffervessel0[i]
        r.T_amb = Tair[i]
        r.T_return = (r.T_supply + r.T_amb) / 2.0  # crude quess
        r.update(r.func_rad_lmtd)

        total.q_vec[upper_layer] += nta.flow.heat_rate * nta.T_cond_out
        total.f_mat[upper_layer, upper_layer] += nta.flow.heat_rate
        total.q_vec[bottom_layer] += r.flow.heat_rate * r.T_return
        total.f_mat[bottom_layer, bottom_layer] += r.flow.heat_rate

        # update q_vector: add heat source for Building
        air_node = total.find_tag_from_node_label("air")
        # total.q_vec[air_node] += Qinst          # GasBoiler via PID controller
        total.q_vec[air_node] += r.q_dot   # via radiator

        # supply flow
        # total.flows[1].set_flow_rate(nta.flow.flow_rate)  # Flow object????
        # demand flow
        # total.flows[0].set_flow_rate(radiator.flow.flow_rate)
        # total.flows[0].set_flow_rate(radiator.flow_rate)

        #total.combine_flows()

        ts = [t[i], t[i + 1]]
        result = solve_ivp(model_radiator_ne, ts, y0,
                           method='RK45', args=inputs,
                           first_step=control_interval)

        Tair[i + 1] = result.y[0, -1]
        Twall[i + 1] = result.y[1, -1]
        Treturn[i] = r.T_return
        Power_gb[i] = Qinst
        Power_hp[i] = nta.P_HP_W
        Power_evaporator[i] = P_PVT_thermal_kW * 1000
        Frost_evaporator[i] = pvt.model.Total_frost
        Tevaporator[i] = pvt.model.T_0 - 273.15

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
                TBuffervessel4, TBuffervessel5, TBuffervessel6, TBuffervessel7,
                Frost_evaporator, Power_evaporator, Tevaporator)

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
        ax[0, 0].plot(time_d, data[18], label='Tevaporator')
        #ax[0, 0].plot(time_sim / (3600 * 24), SP.values, label='SP_Temperature')
        ax[0, 0].plot(time_sim / (3600 * 24), Toutdoor.values, label='Toutdoor')
        ax[0, 0].legend(loc='upper right')
        ax[0, 0].set_title('Nodal Temperatures')
        ax[0, 0].set_xlabel(('Time (s)'))
        ax[0, 0].set_ylabel(('Temperature (°C)'))

        ax[0, 1].plot(time_d, data[7], label='COP', color='r')
        ax[0, 1].plot(time_d, data[16], label='Frost', color='b')
        ax[0, 1].plot(time_d, cop_carnot, label='Frost', color='g')

        ax[0, 1].legend(loc='upper right')
        ax[0, 1].set_title('COP')
        ax[0, 1].set_xlabel(('Time (s)'))
        ax[0, 1].set_ylabel(('COP'))

        ax[1, 0].plot(time_d, data[6], label='Condensor temp', color='c')
        ax[1, 0].legend(loc='upper right')
        ax[1, 0].set_title('Condensor Temperature')
        ax[1, 0].set_xlabel(('Time (s)'))
        ax[1, 0].set_ylabel(('T (°C)'))

        ax[1, 1].plot(time_d, data[3], label='Return temp', color='b')
        ax[1, 1].legend(loc='upper right')
        ax[1, 1].set_title('Return Temperature')
        ax[1, 1].set_xlabel(('Time (s)'))
        ax[1, 1].set_ylabel(('Temperature (°C)'))

        # ax[2, 0].plot(time_d, data[5], label='Power HP', color = 'r')
        ax[2, 0].plot(time_d, P_supply, label='Power HP', color='r')
        ax[2, 0].plot(time_d, Power_evaporator, label='Power evaporator', color='b')
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
