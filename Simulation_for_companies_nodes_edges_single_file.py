# -*- coding: utf-8 -*-
"""
@author: PvK, MJ
"""
import numpy as np
import math
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp  # ODE solver
from tqdm import tqdm, trange

from housemodel.controls.ivPID.PID import PID

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

from pathlib import Path

import logging

logging.basicConfig()
logger = logging.getLogger('COMP')
# logger.setLevel(logging.DEBUG)
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
    param = load_config(str(CONFIGDIR / "for_companies_nodes_edges.yaml"))
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

    r = LinearRadiator("SimpleRadiator")
    section = param["Radiator"]
    r.nodes_from_dict(section["nodes"])
    r.fill_c_inv()
    r.make_empty_k_ext_mat()
    logger.info(f" \n\n C^-1: \n {r.c_inv_mat} \n K_ext: \n {r.k_ext_mat}, \n q_vec: \n {r.q_vec} \n")

    # create Totalsystem object and sort parts
    total = TotalSystem("HouseWithRadiator", [r, h])
    total.sort_parts()
    # compose c-1-matrix from parts and merge tag_lists
    total.merge_c_inv()
    total.merge_tag_lists()
    # compose k-matrix from edges
    total.edges_from_dict(param["edges"])
    total.fill_k(param["edges"])
    total.merge_k_ext()
    total.k_mat += total.k_ext_mat
    total.merge_ambients()  # assignment by reference, no copy!
    total.make_empty_q_vec()
    logger.info(f" \n\n {total.c_inv_mat} \n\n {total.k_mat}, \n\n {total.q_vec} \n")

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

    # q_int = simple_internal(t_on=param['internal']['t1'],
    #                         t_off=param['internal']['t2'],
    #                         Q_day=param['internal']['Q_day'],
    #                         Q_night=param['internal']['Q_day'] - param['internal']['delta_Q'])
    # Qinternal_sim = q_int[0:days_sim*24]

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
    glob.values = glob.values.flatten()
    glob.interpolate_power(time_sim, control_interval)

    cloud = PowerSource('cloud')
    cloud.values = df_nen['bewolkingsgraad'].values
    cloud.values = cloud.values.flatten()
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
    data = house_radiator_ne(time_sim, total, Q_vectors,
                             Toutdoor,
                             Qsolar,
                             # glob_interp, cloud_interp,
                             Qint,
                             SP, control_interval, controllers)
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

    inputs = (total, Q_vectors, control_interval)

    # Note: the algorithm can take an initial step
    # larger than the time between two elements of the "t" array
    # this leads to an "index-out-of-range" error in the last evaluation
    # of the model function, where e.g. SP_T[8760] is called.
    # Therefore, set "first_step" equal or smaller than the spacing of "t".
    # https://github.com/scipy/scipy/issues/9198

    rad_node = total.find_tag_from_node_label("rad")

    for i in trange(len(t) - 1):
        # here comes the "arduino style" controller
        pid.SetPoint = SP.values[i]
        pid.update(Tair[i], t[i])
        # q_vector[2, i] = pid.output
        # tot_sys.q_vec[2] = pid.output
        Q_vectors[rad_node, i] = pid.output

        # Simple PID controller
        # Qinst = (SP_T[i] - Tair[i]) * kp
        # Qinst = np.clip(Qinst, 0, 12000)
        # Q_vectors[2, i] = pid.output

        # Velocity PID controller (not working properly)
        # heating  = heatingPID.update(t[i], SP_T[i], Tair[i], heating)
        # print(f"{heating}")
        # heating  = heatingPID.update(t[i], SP_T[i], Tair[i], heating)
        # print(f"{heating}")
        # Q_vectors[2, i] = pid.output

        ts = [t[i], t[i + 1]]
        result = solve_ivp(model_radiator_ne, ts, y0,
                           method='RK45', args=inputs,
                           first_step=control_interval)

        Tair[i + 1] = result.y[0, -1]
        Twall[i + 1] = result.y[1, -1]
        Tradiator[i + 1] = result.y[2, -1]

        y0 = result.y[:, -1]

    # return t, Tair, Twall, Tradiator, Q_vectors[rad_node, :] / 1000
    data = (t, Tair, Twall, Tradiator, Q_vectors[rad_node, :] / 1000)

    # if show=True, plot the results
    if show:
        time_d = data[0] / (3600 * 24)
        plt.figure(figsize=(15, 5))  # key-value pair: no spaces
        plt.plot(time_d, data[1], label='Tair')
        plt.plot(time_d, data[2], label='Twall')
        plt.plot(time_d, data[3], label='Tradiator')
        plt.plot(time_sim / (3600 * 24), SP.values, label='SP_Temperature')
        plt.plot(time_sim / (3600 * 24), Toutdoor.values, label='Toutdoor')
        plt.plot(time_d, data[4], label='Qinst')
        plt.legend(loc='best')
        plt.title("Simulation2R2C_companies Nodes and Edges")
        plt.xlabel("time [days]")
        plt.ylabel("temperature [\xb0C]")
        plt.show()

    if xl:
        xlname = 'tst_ML.xlsx'
        logger.info(f"writing Excel file {xlname}...")
        # df_out = pd.DataFrame(data[0], columns=['Timestep'])
        df_out = pd.DataFrame({'Timestep': data[0]})
        df_out['Outdoor temperature'] = Toutdoor.values
        df_out['NEN5060_global'] = glob.values
        df_out['cloud_cover'] = cloud.values
        df_out["Heating"] = data[4].tolist()
        df_out['Setpoint'] = SP.values

        for n in total.tag_list:
            lbl = total.find_node_label_from_tag(n)
            df_out["T_{}".format(lbl)] = data[n + 1].tolist()
            # df_out["Solar_{}".format(n)] = Qsolar_sim[n, :]
            if lbl == 'air':
                df_out["Internal_{}".format(lbl)] = Qint.values

        wb = Workbook()
        ws = wb.active
        # ws.append(['DESCRIPTION',
        #            'Resultaten HAN Dynamic Model Heat Built Environment'])
        # ws.append(['Chain number', 0])
        # ws.append(['Designation', None, '2R-2C-1-zone',
        #            None, None, None, '2R-2C-1-zone'])
        # ws.append(['Node number', None, 0, None, None, None, 1])
        # ws.append(['Designation', None,
        #            house_param['chains'][0]['links'][0]['Name'], None, None, None,
        #            house_param['chains'][0]['links'][1]['Name']])
        for r in dataframe_to_rows(df_out, index=False):
            ws.append(r)
        # df_out.to_excel('tst.xlsx', index=False, startrow=10)
        wb.save(xlname)
        logger.info(f"Excel file {xlname} written")


if __name__ == "__main__":
    # compatible with MvdB, TN
    main(show=True, xl=True)  # temporary solution, recommended syntax
