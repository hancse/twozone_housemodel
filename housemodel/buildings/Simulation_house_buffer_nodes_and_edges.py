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

from housemodel.solvers.house_model_for_companies_ne import \
    house_radiator_ne  # exposed function "house" in house module
# function "model" in module house is private

from housemodel.tools.new_configurator import load_config
# from housemodel.tools.ckf_tools import make_c_inv_matrix, make_edges
from housemodel.tools.ckf_tools import (add_c_inv_block,
                                        add_k_block,
                                        stack_q)

from housemodel.sourcesink.NEN5060 import run_qsun
from housemodel.sourcesink.internal_heat_gain import internal_heat_gain, simple_internal
from housemodel.controls.Temperature_SP import simple_thermostat
from housemodel.weather_solar.weatherdata import (read_nen_weather_from_xl,
                                                  NENdatehour2datetime)
from housemodel.buildings.building import Building
from housemodel.sourcesink.buffervessels.stratified import StratifiedBuffer
from housemodel.buildings.linear_radiator import LinearRadiator

# from housemodel.sourcesink.radiators import Radiator
from housemodel.sourcesink.flows import Flow
from housemodel.buildings.totalsystem import TotalSystem
# from housemodel.buildings.components import FixedNode

from housemodel.constants import *
# from housemodel.simulation2 import Simulation2
# from housemodel.for_companies_solver import SolverForCompanies


import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import logging

logging.basicConfig()
logger = logging.getLogger('HBNE')
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

from pathlib import Path

CONFIGDIR = Path(__file__).parent.absolute()


def show(sim):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))  # key-value pair: no spaces
    plt.plot(sim.t, sim.data[KEY_T_AIR], label='Tair')
    plt.plot(sim.t, sim.data[KEY_T_WALL], label='Twall')
    plt.plot(sim.t, sim.data[KEY_T_RADIATOR], label='Tradiator')
    plt.plot(sim.t, sim.SP_sim, label='SP_Temperature')
    plt.plot(sim.t, sim.T_outdoor_sim, label='Toutdoor')
    plt.plot(sim.t, sim.Qinst, label='Qinst')
    plt.legend(loc='best')
    plt.title("Simulation2R2C_companies")
    plt.show()


def export_xl(sim):
    # from housemodel.sourcesink.NEN5060 import run_qsun
    # from housemodel.weather_solar.weatherdata import read_nen_weather_from_xl, NENdatehour2datetime
    # from scipy.interpolate import interp1d

    control_interval = sim.control_interval

    # df_out = pd.DataFrame(data[0], columns=['Timestep'])
    df_out = pd.DataFrame({'Timestep': sim.time_sim})
    df_out['Outdoor temperature'] = sim.T_outdoor_sim
    df_out['NEN5060_global'] = sim.data[KEY_SOLAR]
    df_out['cloud_cover'] = sim.data[KEY_CLOUD]
    df_out["Heating"] = sim.Qinst.tolist()
    df_out['Setpoint'] = sim.SP_sim

    num_links = len(sim.house_param["chains"][0]["links"])

    for n in range(num_links):
        nodename = sim.house_param['chains'][0]['links'][n]['Name']
        df_out["T_{}".format(n)] = sim.solver.data[n + 1].tolist()
        # df_out["Solar_{}".format(n)] = sim.Qsolar_sim[n, :]
        if nodename == 'Internals':
            df_out["Internal_{}".format(n)] = sim.data[KEY_Q_INTERNAL]

    df_out['Tradiator'] = sim.data[KEY_T_RADIATOR].tolist()  # FIXME this is the same as T_2?

    wb = Workbook()
    ws = wb.active

    for r in dataframe_to_rows(df_out, index=False):
        ws.append(r)
    # df_out.to_excel('tst.xlsx', index=False, startrow=10)
    wb.save('tst_ML.xlsx')


def main(show=False, xl=False):
    # read configuration file into dictionary
    param = load_config(str(CONFIGDIR / "xl_for_2R2Chouse_buffer.yml"))
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

    # read ALL edges in the system; select edges connecting nodes in taglist; create conductivity matrix
    # h.edges_from_dict(param["edges"])
    # h.fill_k(param["edges"])

    # create empty q-vector (all zeros)
    # h.make_empty_q_vec()
    # the q_vec is made in the iteration, after all elements of T_outdoor have been read from NEN5060
    # h.add_ambient_to_q()

    logger.info(f" \n\n C^-1: \n {h.c_inv_mat} \n K_ext: \n {h.k_ext_mat}, \n q_vec: \n {h.q_vec} \n")

    # create StratifiedBuffer object
    b = StratifiedBuffer("Buffervessel")
    section = param["Buffer"]
    b.nodes_from_dict(section["nodes"])
    b.fill_c_inv()

    b.boundaries_from_dict(param["boundaries"])  # function selects "indoor" as ambient
    # b.add_fixed_to_k()  # crashes because Fixed node connects to node 2,3,4 and b.k_matrix has rank 3 (index 0,1,2)
    b.make_k_ext_and_add_ambient()

    # b.edges_from_dict(param['edges'])
    # b.fill_k(param["edges"])

    # b.make_empty_q_vec()
    # b.add_ambient_to_q()

    logger.info(f" \n\n C^-1: \n {b.c_inv_mat} \n K_ext: \n {b.k_ext_mat}, \n q_vec: \n {b.q_vec} \n")

    r = LinearRadiator("SimpleRadiator")
    r.nodes_from_dict(param["Radiator"]["nodes"])
    r.fill_c_inv()
    r.make_empty_k_ext_mat()

    logger.info(f" \n\n C^-1: \n {r.c_inv_mat} \n K_ext: \n {r.k_ext_mat}, \n q_vec: \n {r.q_vec} \n")

    # create Totalsystem object and sort parts
    total = TotalSystem("HouseWithRadiator", [r, h])
    total.sort_parts()
    # compose c-1-matrix from parts and merge tag_lists
    total.merge_c_inv()
    total.merge_tag_lists()

    # compose k-matrix from parts
    total.edges_from_dict(param["edges"])
    total.fill_k(param["edges"])

    total.merge_k_ext()
    total.k_mat += total.k_ext_mat

    total.make_empty_q_vec()

    logger.info(f" \n\n {total.c_inv_mat} \n\n {total.k_mat}, \n\n {total.q_vec} \n")

    # r = Radiator(1.3)
    # r.boundaries_from_dict(param["boundaries"])
    # r.T_amb = 20.0  # = h.Tini

    # calculate flow matrices and combine into f_mat_all
    flows = []
    for n in range(len(param['flows'])):
        flows.append(Flow())
        flows[n].flow_from_dict(param['flows'][n])
        flows[n].make_Fmatrix(rank=total.k_mat.shape[0])

    # combine Fmatrices into matrix Fall
    f_mat_all = np.zeros_like(flows[0].f_mat)
    for n in range(len(flows)):
        f_mat_all += flows[n].f_mat
    # f_mat_all = np.add(flows[0].f_mat, flows[1].f_mat)
    print(f_mat_all, "\n")

    # remove matrix elements > 0 from Fall
    f_mat_all = np.where(f_mat_all <= 0, f_mat_all, 0)
    print(f_mat_all, "\n")

    # create diagonal elements in Fall, so that som over each row is zero
    row_sums = np.sum(f_mat_all, axis=1).tolist()
    f_mat_all = f_mat_all - np.diag(np.array(row_sums), k=0)
    print(f_mat_all, "\n")

    # Loading the radiator and buffervessel parameters
    # Heat transfer coefficient of the radiator and het capacity
    # UAradiator = house_param["chains"][0]["links"][2]["Conductance"]
    # Crad =  house_param["chains"][0]["links"][2]["Capacity"]

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

    Qsolar = (df_irr.total_E * param['solar_irradiation']['E'] +
              df_irr.total_SE * param['solar_irradiation']['SE'] +
              df_irr.total_S * param['solar_irradiation']['S'] +
              df_irr.total_SW * param['solar_irradiation']['SW'] +
              df_irr.total_W * param['solar_irradiation']['W'] +
              df_irr.total_NW * param['solar_irradiation']['NW'] +
              df_irr.total_N * param['solar_irradiation']['N'] +
              df_irr.total_NE * param['solar_irradiation']['NE']).values
    Qsolar *= param['solar_irradiation']['g_value']
    Qsolar_sim = Qsolar[0:days_sim * 24]

    # q_int = simple_internal(t_on=param['internal']['t1'],
    #                         t_off=param['internal']['t2'],
    #                         Q_day=param['internal']['Q_day'],
    #                         Q_night=param['internal']['Q_day'] - param['internal']['delta_Q'])
    # Qinternal_sim = q_int[0:days_sim*24]

    Qint = internal_heat_gain(param['internal']['Q_day'],
                              param['internal']['delta_Q'],
                              param['internal']['t1'],
                              param['internal']['t2'])
    Qint = Qint.flatten()
    Qinternal_sim = Qint[0:days_sim * 24]

    Toutdoor = df_nen.loc[:, 'temperatuur'].values
    Toutdoor = Toutdoor.flatten()  # temperature
    T_outdoor_sim = Toutdoor[0:days_sim * 24]

    SP = simple_thermostat(8, 23, 20, 17)
    SP_sim = SP[0:days_sim * 24].flatten()

    # make predictable part of q_dot vector
    # q_vector = np.zeros((t.num_nodes,days_sim*24))
    # leak_to_amb = param["chains"][0]["links"][0]["Conductance"]
    # q_vector[0, :] = (T_outdoor_sim * leak_to_amb) + Qinternal_sim + CF * Qsolar_sim
    # q_vector[1, :] = (1 - CF) * Qsolar_sim

    # Interpolation of data
    # interp_func = interp1d(time_sim, q_vector, fill_value='extrapolate')
    interp_func_SP = interp1d(time_sim, SP_sim, kind='nearest', fill_value='extrapolate')
    interp_func_Q_internal = interp1d(time_sim, Qinternal_sim, fill_value='extrapolate')
    interp_func_Toutdoor = interp1d(time_sim, T_outdoor_sim, fill_value='extrapolate')
    # q_vector = interp_func(np.arange(0, time_sim[-1]+(6*600), control_interval))
    SP_sim = interp_func_SP(np.arange(0, time_sim[-1] + (6 * 600), control_interval))
    T_outdoor_sim = interp_func_Toutdoor(np.arange(0, time_sim[-1] + (6 * 600), control_interval))
    Qinternal_sim = interp_func_Q_internal(np.arange(0, time_sim[-1] + (6 * 600), control_interval))

    glob = df_nen['globale_zonnestraling'].values
    glob = glob.flatten()
    interp_func_glob = interp1d(time_sim, glob, fill_value='extrapolate')
    glob_interp = interp_func_glob(np.arange(0, time_sim[-1] + (6 * 600), control_interval))

    cloud = df_nen['bewolkingsgraad'].values
    cloud = cloud.flatten()
    interp_func_cloud = interp1d(time_sim, cloud, fill_value='extrapolate')
    cloud_interp = interp_func_cloud(np.arange(0, time_sim[-1] + (6 * 600), control_interval))

    time_sim = np.arange(0, time_sim[-1] + (6 * 600), control_interval)

    # time_sim = np.linspace(0, time_sim[-1], (8760-1)*6, endpoint=False)

    # Input PID values in to control
    controllers = []
    for n in range(len(param['controllers'])):
        c = param['controllers'][n]
        controllers.append(c)

    total.ambient = h.ambient
    total.ambient.temp = T_outdoor_sim[0]

    # solve ODE
    data = house_radiator_ne(time_sim, total,
                             T_outdoor_sim,
                             Qsolar_sim,
                             # glob_interp, cloud_interp,
                             Qinternal_sim,
                             SP_sim, control_interval, controllers)

    # if show=True, plot the results
    if show:
        plt.figure(figsize=(15, 5))  # key-value pair: no spaces
        plt.plot(data[0], data[1], label='Tair')
        plt.plot(data[0], data[2], label='Twall')
        plt.plot(data[0], data[3], label='Tradiator')
        plt.plot(time_sim, SP_sim, label='SP_Temperature')
        plt.plot(time_sim, T_outdoor_sim, label='Toutdoor')
        plt.plot(data[0], data[4], label='Qinst')
        plt.legend(loc='best')
        plt.title("Simulation2R2C_companies")
        plt.show()

    if xl:
        # df_out = pd.DataFrame(data[0], columns=['Timestep'])
        df_out = pd.DataFrame({'Timestep': data[0]})
        df_out['Outdoor temperature'] = T_outdoor_sim
        df_out['NEN5060_global'] = glob_interp
        df_out['cloud_cover'] = cloud_interp
        df_out["Heating"] = data[4].tolist()
        df_out['Setpoint'] = SP_sim

        for n in range(total.num_nodes):
            nodename = param['chains'][0]['links'][n]['Name']
            df_out["T_{}".format(n)] = data[n + 1].tolist()
            # df_out["Solar_{}".format(n)] = Qsolar_sim[n, :]
            if nodename == 'Internals':
                df_out["Internal_{}".format(n)] = Qinternal_sim

        df_out['Tradiator'] = data[3].tolist()

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
        wb.save('tst_ML.xlsx')


if __name__ == "__main__":
    # compatible with MvdB, TN
    main(show=True, xl=True)  # temporary solution, recommended syntax

    # compatible with ReneJ (run_for_companies.py)
    # param = load_config(str(CONFIGDIR / "xl_for_2R2Chouse_buffer.yml"))
    # param.setdefault('timing', {})
    # house_param['timing']['days_sim'] = 100

    # currentSolver = SolverForCompanies
    # sim = Simulation2(param, currentSolver, SP=simple_thermostat(8, 23, 20, 17))

    # sim.run()

    # show(sim)
    # export_xl(sim)
