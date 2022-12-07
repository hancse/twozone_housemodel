# -*- coding: utf-8 -*-
"""
@author: PvK, MJ
"""
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from scipy.interpolate import interp1d

from housemodel.solvers.house_model_for_companies import house_radiator_m  # exposed function "house" in house module
# function "model" in module house is private

from housemodel.tools.new_configurator import load_config
from housemodel.tools.ckf_tools import (make_c_inv_matrix,
                                        make_edges,
                                        add_c_inv_block,
                                        add_k_block,
                                        stack_q)

from housemodel.sourcesink.NEN5060 import run_qsun
from housemodel.sourcesink.internal_heat_gain import internal_heat_gain
from housemodel.controls.Temperature_SP import simple_thermostat
from housemodel.weather_solar.weatherdata import (read_nen_weather_from_xl,
                                                  NENdatehour2datetime)
from housemodel.buildings.house import House
from housemodel.sourcesink.buffervessels.stratified import StratifiedBuffer
from housemodel.sourcesink.radiators import Radiator
from housemodel.sourcesink.flows import Flow
from housemodel.buildings.totalsystem import TotalSystem

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import logging
logging.basicConfig()
logger = logging.getLogger('HBNE')
logger.setLevel(logging.INFO)

from pathlib import Path
CONFIGDIR = Path(__file__).parent.absolute()


def main(show=False, xl=False):
    param = load_config(str(CONFIGDIR / "xl_for_2R2Chouse_buffer.yml"))
    days_sim = int(param['timing']['Duration'] / 24)
    print(days_sim)

    h = House()
    h.nodes_from_dict(param["Building"]["nodes"])
    h.fill_c_inv()
    h.edges_from_dict(param["edges"])
    h.fill_k(param["edges"])
    h.boundaries_from_dict(param["boundaries"])

    h.add_fixed_to_k()
    h.make_q_vec()
    h.add_fixed_to_q()

    print(h.c_inv_mat, '\n')
    print(h.k_mat, '\n')
    print(h.q_vec, '\n')

    b = StratifiedBuffer()
    b.nodes_from_dict(param["Buffer"]["nodes"])
    b.fill_c_inv()
    b.edges_from_dict(param['edges'])
    b.fill_k(param["edges"])
    b.make_q_vec()

    t = TotalSystem()
    t.c_inv_mat = add_c_inv_block(h.c_inv_mat, b.c_inv_mat)
    t.k_mat = add_k_block(h.k_mat, b.k_mat)
    t.q_vec = stack_q(h.q_vec, b.q_vec)

    print(t.c_inv_mat, '\n')
    print(t.k_mat, '\n')
    print(t.q_vec, '\n')

    r = Radiator(1.3)
    r.boundaries_from_dict(param["boundaries"])
    r.T_amb = 20.0

    # calculate flow matrices and combine into f_mat_all
    flows = []
    for n in range(len(param['flows'])):
        flows.append(Flow())
        flows[n].flow_from_dict(param['flows'][n])
        flows[n].make_Fmatrix(rank=t.k_mat.shape[0])

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

    #Loading the radiator and buffervessel parameters
    #Heat transfer coefficient of the radiator and het capacity
    # UAradiator = house_param["chains"][0]["links"][2]["Conductance"]
    # Crad =  house_param["chains"][0]["links"][2]["Capacity"]

    # read NEN5060 data from spreadsheet NEN5060-2018.xlsx into pandas DataFrame
    df_nen = read_nen_weather_from_xl()
    # generate and insert timezone-aware UTC and local timestamps (with DST)
    df_nen = NENdatehour2datetime(df_nen)

    df_irr = run_qsun(df_nen)
    print(df_irr.head())

    time_sim = df_irr.iloc[0:days_sim*24, 0].values

    # Interval in seconds the control algorithm
    # Timestep is in minutes
    control_interval = param["timing"]["Timestep"]*60

    Qsolar = (df_irr.total_E * param['solar_irradiation']['E'] +
              df_irr.total_SE * param['solar_irradiation']['SE'] +
              df_irr.total_S * param['solar_irradiation']['S'] +
              df_irr.total_SW * param['solar_irradiation']['SW'] +
              df_irr.total_W * param['solar_irradiation']['W'] +
              df_irr.total_NW * param['solar_irradiation']['NW'] +
              df_irr.total_N * param['solar_irradiation']['N'] +
              df_irr.total_NE * param['solar_irradiation']['NE']).values
    Qsolar *= param['solar_irradiation']['g_value']
    Qsolar_sim = Qsolar[0:days_sim*24]

    Qint = internal_heat_gain(param['internal']['Q_day'],
                              param['internal']['delta_Q'],
                              param['internal']['t1'],
                              param['internal']['t2'])
    Qint = Qint.flatten()
    Qinternal_sim = Qint[0:days_sim*24]

    Toutdoor = df_nen.loc[:, 'temperatuur'].values
    Toutdoor = Toutdoor.flatten()   # temperature
    T_outdoor_sim = Toutdoor[0:days_sim*24]

    SP = simple_thermostat(8, 23, 20, 17)
    SP_sim = SP[0:days_sim * 24].flatten()

    # make predictable part of q_dot vector
    q_vector = np.zeros((t.num_nodes,days_sim*24))
    leak_to_amb = param["chains"][0]["links"][0]["Conductance"]
    q_vector[0, :] = (T_outdoor_sim * leak_to_amb) + Qinternal_sim + CF * Qsolar_sim
    q_vector[1, :] = (1 - CF) * Qsolar_sim

    # Interpolation of data
    interp_func = interp1d(time_sim, q_vector, fill_value='extrapolate')
    interp_func_SP = interp1d(time_sim, SP_sim, fill_value='extrapolate')
    interp_func_Q_internal = interp1d(time_sim, Qinternal_sim, fill_value='extrapolate')
    interp_func_Toutdoor = interp1d(time_sim, T_outdoor_sim, fill_value='extrapolate')
    q_vector = interp_func(np.arange(0, time_sim[-1]+(6*600), control_interval))
    SP_sim = interp_func_SP(np.arange(0, time_sim[-1]+(6*600), control_interval))
    T_outdoor_sim = interp_func_Toutdoor(np.arange(0, time_sim[-1]+(6*600), control_interval))
    Qinternal_sim = interp_func_Q_internal(np.arange(0, time_sim[-1]+(6*600), control_interval))

    glob = df_nen['globale_zonnestraling'].values
    glob = glob.flatten()
    interp_func_glob = interp1d(time_sim, glob, fill_value='extrapolate')
    glob_interp = interp_func_glob(np.arange(0, time_sim[-1] + (6 * 600), control_interval))

    cloud = df_nen['bewolkingsgraad'].values
    cloud = cloud.flatten()
    interp_func_cloud = interp1d(time_sim, cloud, fill_value='extrapolate')
    cloud_interp = interp_func_cloud(np.arange(0, time_sim[-1] + (6 * 600), control_interval))

    time_sim = np.arange(0, time_sim[-1]+(6*600), control_interval)

    # time_sim = np.linspace(0, time_sim[-1], (8760-1)*6, endpoint=False)


    # Input PID values in to control
    control_parameters = np.zeros(3)
    control_parameters[0] = param['controller']['kp']
    control_parameters[1] = param['controller']['ki']
    control_parameters[2] = param['controller']['kd']

    # solve ODE
    data = house_radiator_m(t.c_inv_mat, t.k_mat, t.q_vec,
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
        df_out['NEN5060_global'] = glob_interp
        df_out['cloud_cover'] = cloud_interp
        df_out["Heating"] = data[4].tolist()
        df_out['Setpoint'] = SP_sim

        for n in range(num_links):
            nodename = house_param['chains'][0]['links'][n]['Name']
            df_out["T_{}".format(n)] = data[n+1].tolist()
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
    main(show=True, xl=True)  # temporary solution, recommended syntax
