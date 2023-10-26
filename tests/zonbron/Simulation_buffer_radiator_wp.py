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
    nta = HeatpumpNTANew()
    nta.Pmax = 8           # kW
    nta.set_cal_val([4.0, 3.0, 2.5], [6.0, 2.0, 3.0])

    nta.c_coeff = calc_WP_general(nta.cal_T_evap, nta.cal_T_cond,
                                  nta.cal_COP_val, order=1)

    nta.p_coeff = calc_WP_general(nta.cal_T_evap, nta.cal_T_cond,
                                  nta.cal_Pmax_val, order=1)
    water_temp = np.zeros(len(t))
    cop_hp = np.zeros(len(t))

    # define hysteresis object for heat pump
    # hp_hyst = hyst(dead_band=0.5, state=True)

    # Radiator object
    deg = u"\u00b0"     # degree sign
    radiator = Radiator(name="Rad", exp_rad=1.3)
    radiator.T_supply = TBuffervessel0[0]

    radiator.T_amb = Tair[0]
    radiator.T_return = (radiator.T_supply + radiator.T_amb) / 2.0  # crude quess

    radiator.Km = 12.5
    radiator.flow.set_flow_rate(0.040e-3)  # m^3/s
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


    print()


if __name__ == "__main__":
    main(show=True, xl=True)  # temporary solution, recommended syntax
