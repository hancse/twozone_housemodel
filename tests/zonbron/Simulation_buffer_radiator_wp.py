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

    b = StratifiedBufferNew.from_dict(param["Buffer"])


    print()


if __name__ == "__main__":
    main(show=True, xl=True)  # temporary solution, recommended syntax
