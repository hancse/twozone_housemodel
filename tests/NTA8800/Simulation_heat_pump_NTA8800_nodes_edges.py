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

from housemodel.solvers.house_model_heat_pump_NTA8800_ne import house_radiator_ne  # exposed function "house" in house module
# function "model" in module house is private

from housemodel.tools.new_configurator import (load_config)
from housemodel.sourcesink.NEN5060 import run_qsun

from housemodel.sourcesink.internal_heat_gain import internal_heat_gain
from housemodel.controls.Temperature_SP import simple_thermostat

from housemodel.weather_solar.weatherdata import (read_nen_weather_from_xl,
                                                  NENdatehour2datetime)
from housemodel.buildings.building import Building
from housemodel.sourcesink.radiators.linear_radiator import LinearRadiator
from housemodel.basics.powersource import PowerSource
from housemodel.basics.totalsystem import TotalSystem

# import matplotlib
# matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

import logging
logging.basicConfig()
logger = logging.getLogger('8800NE')
logger.setLevel(logging.INFO)


from pathlib import Path
CONFIGDIR = Path(__file__).parent.absolute()


def main(show=False, xl=False):
    # read configuration file into dictionary
    param = load_config(str(CONFIGDIR / "for_NTA8800_nodes_edges.yaml"))

    # Obtain the number of days simulated
    days_sim = math.ceil(param['timing']['Duration'] / 24)
    print(days_sim)

    # create Building object
    h = Building("MyHouse")
    section = param["Building"]
    # read nodes attribute from dictionary and create capacity matrix
    h.nodes_from_dict(section["nodes"])
    h.fill_c_inv()

    # read FixedNode objects (external nodes)
    h.boundaries_from_dict(param["boundaries"])  # function selects "outdoor" as ambient
    h.make_k_ext_and_add_ambient()  # initialize k_ext_mat and add diagonal elements

    logger.info(f" \n\n C^-1: \n {h.c_inv_mat} \n K_ext: \n {h.k_ext_mat}, \n q_vec: \n {h.q_vec} \n")

    # Create an object for a radiator
    r = LinearRadiator("SimpleRadiator")
    section = param["Radiator"]
    # Fill in the parameters of the radiator
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

    # compose k-matrix from parts
    total.edges_from_dict(param["edges"])
    total.fill_k(param["edges"])
    total.merge_k_ext()
    total.k_mat += total.k_ext_mat

    total.make_empty_q_vec()
    logger.info(f" \n\n {total.c_inv_mat} \n\n {total.k_mat}, \n\n {total.q_vec} \n")

    # read NEN5060 data from spreadsheet NEN5060-2018.xlsx into pandas DataFrame
    df_nen = read_nen_weather_from_xl()
    # generate and insert timezone-aware UTC and local timestamps (with DST)
    df_nen = NENdatehour2datetime(df_nen)
    df_irr = run_qsun(df_nen)
    print(df_irr.head())
    time_sim = df_irr.iloc[0:days_sim*24, 0].values

    # Read control interval from config file
    control_interval = param["timing"]["Timestep"] * 60

    # Add Solar irradiation as a power source and get the orientation of the windows from the config file
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

    # TODO: What happens here?
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
    # TODO: check if this is neccesary
    time_sim = np.arange(0, time_sim[-1] + (6 * 600), control_interval)

    # TODO: Remove this, nbo PID needed for NTA8800 heatpump, works with hysteresis
    # Input PID values in to control
    controllers = []
    for n in range(len(param['controllers'])):
        c = param['controllers'][n]
        controllers.append(c)

    # solve ODE
    data = house_radiator_ne(time_sim, total, Q_vectors,
                             Toutdoor,
                             Qsolar,
                             Qint,
                             SP, control_interval, controllers)

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
        fig, ax = plt.subplots(2, 2, sharex='all')
        ax[0, 0].plot(data[0],data[1], label='Tair')
        ax[0, 0].plot(data[0],data[2], label='Twall')
        ax[0, 0].plot(data[0], data[3], label='Tradiator')
        ax[0, 0].plot(data[0], SP.values, label='SP_Temperature')
        ax[0, 0].plot(data[0], Toutdoor.values, label='Toutdoor')
        ax[0, 0].legend(loc='upper right')
        ax[0, 0].set_title('Nodal Temperatures')
        ax[0, 0].set_xlabel(('Time (s)'))
        ax[0, 0].set_ylabel(('Temperature (°C)'))

        ax[0, 1].plot(data[0], data[6], label='COP', color='r')
        ax[0, 1].legend(loc='upper right')
        ax[0, 1].set_title('COP')
        ax[0, 1].set_xlabel(('Time (s)'))
        ax[0, 1].set_ylabel(('COP'))

        ax[1, 0].plot(data[0], data[4], label='Power', color='c')
        ax[1, 0].legend(loc='upper right')
        ax[1, 0].set_title('Power')
        ax[1, 0].set_xlabel(('Time (s)'))
        ax[1, 0].set_ylabel(('Power (kW)'))

        ax[1, 1].plot(data[0], data[5], label='Water temp',color='b')
        ax[1, 1].legend(loc='upper right')
        ax[1, 1].set_title('Water Temperature')
        ax[1, 1].set_xlabel(('Time (s)'))
        ax[1, 1].set_ylabel(('Temperature (°C)'))
        plt.tight_layout()
        plt.suptitle(Path(__file__).stem)
        plt.show()

    if xl:
        xlname = 'tst_NTA8800.xlsx'
        logger.info(f"writing Excel file {xlname}...")
        # df_out = pd.DataFrame(data[0], columns=['Timestep'])
        df_out = pd.DataFrame({'Timestep': data[0]})
        df_out['Outdoor temperature'] = Toutdoor.values
        # df_out['NEN5060_global'] = glob.values
        # df_out['cloud_cover'] = cloud.values
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

        for r in dataframe_to_rows(df_out, index=False):
            ws.append(r)
        # df_out.to_excel('tst.xlsx', index=False, startrow=10)
        wb.save(xlname)
        logger.info(f"Excel file {xlname} written")


if __name__ == "__main__":
    main(show=True, xl=True)  # temporary solution, recommended syntax
