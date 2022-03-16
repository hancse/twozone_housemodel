import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

from housemodel.simulation import Simulation
from housemodel.for_companies_solver import SolverForCompanies

from housemodel.constants import *

from housemodel.controls.Temperature_SP import simple_thermostat

# Contains a main to run show/plot and export to excel just for testing Simulation/Solver.
# See run_for_companies.py for comments also applicable to this file.

# Suggestion is to make a show/plot function that create (sub)plots based on configuration. See also comments in
# Simulation with 'data' dictionary.

# show function should be in separate file(s)/module(s)
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
    from housemodel.sourcesink.NEN5060 import run_qsun
    from housemodel.weather_solar.weatherdata import read_nen_weather_from_xl, NENdatehour2datetime
    from scipy.interpolate import interp1d

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


# example main to be used in separate run/test/show
if __name__ == '__main__':
    from housemodel.tools.new_configurator import load_config

    house_param = load_config("excel_for_companies.yaml")
    house_param.setdefault('timing', {})
    # house_param['timing']['days_sim'] = 100
    sim = Simulation(house_param, SolverForCompanies, SP=simple_thermostat(8, 23, 20, 17))


    sim.run()

    show(sim)
    export_xl(sim)
