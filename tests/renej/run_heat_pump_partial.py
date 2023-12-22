import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

from simulation import Simulation
from heat_pump_solver import SolverHeatPumpNTA8800PartLoad

from constants import *


# show function should be in separate file(s)/module(s)
def show(sim):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, sharex=True)
    t = sim.t
    ax[0, 0].plot(t, sim.data[KEY_T_AIR], label='Tair')
    ax[0, 0].plot(t, sim.data[KEY_T_WALL], label='Twall')
    ax[0, 0].plot(t, sim.data[KEY_T_RADIATOR], label='Tradiator')
    ax[0, 0].plot(t, sim.data[KEY_SETPOINT], label='SP_Temperature')
    ax[0, 0].plot(t, sim.data[KEY_T_OUTDOOR], label='Toutdoor')
    ax[0, 0].legend(loc='upper right')
    ax[0, 0].set_title('Nodal Temperatures')
    ax[0, 0].set_xlabel(('Time (s)'))
    ax[0, 0].set_ylabel(('Temperature (°C)'))

    ax[0, 1].plot(t, sim.cop_hp, label='COP', color='r')
    ax[0, 1].plot(t, sim.cop_hp_corrected, label='COP corrected', color='b')
    ax[0, 1].legend(loc='upper right')
    ax[0, 1].set_title('COP')
    ax[0, 1].set_xlabel(('Time (s)'))
    ax[0, 1].set_ylabel(('COP'))

    ax[1, 0].plot(t, sim.controlled_power, label='Controlled Power', color='c')
    ax[1, 0].plot(t, sim.p_hp, label='Maximum Power', color='g')
    ax[1, 0].legend(loc='upper right')
    ax[1, 0].set_title('Power')
    ax[1, 0].set_xlabel(('Time (s)'))
    ax[1, 0].set_ylabel(('Power (kW)'))

    ax[1, 1].plot(t, sim.water_temp, label='Water temp', color='b')
    ax[1, 1].legend(loc='upper right')
    ax[1, 1].set_title('Water Temperature')
    ax[1, 1].set_xlabel(('Time (s)'))
    ax[1, 1].set_ylabel(('Temperature (°C)'))
    plt.tight_layout()
    plt.suptitle("Simulation2R2C_heat_pump_part_load")
    plt.show()


def export_xl(sim):
    house_param = sim.house_param
    # df_out = pd.DataFrame(data[0], columns=['Timestep'])
    df_out = pd.DataFrame({'Timestep': sim.solver.data[0]})
    df_out['Outdoor temperature'] = sim.data[KEY_T_OUTDOOR]
    num_links = len(sim.house_param["chains"][0]["links"])

    for n in range(num_links):
        nodename = house_param['chains'][0]['links'][n]['Name']
        df_out["T_{}".format(n)] = sim.solver.data[n + 1].tolist()
        # df_out["Solar_{}".format(n)] = Qsolar_sim[n, :]
        if nodename == 'Internals':
            df_out["Internal_{}".format(n)] = sim.data[KEY_Q_INTERNAL]

    df_out['Tradiator'] = sim.solver.data[3].tolist()
    df_out["Heating"] = sim.solver.data[4].tolist()

    wb = Workbook()
    ws = wb.active
    ws.append(['DESCRIPTION',
               'Resultaten HAN Dynamic Model Heat Built Environment'])
    ws.append(['Chain number', 0])
    ws.append(['Designation', None, '2R-2C-1-zone',
               None, None, None, '2R-2C-1-zone'])
    ws.append(['Node number', None, 0, None, None, None, 1])
    ws.append(['Designation', None,
               house_param['chains'][0]['links'][0]['Name'], None, None, None,
               house_param['chains'][0]['links'][1]['Name']])
    for r in dataframe_to_rows(df_out, index=False):
        ws.append(r)
    # df_out.to_excel('tst.xlsx', index=False, startrow=10)
    wb.save('tst.xlsx')


# example main to be used in separate run/test/show
if __name__ == '__main__':
    from housemodel.tools.new_configurator import load_config

    house_param = load_config("excel_for_companies.yaml")
    house_param.setdefault('timing', {})
    house_param['timing']['days_sim'] = 50
    sim = Simulation(house_param, SolverHeatPumpNTA8800PartLoad)

    sim.run()

    show(sim)
    export_xl(sim)
