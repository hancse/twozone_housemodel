import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp  # ODE solver
from housemodel.tools.new_configurator import (add_chain_to_k, make_c_inv_matrix)
from housemodel.sourcesink.NEN5060 import run_qsun
from housemodel.weather_solar.weatherdata import read_nen_weather_from_xl, NENdatehour2datetime
from housemodel.sourcesink.internal_heat_gain import internal_heat_gain
from housemodel.controls.Temperature_SP import simple_thermostat

from housemodel.constants import *

import logging

logging.basicConfig()
logger = logging.getLogger('matrix')
logger.setLevel(logging.INFO)


# Note that most is copied from the original files and number of changes are not made to keep the original code
# mostly intact.

# Idea is to have a 'generic' Simulation class with a 'run' method and some methods for default settings.
# The kwargs can be used to set initial settings.
# One of the arguments is a 'solver creator' (can be function or class) that is used as a Solver (declared after
# Simulation class) in the simulation run.
class Simulation:
    def __init__(self, house_param, solver_creator, *args, **kwargs):
        self.house_param = house_param
        self.days_sim = self.house_param.get('timing', {}).get('days_sim', 365)
        self.solver_creator = solver_creator
        self.SP = None
        self.Qint = None

        # Following allows overriding defaults, but better is to use more understandable names than 'SP' and 'Qint',
        # e.g. thermostat_setpoints and internal_heat_gain.
        # Wasn't changed yet to not further change the algorithmic/run part.
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_thermostat_setpoints(self, house_param):
        if self.SP is None:
            self.SP = simple_thermostat(8, 23, 20, 17)
        return self.SP

    def get_internal_heat_gain(self, house_param):
        if self.Qint is None:
            internal = house_param['internal']
            self.Qint = internal_heat_gain(internal['Q_day'],
                                           internal['delta_Q'],
                                           internal['t1'],
                                           internal['t2'])
        return self.Qint

    def run(self, *args, **kwargs):
        house_param = self.house_param
        days_sim = self.days_sim

        CF = house_param['ventilation']['CF']

        links = house_param["chains"][0]["links"]
        num_links = len(links)
        cap_list = []
        for n in range(num_links):
            cap_list.append(links[n]["Capacity"])
        cap_mat_inv = make_c_inv_matrix(cap_list)

        cond_list = []
        for n in range(num_links):
            cond_list.append(links[n]["Conductance"])

        cond_mat = add_chain_to_k(np.array([cond_list[0]]), cond_list[1], 0)
        cond_mat = add_chain_to_k(cond_mat, cond_list[2], 0)
        print(days_sim)

        # Loading the radiator and buffervessel parameters
        # Heat transfer coefficient of the radiator and het capacity
        UAradiator = links[2]["Conductance"]  # not used?
        Crad = links[2]["Capacity"]  # not used?

        # Interval in seconds the control algorithm
        control_interval = house_param["Timescale"] * 60

        # read NEN5060 data from spreadsheet NEN5060-2018.xlsx into pandas DataFrame
        df_nen = read_nen_weather_from_xl()
        # generate and insert timezone-aware UTC and local timestamps (with DST)
        df_nen = NENdatehour2datetime(df_nen)

        # df_nen = nen5060_to_dataframe()
        df_irr = run_qsun(df_nen)
        print(df_irr.head())
        time_sim = df_irr.iloc[0:days_sim * 24, 0].values

        # for glob and cloud the interp1d needs truncation by time_sim length.
        glob = df_nen['globale_zonnestraling'].values
        glob = glob.flatten()
        interp_func_glob = interp1d(time_sim, glob[0:len(time_sim)], fill_value='extrapolate')
        glob_interp = interp_func_glob(np.arange(0, time_sim[-1] + (6 * 600), control_interval))

        cloud = df_nen['bewolkingsgraad'].values
        cloud = cloud.flatten()
        interp_func_cloud = interp1d(time_sim, cloud[0:len(time_sim)], fill_value='extrapolate')
        cloud_interp = interp_func_cloud(np.arange(0, time_sim[-1] + (6 * 600), control_interval))

        print('RENEJ len(time_sim):', len(time_sim))

        solar_irradiation = house_param['solar_irradiation']
        Qsolar = (df_irr.total_E * solar_irradiation['E'] +
                  df_irr.total_SE * solar_irradiation['SE'] +
                  df_irr.total_S * solar_irradiation['S'] +
                  df_irr.total_SW * solar_irradiation['SW'] +
                  df_irr.total_W * solar_irradiation['W'] +
                  df_irr.total_NW * solar_irradiation['NW'] +
                  df_irr.total_N * solar_irradiation['N'] +
                  df_irr.total_NE * solar_irradiation['NE']).values
        Qsolar *= solar_irradiation['g_value']
        Qsolar_sim = Qsolar[0:days_sim * 24]

        Qint = self.get_internal_heat_gain(house_param)
        Qint = Qint.flatten()
        Qinternal_sim = Qint[0:days_sim * 24]

        Toutdoor = df_nen.loc[:, 'temperatuur'].values  # division by 10 already done in read_nen_weather_from_xl()
        Toutdoor = Toutdoor.flatten()  # temperature
        T_outdoor_sim = Toutdoor[0:days_sim * 24]

        SP = self.get_thermostat_setpoints(house_param)
        SP_sim = SP[0:days_sim * 24].flatten()

        # make predictable part of q_dot vector
        q_vector = np.zeros((num_links, days_sim * 24))
        leak_to_amb = house_param["chains"][0]["links"][0]["Conductance"]
        q_vector[0, :] = (T_outdoor_sim * leak_to_amb) + Qinternal_sim + CF * Qsolar_sim
        q_vector[1, :] = (1 - CF) * Qsolar_sim

        # Interpolation of data
        interp_func = interp1d(time_sim, q_vector, fill_value='extrapolate')
        interp_func_SP = interp1d(time_sim, SP_sim, fill_value='extrapolate')
        interp_func_Q_internal = interp1d(time_sim, Qinternal_sim, fill_value='extrapolate')
        interp_func_Toutdoor = interp1d(time_sim, T_outdoor_sim, fill_value='extrapolate')
        q_vector = interp_func(np.arange(0, time_sim[-1] + (6 * 600), control_interval))
        SP_sim = interp_func_SP(np.arange(0, time_sim[-1] + (6 * 600), control_interval))
        T_outdoor_sim = interp_func_Toutdoor(np.arange(0, time_sim[-1] + (6 * 600), control_interval))
        Qinternal_sim = interp_func_Q_internal(np.arange(0, time_sim[-1] + (6 * 600), control_interval))

        time_sim = np.arange(0, time_sim[-1] + (6 * 600), control_interval)

        # time_sim = np.linspace(0, time_sim[-1], (8760-1)*6, endpoint=False)

        self.time_sim = time_sim
        self.SP_sim = SP_sim
        self.T_outdoor_sim = T_outdoor_sim
        self.Qsolar_sim = Qsolar_sim
        # self.Qinternal_sim = Qinternal_sim

        solver = self.solver_creator(self,  *args, **kwargs)
        solver.solve(cap_mat_inv, cond_mat, q_vector, SP_sim, control_interval)

        self.control_interval = control_interval
        self.t = solver.t
        # self.Tair = solver.Tair
        # self.Twall = solver.Twall
        # self.Tradiator = solver.Tradiator

        # The 'solver' needed to maintain copied code using chains and links.
        self.solver = solver
        # The simulation 'data' contains all simulation data in a dict so the data can be inspected and retrieved.
        # Idea is that after the simulation run the data is enough to make plots, export to excel, etc. and if
        # supported (some day) an external query to show specific or other simulation data. For example, a user
        # runs a simulation and then wants to view some plots or download specific parts of the data, and do
        # various of those things while still operating on the same simulation data (no rerun needed, because
        # simulation state is kept).
        self.data = {
            KEY_TIME: time_sim,
            KEY_T_AIR: solver.Tair,
            KEY_T_WALL: solver.Twall,
            KEY_T_RADIATOR: solver.Tradiator,
            KEY_T_OUTDOOR: T_outdoor_sim,
            KEY_SETPOINT: SP_sim,
            KEY_Q_SOLAR: Qsolar_sim,
            KEY_Q_INTERNAL: Qinternal_sim,
            KEY_SOLAR: glob_interp,
            KEY_CLOUD: cloud_interp,
            # FIXME all outputs by key
        }


# Base Solver class to be derived from for simulations. The Simulation class uses the 'solve' method and the 'solve'
# method uses the 'step' method for each iteration in the the solve loop. Idea is that derived Solver classes only
# have to override the 'step' method.
# Currently, there is a method 'get_mode_function' expected that provides the function used a model. Might be nicer if
# the model function is passed to the Simulation (like solver_creator) and then passed to the Solver by the Simulation.
class Solver:
    def __init__(self, simulation, *args, Tair0=15, Twall0=20, Tradiator0=40, **kwargs):
        # self.time_sim = time_sim

        self.simulation = simulation
        self.house_param = simulation.house_param
        self.time_sim = simulation.time_sim
        self.y0 = [Tair0, Twall0, Tradiator0]
        self.t = simulation.time_sim  # Define Simulation time with sampling time
        self.Tair = np.ones(len(self.t)) * Tair0
        self.Twall = np.ones(len(self.t)) * Twall0
        self.Tradiator = np.ones(len(self.t)) * Tradiator0

        for k, v in kwargs.items():
            setattr(self, k, v)

    def solve(self, cap_mat_inv, cond_mat, q_vector, SP_T, control_interval, *args, **kwargs):
        self.cap_mat_inv = cap_mat_inv
        self.q_vector = q_vector
        self.control_interval = control_interval
        self.cond_mat = cond_mat
        self.SP_T = SP_T
        self.q_vector = q_vector
        self.inputs = (cap_mat_inv, cond_mat, q_vector, control_interval)
        self.model = self.get_model_function()

        # Note: the algorithm can take an initial step
        # larger than the time between two elements of the "t" array
        # this leads to an "index-out-of-range" error in the last evaluation
        # of the model function, where e.g SP_T[8760] is called.
        # Therefore set "first_step" equal or smaller than the spacing of "t".
        # https://github.com/scipy/scipy/issues/9198
        for i in range(len(self.t) - 1):
            self.step(i)

    def step(self, i):
        ts = [self.t[i], self.t[i + 1]]
        result = solve_ivp(self.model, ts, self.y0, method='RK45', args=self.inputs, first_step=self.control_interval)

        self.Tair[i + 1] = result.y[0, -1]
        self.Twall[i + 1] = result.y[1, -1]
        self.Tradiator[i + 1] = result.y[2, -1]

        self.y0 = result.y[:, -1]

    def get_model_function(self):
        return self.default_model_function

    @staticmethod
    def default_model_function(t, x, cap_mat_inv, cond_mat, q_vector, control_interval):
        """model function for scipy.integrate.odeint.

        Args:
            t:           (array):   variable array dependent on time with the vairable Air temperature, Wall temperature Radiator
            x:           (float):
            cap_mat_inv: (float):  diagonal heat capacity matrix
            cond_mat:    (float):  symmetric conductance matrix
            q_vector:    (float):  external heat sources and sinks
            control_interval:

        Returns:
            (list): vector elements of dx/dt
        """
        # States :
        # Tair = x[0]

        # Parameters :
        index = int(t / control_interval)

        # Equations :
        local_q_vector = np.zeros((3, 1))
        local_q_vector[0, 0] = q_vector[0, index]
        local_q_vector[1, 0] = q_vector[1, index]
        local_q_vector[2, 0] = q_vector[2, index]

        # Conversion of 1D array to a 2D array
        # https://stackoverflow.com/questions/5954603/transposing-a-1d-numpy-array
        x = np.array(x)[np.newaxis]

        dTdt = (-cond_mat @ x.T) + local_q_vector
        dTdt = np.dot(cap_mat_inv, dTdt)

        return dTdt.flatten().tolist()
