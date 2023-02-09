"""
house model base on 2R2C model with a buffervessel and a radiator
"""

from scipy.integrate import solve_ivp       # ODE solver
import numpy as np                       # linear algebra
# from housemodel.tools.PIDsim import PID
from housemodel.controls.ivPID.PID import PID
from housemodel.sourcesink.heatpumps.Heatpump_HM import Heatpump_NTA
from housemodel.controls.heating_curves import hyst, outdoor_reset
from housemodel.sourcesink.heatpumps.NTA8800_Q.HPQ9 import calc_WP_general

def model_radiator_ne(t, x,
                      tot_sys, Q_vectors, control_interval):
    """model function for scipy.integrate.odeint.

    Args:
        t:           (array):   variable array dependent on time with the vairable Air temperature, Wall temperature Radiator
        x:           (float):
        cap_mat_inv: (float):  diagonal heat capacity matrix
        cond_mat:    (float):  symmetric conductance matrix
        q_vector:    (float):  external heat sources and sinks
        SP_T:        (float): thermostat setpoints

    Returns:
        (list): vector elements of dx/dt
    """

    # Parameters :
    index = int(t/control_interval)
    # Take Q vector for certain interval
    local_q_vector = Q_vectors[:, [index]]


    # Conversion of 1D array to a 2D array
    # https://stackoverflow.com/questions/5954603/transposing-a-1d-numpy-array
    x = np.array(x)[np.newaxis]

    dTdt = (-tot_sys.k_mat @ x.T) + local_q_vector
    dTdt = np.dot(tot_sys.c_inv_mat, dTdt)

    return dTdt.flatten().tolist()


def house_radiator_ne(time_sim, tot_sys, Q_vectors,
                            T_outdoor,
                            Q_solar,
                            Qinternal,
                            SP_T,  cntrl_intrvl, cntrllrs):
    """Compute air and wall temperature inside the house.

    Args:
        cap_mat:    (float):  diagonal heat capacity matrix
        cond_mat:   (float):  symmetric conductance matrix
        q_vector:   (float):  external heat sources and sinks
        SP_T:       (array):  Setpoint temperature from thermostat.
        time_sim:   (array)  :  simulation time

    Returns:
        tuple :  (array) containing Tair, Twall, Tradiator and evaluation time:

    Note:
        - Tair (float):   air temperature inside the house in degree C.
        - Twall (float):  wall temperature inside the house in degree C

        - Qinst ?	  (array):  instant heat from heat source such as HP or boiler [W].
    """

    # initial values for solve_ivp
    # make a list of all nodes in total_system
    yn = [n for node in [p.nodes for p in tot_sys.parts] for n in node]
    # make a list of the (initial) temperatures of all nodes
    y0 = [cn.temp for cn in yn]

    t = time_sim  # Define Simulation time with sampling time
    Tair = np.ones(len(t)) * y0[0]
    Twall = np.ones(len(t)) * y0[1]
    Tradiator = np.ones(len(t)) * y0[2]

    # Heat pump initialization
    nta = Heatpump_NTA()
    nta.Pmax = 8
    nta.set_cal_val([4.0, 3.0, 2.5], [6.0, 2.0, 3.0])

    nta.c_coeff = calc_WP_general(nta.cal_T_evap, nta.cal_T_cond,
                                  nta.cal_COP_val, order=1)

    nta.p_coeff = calc_WP_general(nta.cal_T_evap, nta.cal_T_cond,
                                  nta.cal_Pmax_val, order=1)

    water_temp = np.zeros_like(T_outdoor.values)
    cop_hp = np.zeros_like(T_outdoor.values)

    # define hysteresis object for heat pump
    hp_hyst = hyst(dead_band=0.5, state=True)

    inputs = (tot_sys, Q_vectors, cntrl_intrvl)

    # Note: the algorithm can take an initial step
    # larger than the time between two elements of the "t" array
    # this leads to an "index-out-of-range" error in the last evaluation
    # of the model function, where e.g SP_T[8760] is called.
    # Therefore set "first_step" equal or smaller than the spacing of "t".
    # https://github.com/scipy/scipy/issues/9198

    rad_node = tot_sys.find_tag_from_node_label("rad")

    for i in range(len(t)-1):
        # Heat pump NTA800
        # p_hp = 0
        # determine new setting for COP and heat pump power
        test = T_outdoor.values[i]
        water_temp[i] = outdoor_reset(T_outdoor.values[i], 0.7, 20)
        cop_hp[i], p_hp = nta.update(T_outdoor.values[i], water_temp[i])

        # incorporate hysteresis to control
        p_hp = hp_hyst.update(Tair[i], SP_T.values[i], p_hp)

        # update q_vector
        Q_vectors[rad_node, i] = p_hp*1000

        ts = [t[i], t[i+1]]
        result = solve_ivp(model_radiator_ne, ts, y0,
                        method='RK45', args=inputs,
                        first_step=cntrl_intrvl)

        Tair[i+1] = result.y[0, -1]
        Twall[i+1] = result.y[1, -1]
        Tradiator[i+1] = result.y[2, -1]

        y0 = result.y[:, -1]


    return t, Tair, Twall, Tradiator, Q_vectors[rad_node, :]/1000, water_temp, cop_hp

