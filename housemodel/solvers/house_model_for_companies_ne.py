"""
house model base on 2R2C model with a buffervessel and a radiator
"""

from scipy.integrate import solve_ivp       # ODE solver
import numpy as np                       # linear algebra
# from housemodel.tools.PIDsim import PID
from housemodel.controls.ivPID.PID import PID

from housemodel.buildings.totalsystem import TotalSystem
from housemodel.tools.ckf_tools import (make_c_inv_matrix,
                                        make_edges,
                                        add_c_inv_block,
                                        add_k_block,
                                        stack_q)

import logging

logging.basicConfig(level="INFO")


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
    # States :
    # Tair = x[0]

    # Parameters :
    index = int(t/control_interval)

    # Equations :
    local_q_vector = Q_vectors[:, [index]]
    # local_q_vector[0,0] += q_vector[0,index]
    # local_q_vector[1,0] = q_vector[1,index]
    # local_q_vector[2,0] = q_vector[2,index]

    # tot_sys.add_ambient_to_q()
    # tot_sys.add_source_to_q(Q_solar, index)

    # Conversion of 1D array to a 2D array
    # https://stackoverflow.com/questions/5954603/transposing-a-1d-numpy-array
    x = np.array(x)[np.newaxis]

    dTdt = (-tot_sys.k_mat @ x.T) + local_q_vector
    dTdt = np.dot(tot_sys.c_inv_mat, dTdt)

    return dTdt.flatten().tolist()


def house_radiator_ne(time_sim, tot_sys, Q_vectors,
                      T_outdoor,
                      Q_solar,
                      Q_int,
                      SP_T, cntrl_intrvl, cntrllrs):
    """Compute air and wall temperature inside the house.

    Args:
        time_sim:   (array)  : simulation time
        tot_sys:    (object) : total system C-1 and K matrices, q-vector
        T_outdoor:  (array)  : outdoor temperature
        Q_solar:    (array)  : solar irradiation
        Q_int:      (array)  : internal heat generation
        SP_T:       (array)  : setpoint room temperature from thermostat.
        cntrl_intrvl: (int)  : control interval in seconds
        cntrllrs:     (dict) : PID-like controller objects with kp, ki, kd and maximum output

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
    # in one statement
    # y0 = [cn.temp for cn in [n for node in [p.nodes for p in tot_sys.parts] for n in node]]

    t = time_sim           # Define Simulation time with sampling time
    Tair = np.ones(len(t)) * y0[0]
    Twall = np.ones(len(t)) * y0[1]
    Tradiator = np.ones(len(t)) * y0[2]

    # Controller initialization
    # heatingPID = PID(Kp=5000, Ki=0, Kd=0, beta=1, MVrange=(0, 12000), DirectAction=False)
    # heating = 0
    # kp = control_parameters[0]
    # ki = control_parameters[1]
    # kd = control_parameters[2]

    pid = PID(cntrllrs[0]['kp'],
              cntrllrs[0]['ki'],
              cntrllrs[0]['kd'],
              t[0])

    pid.SetPoint = 17.0
    pid.setSampleTime(0)
    pid.setBounds(0, cntrllrs[0]["maximum"])
    pid.setWindup(cntrllrs[0]["maximum"]/cntrl_intrvl)

    """
    # build initial q-vector
    tot_sys.parts[0].ambient.update(T_outdoor.values[0])  # initial
    # add terms "ambient temperature*conductivity_to_ambient" to q-vector
    q = [c for conn in [p.ambient.connected_to for p in tot_sys.parts if p.ambient is not None] for c in conn]
    tot_sys.add_ambient_to_q()

    source_list = [Q_solar, Q_int]
    for s in source_list:
        tot_sys.add_source_to_q(s, 0)
    logging.debug(f" q_vector: \n {tot_sys.q_vec}")
    """

    inputs = (tot_sys, Q_vectors, cntrl_intrvl)

    # Note: the algorithm can take an initial step
    # larger than the time between two elements of the "t" array
    # this leads to an "index-out-of-range" error in the last evaluation
    # of the model function, where e.g. SP_T[8760] is called.
    # Therefore set "first_step" equal or smaller than the spacing of "t".
    # https://github.com/scipy/scipy/issues/9198

    for i in range(len(t)-1):
        # here comes the "arduino style" controller
        pid.SetPoint = SP_T.values[i]
        pid.update(Tair[i], t[i])
        # q_vector[2, i] = pid.output
        # tot_sys.q_vec[2] = pid.output
        Q_vectors[2, i] = pid.output

        # Simple PID controller
        # Qinst = (SP_T[i] - Tair[i]) * kp
        # Qinst = np.clip(Qinst, 0, 12000)
        # q_vector[2, i] = Qinst

        # Velocity PID controller (not working properly)
        # heating  = heatingPID.update(t[i], SP_T[i], Tair[i], heating)
        # print(f"{heating}")
        # heating  = heatingPID.update(t[i], SP_T[i], Tair[i], heating)
        # print(f"{heating}")
        # q_vector[2, i] = heating

        ts = [t[i], t[i+1]]
        result = solve_ivp(model_radiator_ne, ts, y0,
                           method='RK45', args=inputs,
                           first_step=cntrl_intrvl)

        Tair[i+1] = result.y[0, -1]
        Twall[i+1] = result.y[1, -1]
        Tradiator[i+1] = result.y[2, -1]

        y0 = result.y[:, -1]

    return t, Tair, Twall, Tradiator, Q_vectors[2, :]/1000
