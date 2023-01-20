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


def model_radiator_ne(t, x, cap_mat_inv, cond_mat, q_vector,
                     control_interval):
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
    local_q_vector = np.zeros((3,1))
    local_q_vector[0,0] = q_vector[0,index]
    local_q_vector[1,0] = q_vector[1,index]
    local_q_vector[2,0] = q_vector[2,index]

    # Conversion of 1D array to a 2D array
    # https://stackoverflow.com/questions/5954603/transposing-a-1d-numpy-array
    x = np.array(x)[np.newaxis]

    dTdt = (-cond_mat @ x.T) + local_q_vector
    dTdt = np.dot(cap_mat_inv, dTdt)

    return dTdt.flatten().tolist()


def house_radiator_ne(time_sim, total_system,
                      T_outdoor,
                      Q_solar,
                      Q_int,
                      SP_T, control_interval, controllers):
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
    # Tair0 = 15
    # Twall0 = 20
    # Tradiator0 = 40
    # y0 = [Tair0, Twall0, Tradiator0]

    y0 = total_system.Tini
    total_system.add_fixed_to_q()

    t = time_sim           # Define Simulation time with sampling time
    Tair = np.zeros(len(t))
    Twall = np.zeros(len(t))
    Tradiator = np.zeros(len(t))

    # Controller initialization
    # heatingPID = PID(Kp=5000, Ki=0, Kd=0, beta=1, MVrange=(0, 12000), DirectAction=False)
    # heating = 0
    # kp = control_parameters[0]
    # ki = control_parameters[1]
    # kd = control_parameters[2]

    pid = PID(controllers[0].kp,
              controllers[0].ki,
              controllers[0].kd,
              t[0])

    pid.SetPoint=17.0
    pid.setSampleTime(0)
    pid.setBounds(0, 12000)
    pid.setWindup(12000/control_interval)

    inputs = (total_system, control_interval)

    # Note: the algorithm can take an initial step
    # larger than the time between two elements of the "t" array
    # this leads to an "index-out-of-range" error in the last evaluation
    # of the model function, where e.g. SP_T[8760] is called.
    # Therefore set "first_step" equal or smaller than the spacing of "t".
    # https://github.com/scipy/scipy/issues/9198

    for i in range(len(t)-1):
        # here comes the "arduino style" controller
        pid.SetPoint = SP_T[i]
        pid.update(Tair[i], t[i])
        # q_vector[2, i] = pid.output
        total_system.q_vec[2] = pid.output

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
                           first_step=control_interval)

        Tair[i+1] = result.y[0, -1]
        Twall[i+1] = result.y[1, -1]
        Tradiator[i+1] = result.y[2, -1]

        y0 = result.y[:, -1]

    return t, Tair, Twall, Tradiator, total_system.q_vec[2,:]/1000

