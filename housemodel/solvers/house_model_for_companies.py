"""
house model base on 2R2C model with a buffervessel and a radiator
"""

from scipy.integrate import solve_ivp       # ODE solver
import numpy as np                       # linear algebra
from housemodel.tools.PIDsim import PID

def model_radiator_m(t, x, cap_mat_inv, cond_mat, q_vector,
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

def house_radiator_m(cap_mat_inv, cond_mat, q_vector,
                     SP_T, time_sim, control_interval, control_parameters):
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
    #initial values for solve_ivp
    Tair0 = 15
    Twall0 = 20
    Tradiator0 = 40

    y0 = [Tair0, Twall0, Tradiator0]

    t = time_sim           # Define Simulation time with sampling time
    Tair = np.ones(len(t)) * Tair0
    Twall = np.ones(len(t)) * Twall0
    Tradiator = np.ones(len(t)) * Tradiator0

    # Controller initialization
    # heatingPID = PID(Kp=10, Ki=100, Kd=0, beta=1, MVrange=(0, 12000), DirectAction=False)
    # heating = 0
    kp = control_parameters[0]
    ki = control_parameters[1]
    kd = control_parameters[2]


    inputs = (cap_mat_inv, cond_mat, q_vector, control_interval)
    # Note: the algorithm can take an initial step
    # larger than the time between two elements of the "t" array
    # this leads to an "index-out-of-range" error in the last evaluation
    # of the model function, where e.g SP_T[8760] is called.
    # Therefore set "first_step" equal or smaller than the spacing of "t".
    # https://github.com/scipy/scipy/issues/9198
    for i in range(len(t)-1):

        # here comes the controller

        # Simple PID controler
        Qinst = (SP_T[i] - Tair[i]) * kp
        Qinst = np.clip(Qinst, 0, 12000)
        q_vector[2, i] = Qinst

        #Velocity PID controller (not working properly)
        # heating  = heatingPID.update(t[i], SP_T[i], Tair[i], heating)
        # q_vector[2, i] = heating

        ts = [t[i], t[i+1]]
        result = solve_ivp(model_radiator_m, ts, y0,
                        method='RK45', args=inputs,
                        first_step=control_interval)

        Tair[i+1] = result.y[0, -1]
        Twall[i+1] = result.y[1, -1]
        Tradiator[i+1] = result.y[2, -1]

        y0 = result.y[:, -1]

    # heatingPID.plot()
    return Tair, Twall, Tradiator, t

