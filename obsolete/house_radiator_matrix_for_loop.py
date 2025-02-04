"""
house model base on 2R2C model with a buffervessel and a radiator
"""

from scipy.integrate import solve_ivp       # ODE solver
import numpy as np                       # linear algebra

def controllerTemperature_m(setpointTemperature, Tair, Kp = 7000):
    errorroomtemperature = setpointTemperature - Tair

    Qinst = errorroomtemperature * Kp
    Qinst = np.clip(Qinst, 0, 12000)
        
    return Qinst
    

def model_radiator_m(t, x, cap_mat_inv, cond_mat, q_vector,
                     SP_T):
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
    Tair = x[0]

    # Parameters :
    index = int(t/3600)
    # print(f"t: {t}, index: {index}")
    setpointRoomTemperature = SP_T[index]
    #setpointRoomTemperature = SP_T[int(t/3600)]

    # Control :
    Qinst = controllerTemperature_m(setpointRoomTemperature, Tair)
 
    # Equations :
    local_q_vector = np.zeros((3,1))
    local_q_vector[0,0] = q_vector[0,index]
    local_q_vector[1,0] = q_vector[1,index]
    local_q_vector[2,0] = Qinst
    # q_vector[0,0] = (T_outdoor[int(t/3600)] * 214.9718240562546253373451579691) + Q_internal[int(t/3600)] + CF * Q_solar[int(t/3600)]
    # q_vector[1,0] = (1 - CF) * Q_solar[int(t/3600)]
    # q_vector[2,0] = Qinst

    # Conversion of 1D array to a 2D array
    # https://stackoverflow.com/questions/5954603/transposing-a-1d-numpy-array
    x = np.array(x)[np.newaxis]

    dTdt = (-cond_mat @ x.T) + local_q_vector
    dTdt = np.dot(cap_mat_inv, dTdt)

    return dTdt.flatten().tolist()


def house_radiator_m(cap_mat_inv, cond_mat, q_vector,
                     SP_T, time_sim):
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
    Tair0 = 20
    Twall0 = 20
    Tradiator0 = 40

    y0 = [Tair0, Twall0, Tradiator0]

    t = time_sim           # Define Simulation time with sampling time
    print(f"t: {t[0]} - {t[-1]}")
    Tair = np.ones(len(t)) * Tair0
    Twall = np.ones(len(t)) * Twall0
    Tradiator = np.ones(len(t)) * Tradiator0

    inputs = (cap_mat_inv, cond_mat, q_vector, SP_T)
    # Note: the algorithm can take an initial step
    # larger than the time between two elements of the "t" array
    # this leads to an "index-out-of-range" error in the last evaluation
    # of the model function, where e.g SP_T[8760] is called.
    # Therefore set "first_step" equal or smaller than the spacing of "t".
    # https://github.com/scipy/scipy/issues/9198
    for i in range(len(t)-1):

        # here comes the controller

        ts = [t[i], t[i+1]]
        # print(f"index: {i}, ts: {ts[0]} {ts[1]}")
        result = solve_ivp(model_radiator_m, (t[i], t[i+1]), y0,
                        method='RK45', args=inputs,
                        first_step=3600)

        Tair[i+1] = result.y[0, -1]
        Twall[i+1] = result.y[1, -1]
        Tradiator[i+1] = result.y[2, -1]
        # integral[i+1]         = y[-1][2]

        # Adjust initial condition for next loop

        y0 = result.y[:, -1]

    #Tair = result.y[0, :]
    #Twall = result.y[1, :]
    #Tradiator = result.y[2, :]
    return Tair, Twall, Tradiator, t

