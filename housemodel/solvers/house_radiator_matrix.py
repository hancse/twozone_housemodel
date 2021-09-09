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
    

def model_radiator_m(t, x, cap_mat_inv, cond_mat,
                     T_outdoor, Q_internal, Q_solar, SP_T, CF):
    """model function for scipy.integrate.odeint.

    Args:
        t:          (array):   variable array dependent on time with the vairable Air temperature, Wall temperature Radiator
        x:          (float):
        cap_mat:    (float):  Outdoor temperature in degree C
        cond_mat:   (float):  Internal heat gain in [W]
        T_outdoor:  (float):  Solar irradiation on window [W]
        Q_internal: (float):  Setpoint temperature from thermostat. [C]
        Q_solar:    (float):  Heating power delivered to the buffervessel [W]
        SP_T:
        CF:         (float):  factor of Q_solar heat transferred to the air (Unitless)

    Returns:

    """
    # States :
    Tair = x[0]

    # Parameters :
    # index = int(t/3600)
    # print(t, index)
    #setpointRoomTemperature = SP_T[index]
    setpointRoomTemperature = SP_T[int(t/3600)]

    # Control :
    Qinst = controllerTemperature_m(setpointRoomTemperature, Tair)
 
    # Equations :
    q_vector = np.zeros((3,1))
    q_vector[0,0] = (T_outdoor[int(t/3600)] * 214.9718240562546253373451579691) + Q_internal[int(t/3600)] + CF * Q_solar[int(t/3600)]
    q_vector[1,0] = (1 - CF) * Q_solar[int(t/3600)]
    q_vector[2,0] = Qinst

    x = np.array(x)[np.newaxis] #Converts 1D array to a 2D array; https://stackoverflow.com/questions/5954603/transposing-a-1d-numpy-array

    dTdt = (-cond_mat @ x.T)  + q_vector
    dTdt = np.dot(cap_mat_inv, dTdt)

    return dTdt


def house_radiator_m(cap_mat_inv, cond_mat,
                     T_outdoor, Q_internal, Q_solar, SP_T,
                     time_sim, CF):
    """Compute air and wall temperature inside the house.

    Args:
        cap_mat:
        cond_mat:
        T_outdoor_sim: (array):  Outdoor temperature in degree C
        Qinternal_sim: (array):  Internal heat gain in w.
        Qsolar_sim:    (array):  Solar irradiation on window [W]
        SP_sim:        (array):  Setpoint temperature from thermostat.
        time_sim:      (array)  :  simulation time
        CF:

    Returns:
        tuple :  Tuple containing (Tair, Twall, Tradiator):

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

    # Tair = np.ones(len(t)) * Tair0
    # Twall = np.ones(len(t)) * Twall0
    # Tradiator = np.ones(len(t)) * Tradiator0

    inputs = (cap_mat_inv, cond_mat,
              T_outdoor, Q_internal, Q_solar, SP_T, CF)
    # print(T_outdoor[1], Q_internal[1], Q_solar[1], SP_T[1], CF)
    result = solve_ivp(model_radiator_m, [0, t[-1]], y0,
                  method='RK45', t_eval= time_sim,
                  args=inputs)

    Tair = result.y[0, :]
    Twall = result.y[1, :]
    Tradiator = result.y[2, :]
    return Tair, Twall, Tradiator, result.t

