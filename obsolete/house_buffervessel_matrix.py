"""
house model base on 2R2C model with a buffervessel and a radiator
"""

from scipy.integrate import solve_ivp       # ODE solver
import numpy as np                       # linear algebra

def control_temp_buffer(setpointTemperature,
                        setpointBuffervessel,
                        Tair, Tbuffervessel):
    """controller for heating of buffervessel

    Args:
        setpointTemperature:   setpoint for air temperature in house
        setpointBuffervessel:  setpoint for temperature in buffervessel
        Tair:                  actual air temperature
        Tbuffervessel:         actuel temperature of buffervessel

    Returns:
        elements (terms) in q_vector
    """
    error_buffervessel = setpointBuffervessel - Tbuffervessel  # SP - PV
    error_roomtemp = setpointTemperature - Tair         # SP - PV
    
    q_inst = error_buffervessel * 300    # kp = 300  [W/K]
    q_inst = np.clip(q_inst, 0, 12000)   # Pmax = 12000 [W]

    mdot = error_roomtemp*0.05   # kp = 0.05 kg/(K s)
    mdot = np.clip(mdot, 0, 0.15)  # mdot_max = 0.15 ks/s

    cp_water = 4200 # J/(kg K)
    mdot_cp = error_roomtemp*(0.05*cp_water)  # kp = 0.05*4200 = 210 [W/K]
    mdot_cp = np.clip(0, 0.15*cp_water)       # Pmax = 630 W

    # return q_inst, mdot
    return q_inst, mdot_cp
    
    

def model_buffervessel_matrix(t, T_vector, SP_T,
                              cap_mat_inv, cond_mat_minus,
                              q_vector):
    """model function for scipy.integrate.solve_ivp.

    Args:
        t:
        T_vector:
        SP_T:
        cap_mat_inv:
        cond_mat_minus:
        q_vector:

    Returns:
        (array):  Differentials of the variables in T_vector
    """
    index = int(t/3600)
    setpointRoomTemperature = SP_T[index]
    setpointBuffervessel = 80

    # Control :
    Qinst, mdot = control_temp_buffer(setpointRoomTemperature,
                                      setpointBuffervessel,
                                      T_vector[0], T_vector[3])
 
    # Equations :
    dTdt = np.dot(cond_mat_minus, T_vector.T) + q_vector(index)
    dTdt = np.dot(cap_mat_inv, dTdt)

    """
    Tairdt = ((T_outdoor[int(t/3600)] - Tair) / Rair_outdoor + (Twall - Tair) / Rair_wall + UAradiator*(Treturn-Tair) + Q_internal[int(t/3600)] + CF * Q_solar[int(t/3600)]) / Cair
    Twalldt = ((Tair - Twall) / Rair_wall + (1 - CF) * Q_solar[int(t/3600)]) / Cwall
    Treturndt = ((mdot*cpwater*(Tbuffervessel-Treturn)) + UAradiator*(Tair-Treturn)) / Crad
    Tbuffervesseldt = (Qinst + (cpwater*mdot*(Treturn-Tbuffervessel)))/Cbuffervessel
    energydt = Qinst
    """

    return dTdt   # [Tairdt, Twalldt, Treturndt, Tbuffervesseldt, energydt]


def house_buffervessel_matrix(T_outdoor, Q_internal, Q_solar, SP_T, time_sim, CF,
                              cap_mat, cond_mat,
                              UAradiator, Crad, Cbuffervessel, cpwater):
    """Compute air and wall temperature inside the house.

    :param T_outdoor:    (array):  Outdoor temperature in degree C
    :param Q_internal:   (array):  Internal heat gain in w.
    :param Q_solar:      (array):  Solar irradiation on window [W]
    :param SP_T:         (array):  Setpoint tempearature from thermostat.
    :param time_sim:     (array)  :  simulation time

    :param CF:
    :param Rair_outdoor:
    :param Rair_wall:
    :param Cair:
    :param Cwall:
    :return:             tuple :  Tuple containing (Tair, Twall):

                - Tair (float):   air temperature inside the house in degree C.
                - Twall (float):  wall temperature inside the house in degree C

    Qinst ?	  (array):  instant heat from heat source such as HP or boiler [W].

    """
    # initial values for odeint
    Tair0 = 20
    Twall0 = 20
    Treturn0 = 40
    Tbuffervessel0 = 60
    energy0 = 0

    y0 = [Tair0, Twall0, Treturn0, Tbuffervessel0, energy0]

    t = time_sim           # Define Simulation time with sampling time

    # Tair = np.ones(len(t)) * Tair0
    # Twall = np.ones(len(t)) * Twall0
    # Treturn = np.ones(len(t)) * Treturn0
    # Tbuffervessel = np.ones(len(t)) * Tbuffervessel0

    inputs = (SP_T, cap_mat_inv, cond_mat_minus,
                              q_vector)

    # y = solve_ivp(model_buffervessel, [0, t[-1]], y0, args=inputs)
    result = solve_ivp(model_buffervessel_matrix, [0, t[-1]], y0,
                  method='RK45', t_eval= time_sim,
                  args=inputs)

    Tair = result.y[0, :]
    Twall = result.y[1, :]
    Treturn = result.y[2, :]
    Tbuffervessel = result.y[3, :]
    print(result.y[4, -1]/3600000)
    return Tair, Twall, Treturn, Tbuffervessel, result.t

