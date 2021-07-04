"""
house model base on 2R2C model with a buffervessel and a radiator
"""

from scipy.integrate import solve_ivp       # ODE solver
import numpy as np                       # linear algebra

def controllerTemperatureandBuffervessel(setpointTemperature, setpointBuffervessel, Tair, Tbuffervessel):
    errorbuffervessel = setpointBuffervessel - Tbuffervessel
    errorroomtemperature = setpointTemperature - Tair
    
    Qinst = errorbuffervessel * 300
    Qinst = np.clip(Qinst, 0, 12000)
        
    mdot = np.clip(errorroomtemperature*0.05, 0, 0.15)
    return Qinst, mdot
    
    

def model_buffervessel_matrix(t, T_vector,
                              T_outdoor, Q_internal, Q_solar, SP_T, CF,
                              cap_mat, cond_mat,
                              UAradiator, Crad, Cbuffervessel, cpwater):
    """model function for scipy.integrate.odeint.

    :param x:            (array):   variable array dependent on time with the vairable Air temperature, Wall temperature Return water temperature and buffervessel temperature
    :param t:            (float):
    :param T_outdoor:    (float):  Outdoor temperature in degree C
    :param Q_internal:   (float):  Internal heat gain in [W]
    :param Q_solar:      (float):  Solar irradiation on window [W]
    :param SP_T:         (float):  Setpoint tempearature from thermostat. [C]
    :param Qinst:        (float):  Heating power delivered to the buffervessel [W]
    :param CF:           (float):  factor of Q_solar heat transferred to the air (Unitless)
    :param Rair_outdoor: (float):  Thermal resistance from indoor air to outdoor air [K/W]
    :param Rair_wall:    (float):  Thermal resistance from indoor air to the wall [K/W]
    :param Cair:         (float):  Thermal capacity of the air
    :param Cwall:        (float):  Thermal capacity of the wall
    :param mdot:         (float):  waterflow in the radiator [kg/s]
    :param UAradiator    (float):  Heat transfer coeffiecient of the radiator 
    :return:             (array):  Difference over of the variables in x      

    x,t: ode input function func : callable(x, t, ...) or callable(t, x, ...)
    Computes the derivative of y at t.
    If the signature is ``callable(t, y, ...)``, then the argument tfirst` must be set ``True``.
    """
    index = int(t/3600)
    # States :
    Tair = x[0]
    Twall = x[1]
    Treturn = x[2]
    Tbuffervessel = x[3]
    
    # Parameters :
        
    setpointRoomTemperature = SP_T[index]
    setpointBuffervessel = 80
    
    # Control :
        
    Qinst, mdot = controllerTemperatureandBuffervessel(setpointRoomTemperature, setpointBuffervessel, Tair, Tbuffervessel)
 
    # Equations :
    dTdt = (-1.0 * np.dot(np.linalg.inv(cap_mat), np.dot(cond_mat, T_vector.T)) +
            np.dot(np.linalg.inv(cap_mat), q_vector(index))

    # q_vector = np.zeros
    Tairdt = ((T_outdoor[int(t/3600)] - Tair) / Rair_outdoor + (Twall - Tair) / Rair_wall + UAradiator*(Treturn-Tair) + Q_internal[int(t/3600)] + CF * Q_solar[int(t/3600)]) / Cair
    Twalldt = ((Tair - Twall) / Rair_wall + (1 - CF) * Q_solar[int(t/3600)]) / Cwall
    Treturndt = ((mdot*cpwater*(Tbuffervessel-Treturn)) + UAradiator*(Tair-Treturn)) / Crad
    Tbuffervesseldt = (Qinst + (cpwater*mdot*(Treturn-Tbuffervessel)))/Cbuffervessel
    energydt = Qinst
    

    return [Tairdt, Twalldt, Treturndt, Tbuffervesseldt, energydt]


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

    Tair = np.ones(len(t)) * Tair0
    Twall = np.ones(len(t)) * Twall0
    Treturn = np.ones(len(t)) * Treturn0
    Tbuffervessel = np.ones(len(t)) * Tbuffervessel0

    inputs = (T_outdoor, Q_internal, Q_solar, SP_T, CF,
              cap_mat, cond_mat,
              UAradiator, Crad, Cbuffervessel, cpwater)
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

