"""
house model base on 2R2C model with a buffervessel and a radiator
"""

from scipy.integrate import solve_ivp       # ODE solver
import numpy as np                       # linear algebra

def controllerTemperature(setpointTemperature, Tair, Kp = 7000):
    errorroomtemperature = setpointTemperature - Tair

    Qinst = errorroomtemperature * Kp
    Qinst = np.clip(Qinst, 0, 12000)
        
    return Qinst
    

def model_radiator(t, x, T_outdoor, Q_internal, Q_solar, SP_T, CF, Rair_outdoor, Rair_wall, Cair, Cwall, UAradiator, Crad):
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

    # States :

    Tair = x[0]
    Twall = x[1]
    Tradiator = x[2]

    # Parameters :
        
    setpointRoomTemperature = SP_T[int(t/3600)]

    # Control :
        
    Qinst = controllerTemperature(setpointRoomTemperature, Tair)
 
    # Equations :
        
    Tairdt = ((T_outdoor[int(t/3600)] - Tair) / Rair_outdoor + (Twall - Tair) / Rair_wall + UAradiator*(Tradiator-Tair) + Q_internal[int(t/3600)] + CF * Q_solar[int(t/3600)]) / Cair
    Twalldt = ((Tair - Twall) / Rair_wall + (1 - CF) * Q_solar[int(t/3600)]) / Cwall
    Tradiatordt = (Qinst + UAradiator*(Tair-Tradiator)) / Crad

    return [Tairdt, Twalldt, Tradiatordt]


def house_radiator(T_outdoor, Q_internal, Q_solar, SP_T, time_sim, CF,
          Rair_outdoor, Rair_wall, Cair, Cwall, UAradiator, Crad):
    """Compute air and wall tempearature inside the house.

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
    #initial values for solve_ivp
    Tair0 = 20
    Twall0 = 20
    Tradiator0 = 40

    y0 = [Tair0, Twall0, Tradiator0]

    t = time_sim           # Define Simulation time with sampling time

    # Tair = np.ones(len(t)) * Tair0
    # Twall = np.ones(len(t)) * Twall0
    # Tradiator = np.ones(len(t)) * Tradiator0

    inputs = (T_outdoor, Q_internal, Q_solar, SP_T, CF,
              Rair_outdoor, Rair_wall, Cair, Cwall, UAradiator, Crad)
    result = solve_ivp(model_radiator, [0, t[-1]], y0,
                  method='RK45', t_eval= time_sim,
                  args=inputs)

    Tair = result.y[0, :]
    Twall = result.y[1, :]
    Tradiator = result.y[2, :]
    return Tair, Twall, Tradiator, result.t

