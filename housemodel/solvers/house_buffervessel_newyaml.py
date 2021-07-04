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
    

def model_buffervessel(t, x, T_outdoor, Q_internal, Q_solar, SP_T, Rair_outdoor, Rair_wall, Cair, Cwall, UAradiator, Crad, Cbuffervessel, cpwater):
    """model function for scipy.integrate.odeint.

    :param x:            (array):   variable array dependent on time with the vairable Air temperature, Wall temperature Return water temperature and buffervessel temperature
    :param t:            (float):
    :param T_outdoor:    (float):  Outdoor temperature in degree C
    :param Q_internal:   (float):  Internal heat gain in [W]
    :param Q_solar:      (float):  Solar irradiation on window [W]
    :param SP_T:         (float):  Setpoint tempearature from thermostat. [C]
    :param Rair_outdoor: (float):  Thermal resistance from indoor air to outdoor air [K/W]
    :param Rair_wall:    (float):  Thermal resistance from indoor air to the wall [K/W]
    :param Cair:         (float):  Thermal capacity of the air
    :param Cwall:        (float):  Thermal capacity of the wall
    :param UAradiator    (float):  Heat transfer coeffiecient of the radiator 
    :return:             (array):  Difference over of the variables in x      

    x,t: ode input function func : callable(x, t, ...) or callable(t, x, ...)
    Computes the derivative of y at t.
    If the signature is ``callable(t, y, ...)``, then the argument tfirst` must be set ``True``.
    """

    # States :
    Tair = x[0]
    Twall = x[1]
    Treturn = x[2]
    Tbuffervessel = x[3]
    
    # Parameters :
    setpointRoomTemperature = SP_T[int(t/3600)]
    setpointBuffervessel = 80
    
    # Control :
    Qinst, mdot = controllerTemperatureandBuffervessel(setpointRoomTemperature, setpointBuffervessel, Tair, Tbuffervessel)
 
    # Equations :
    Tairdt = ((T_outdoor[int(t/3600)] - Tair) / Rair_outdoor + (Twall - Tair) / Rair_wall + UAradiator*(Treturn-Tair) + Q_internal[int(t/3600)] + Q_solar[0, int(t/3600)]) / Cair

    Twalldt = ((Tair - Twall) / Rair_wall + Q_solar[1, int(t/3600)]) / Cwall

    Treturndt = ( (mdot * cpwater * (Tbuffervessel - Treturn)) + UAradiator * (Tair - Treturn) ) / Crad

    Tbuffervesseldt = ( Qinst + (cpwater * mdot * (Treturn - Tbuffervessel)) ) / Cbuffervessel

    energydt = Qinst

    if (t/3600 % 1000) < 1:
        print(int(t/3600))

    return [Tairdt, Twalldt, Treturndt, Tbuffervesseldt, energydt]


def house_buffervessel(T_outdoor, Q_internal, Q_solar, SP_T, time_sim,
          Rair_outdoor, Rair_wall, Cair, Cwall, UAradiator, Crad, Cbuffervessel, cpwater):
    """Compute air and wall tempearature inside the house.

    :param T_outdoor:    (array):  Outdoor temperature in degree C
    :param Q_internal:   (array):  Internal heat gain in w.
    :param Q_solar:      (array):  Solar irradiation on window [W]
    :param SP_T:         (array):  Setpoint tempearature from thermostat.
    :param time_sim:     (array)  :  simulation time

    :param Rair_outdoor:
    :param Rair_wall:
    :param Cair:
    :param Cwall:
    :return:             tuple :  Tuple containing (Tair, Twall):

                - Tair (float):   air temperature inside the house in degree C.
                - Twall (float):  wall temperature inside the house in degree C

    Qinst ?	  (array):  instant heat from heat source such as HP or boiler [W].

    """
    # initial values for solve_ivp
    Tair_0 = 20
    Twall_0 = 20
    Treturn_0 = 40
    Tbuffervessel_0 = 60
    energy_0 = 0

    y0 = [Tair_0, Twall_0, Treturn_0, Tbuffervessel_0, energy_0]

    inputs = (T_outdoor, Q_internal, Q_solar, SP_T,
              Rair_outdoor, Rair_wall, Cair, Cwall,
              UAradiator, Crad, Cbuffervessel, cpwater)

    result = solve_ivp(model_buffervessel, [0, time_sim[-1]], y0,
                       method='RK45', t_eval= time_sim,
                       args=inputs)

    Tair = result.y[0, :]
    Twall = result.y[1, :]
    Treturn = result.y[2, :]
    Tbuffervessel = result.y[3, :]
    print(result.y[4, -1]/3600000)
    return Tair, Twall, Treturn, Tbuffervessel, result.t


if __name__ == "__main__":
    pass

