"""
house model base on 2R2C model with a buffervessel and a radiator
"""

from scipy.integrate import odeint       # ODE solver
import numpy as np                       # linear algebra

def radiator_power(Q50, Tradiatorin, Tradiatorout, Troom, nrad):
    asdf = Tradiatorout-Troom
    if(asdf<=0):
        radiator = 0
    else:
        radiator=(Q50*((Tradiatorin-Tradiatorout)/((np.log(Tradiatorin-Troom)-np.log(asdf))*49.32))**nrad)
    return radiator

def model_buffervessel_with_radiator(x, t, T_outdoor, Q_internal, Q_solar, SP_T, Qinst, CF, Rair_outdoor, Rair_wall, Cair, Cwall, mdot):
    """model function for scipy.integrate.odeint.

    :param x: variable array dependent on time
    :param t:
    :param T_outdoor:
    :param Q_internal:
    :param Q_solar:
    :param SP_T:
    :param Qinst:
    :param CF:
    :param Rair_outdoor:
    :param Rair_wall:
    :param Cair:
    :param Cwall:
    :param mdot:
    :return:

    x,t: ode input function func : callable(x, t, ...) or callable(t, x, ...)
    Computes the derivative of y at t.
    If the signature is ``callable(t, y, ...)``, then the argument tfirst` must be set ``True``.
    """

    # States :

    Tair = x[0]  # Temperature Buffer Tank (K)
    Twall = x[1]  # Return Temperature to Floor (K)
    Treturn = x[2]
    Tbuffervessel = x[3]
    
    #Parameters that should be in the dict
    Urad = 30
    Arad = 10
    cpwater = 4180
    volumeRadiator = 50
    Crad = volumeRadiator * 4180
    volumeBuffervessel = 0.150
    rhowater = 1000
    Cbuffervessel = cpwater*volumeBuffervessel*rhowater
    Q50 = 7000
    nrad = 1.33
    
    # Equations :
        
    Tairdt = ((T_outdoor - Tair) / Rair_outdoor + (Twall - Tair) / Rair_wall + radiator_power(Q50, Tbuffervessel, Treturn, Tair, nrad) + Q_internal + CF * Q_solar) / Cair
    Twalldt = ((Tair - Twall) / Rair_wall + (1 - CF) * Q_solar) / Cwall
    Treturndt = ((mdot*cpwater*(Tbuffervessel-Treturn)) - radiator_power(Q50, Tbuffervessel, Treturn, Tair, nrad)) / Crad
    Tbuffervesseldt = (Qinst + (cpwater*mdot*(Treturn-Tbuffervessel)))/Cbuffervessel

    return [Tairdt, Twalldt, Treturndt, Tbuffervesseldt]


def house_buffervessel_with_radiator(T_outdoor, Q_internal, Q_solar, SP_T, time_sim, CF,
          Rair_outdoor, Rair_wall, Cair, Cwall):
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

    Tair0 = 20
    Twall0 = 20
    Treturn0 = 40
    Tbuffervessel0 = 60

    y0 = [Tair0, Twall0, Treturn0, Tbuffervessel0]

    t = time_sim           # Define Simulation time with sampling time

    Tair = np.ones(len(t)) * Tair0
    Twall = np.ones(len(t)) * Twall0
    Treturn = np.ones(len(t)) * Treturn0
    Tbuffervessel = np.ones(len(t)) * Tbuffervessel0
    consumption = np.ones(len(t))
    radiatorpower = np.ones(len(t))
    kp = 300
    
    for i in range(len(t)-1):

        err = 80 - Tbuffervessel[i]
        err2 = SP_T[i+1] - Tair[i]
        Qinst = err * kp
        Qinst = np.clip(Qinst, 0, 12000)
        
        if (err2 > 0):
            mdot = 0.1
        else:
            mdot = 0

        inputs = (T_outdoor[i], Q_internal[i], Q_solar[i], SP_T[i], Qinst, CF,
                  Rair_outdoor, Rair_wall, Cair, Cwall, mdot)
        ts = [t[i], t[i+1]]
        y = odeint(model_buffervessel_with_radiator, y0, ts, args=inputs)

        Tair[i+1] = y[-1][0]
        Twall[i+1] = y[-1][1]
        Treturn[i+1] = y[-1][2]
        Tbuffervessel[i+1] = y[-1][3]

        # Adjust initial condition for next loop
        y0 = y[-1]
        consumption[i] = Qinst
        radiatorpower[i] = radiator_power(12000, 80, Treturn[i+1], Tair[i+1], 1.33)
    return Tair, Twall, Treturn, Tbuffervessel, consumption, radiatorpower

