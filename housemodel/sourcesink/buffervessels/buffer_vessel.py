from scipy.integrate import solve_ivp  # ODE solver
import numpy as np  # linear algebra
from math import e
import matplotlib

matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

class StratifiedBuffer():
    """parent class for cylindrical stratified buffer vessel

    """
    def __init__(self, n_layers: int, hot_node, cold_node):
        # super.__init__()
        self.n_layers = n_layers
        self.hot_node = hot_node    # anchor point of hot water supply to house model
        self.cold_node = cold_node  # anchor point of cold water return from house model
        self.ratio = None
        self.volume = None
        self.uwall = None
        self.lambda_water = 0.644  # W/mK
        self.cp_water = 4190       # J/kgK
        self.rho = 1000            # kg/m^3
        self.temperatures = None

    def set_ratio(self, ratio):
        """

        Args:
            ratio: ratio between height and diameter of (cylindrical) vessel

        Returns:
            None
        """
        self.ratio = ratio

    def set_volume(self, vol):
        self.volume = vol

    def set_uwall(self, u):
        self.uwall = u

    def set_temperatures(self, T):
        T = np.asarray(T)
        # check if length of array equals n_layers!
        self.temperatures = T


    def model_buffervessel(self, t, x, Pin, U, A, T_amb, rho, volume, cp, flux, Twaterin):
        """single-layer cylindrical buffer vessel

        Args:
            x:
            Pin:
            U:
            A:
            T_amb:
            rho:
            volume:
            cp:
            flux:
            Twaterin:

        Returns:

        """
        """model function for scipy.integrate.odeint.

    :param x:            (array):   variable array dependent on time with the vairable Air temperature, Wall temperature Return water temperature and buffervessel temperature
    :param t:            (float):
    :param Pin:          (float):  Power input in [W]
    :param U:            (float):
    :param A:            (float):  Area of
    :param T_amb:        (float):
    :param rho:          (float):
    :param volume:       (float):
    :param cp: (float):  Thermal resistance from indoor air to outdoor air [K/W]

    x,t: ode input function func : callable(x, t, ...) or callable(t, x, ...)
    Computes the derivative of y at t.
    If the signature is ``callable(t, y, ...)``, then the argument tfirst` must be set ``True``.
    """

        # States :

        t_buffervesseldt = (Pin + U * A * (T_amb - x[0]) + (rho * cp * flux * (Twaterin - x[0]))) / (rho * volume * cp)

        return t_buffervesseldt

def model_stratified_buffervessel(t, x, U, As, Aq, Tamb, Tsupply, Treturn, cpwater, lamb, mdots, mdotd, mass_water, z):
    """model function for scipy.integrate.odeint.

    :param x:            (array):   variable array dependent on time with the vairable Air temperature, Wall temperature Return water temperature and buffervessel temperature
    :param t:            (float):
    :param Pin:          (float):  Power input in [W]
    :param U:            (float):
    :param A:            (float):  Area of
    :param T_amb:        (float):
    :param rho:          (float):
    :param volume:       (float):
    :param cp: (float):  Thermal resistance from indoor air to outdoor air [K/W]

    x,t: ode input function func : callable(x, t, ...) or callable(t, x, ...)
    Computes the derivative of y at t.
    If the signature is ``callable(t, y, ...)``, then the argument tfirst` must be set ``True``.
    """

    # me = ms - md

    if (t/3600) > 11:
        mdots = 2400000/(4190*(80-x[7]))
        mdotd = 0
        #mdote = 10
        #mdots = 10
        #mdotd = 0

    if (t/3600) > 15:
        mdots = 0
        mdotd = 10

    if(t/3600) > 17.5:
        mdots = 2400000/(4190*(80-x[7]))
        mdotd = 10


    # States :
    mdote = mdots - mdotd


    if mdote > 0:
        deltaPlus = 1
    else:
        deltaPlus = 0

    if mdote < 0:
        deltaMinus = 1
    else:
        deltaMinus = 0




    dT1 = ((mdots * cpwater * (Tsupply - x[0])) + (mdote *cpwater*(x[0] - x[1]) * deltaMinus) - (U * As * (x[0]- Tamb)) + ((Aq * lamb) / z) * (x[0] - x[1])) / (mass_water*cpwater)
    dT2 = ((mdote *cpwater*(x[0] - x[1]) * deltaPlus) + (mdote *cpwater*(x[1] - x[2]) * deltaMinus) - (U * As * (x[1]- Tamb)) + ((Aq * lamb) / z) * (x[0] + x[2] - (2*x[1]))) / (mass_water*cpwater)
    dT3 = ((mdote *cpwater*(x[1] - x[2]) * deltaPlus) + (mdote *cpwater*(x[2] - x[3]) * deltaMinus) - (U * As * (x[2]- Tamb)) + ((Aq * lamb) / z) * (x[1] + x[3] - (2*x[2]))) / (mass_water*cpwater)
    dT4 = ((mdote *cpwater*(x[2] - x[3]) * deltaPlus) + (mdote *cpwater*(x[3] - x[4]) * deltaMinus) - (U * As * (x[3]- Tamb)) + ((Aq * lamb) / z) * (x[2] + x[4] - (2*x[3]))) / (mass_water*cpwater)
    dT5 = ((mdote *cpwater*(x[3] - x[4]) * deltaPlus) + (mdote *cpwater*(x[4] - x[5]) * deltaMinus) - (U * As * (x[4]- Tamb)) + ((Aq * lamb) / z) * (x[3] + x[5] - (2*x[4]))) / (mass_water*cpwater)
    dT6 = ((mdote *cpwater*(x[4] - x[5]) * deltaPlus) + (mdote *cpwater*(x[5] - x[6]) * deltaMinus) - (U * As * (x[5]- Tamb)) + ((Aq * lamb) / z) * (x[4] + x[6] - (2*x[5]))) / (mass_water*cpwater)
    dT7 = ((mdote *cpwater*(x[5] - x[6]) * deltaPlus) + (mdote *cpwater*(x[6] - x[7]) * deltaMinus) - (U * As * (x[6]- Tamb)) + ((Aq * lamb) / z) * (x[5] + x[7] - (2*x[6]))) / (mass_water*cpwater)
    dT8 = ((mdotd * cpwater * (Treturn - x[7])) + (mdote * cpwater * (x[6] - x[7]) * deltaPlus) - (U * As * (x[7] - Tamb)) + ((Aq * lamb) / z) * (x[6] - x[7])) / (mass_water*cpwater)

    return [dT1, dT2, dT3, dT4, dT5, dT6, dT7, dT8]


def buffervessel(Pin, U, A, T_amb, rho, volume, cp, flux, Twaterin):
    """Compute air and wall tempearature inside the house.

    :param T_outdoor:    (array):  Outdoor temperature in degree C
    :param Q_internal:   (array):  Internal heat gain in w.
    :param Q_solar:      (array):  Solar irradiation on window [W]
    :param SP_T:         (array):  Setpoint tempearature from thermostat.
    :param time_sim:     (array)  :  simulation time

    :return:             tuple :  Tuple containing (Tbuffeervessel):

    """
    # initial values for odeint
    inputs = (Pin, U, A, T_amb, rho, volume, cp, flux, Twaterin)
    result = solve_ivp(model_buffervessel, [0, 1000], [15], args=inputs)
    Tbuffervessel = result.y[0, :]
    return [result.t, Tbuffervessel]

def stratified_buffervessel(U, As, Aq, Tamb, Tsupply, Treturn, cpwater, lamb, mdotsupply, mdotd, mass_water, z):
    """Compute air and wall tempearature inside the house.

    :param T_outdoor:    (array):  Outdoor temperature in degree C
    :param Q_internal:   (array):  Internal heat gain in w.
    :param Q_solar:      (array):  Solar irradiation on window [W]
    :param SP_T:         (array):  Setpoint tempearature from thermostat.
    :param time_sim:     (array)  :  simulation time

    :return:             tuple :  Tuple containing (Tbuffeervessel):

    """
    # initial values for odeint
    inputs = (U, As, Aq, Tamb, Tsupply, Treturn, cpwater, lamb, mdotsupply, mdotd, mass_water, z)
    result = solve_ivp(model_stratified_buffervessel, [0, 3600*2], [80, 80, 80, 80, 80, 80, 80, 80], args=inputs)
    return [result.t, result.y[0, :], result.y[1, :], result.y[2, :], result.y[3, :], result.y[4, :], result.y[5, :], result.y[6, :], result.y[7, :]]


if __name__ == "__main__":
    """test = buffervessel(20000, 14, 0.7, 20, 1000, 0.02, 4180, 1 * 10 ** -4, 15)

    calculatedTbuffer = np.zeros_like(test[0])
    for i in range(len(test[0])):
        calculatedTbuffer[i] = 61.9 - (46.9 * e ** ((-5.12 * 10 ** -3) * test[0][i]))

    plt.figure(figsize=(15, 5))  # key-value pair: no spaces
    plt.plot(test[0], test[1], label='Simulatie')
    plt.plot(test[0], calculatedTbuffer, label='Formule')
    plt.legend(loc='best')
    plt.title("Buffervessel Simulation")
    plt.show()"""


    #me = ms - md
    stratified_vessel = stratified_buffervessel(0.12, 0.196, 0.196, 10, 80, 20, 4190, 0.644, 0, 0.02, 150 / 8, 1 / 8)

    plt.figure(figsize=(15, 5))  # key-value pair: no spaces
    plt.plot(stratified_vessel[0]/3600, stratified_vessel[1], label='T1')
    plt.plot(stratified_vessel[0]/3600, stratified_vessel[2], label='T2')
    plt.plot(stratified_vessel[0]/3600, stratified_vessel[3], label='T3')
    plt.plot(stratified_vessel[0]/3600, stratified_vessel[4], label='T4')
    plt.plot(stratified_vessel[0]/3600, stratified_vessel[5], label='T5')
    plt.plot(stratified_vessel[0]/3600, stratified_vessel[6], label='T6')
    plt.plot(stratified_vessel[0]/3600, stratified_vessel[7], label='T7')
    plt.plot(stratified_vessel[0]/3600, stratified_vessel[8], label='T8')
    plt.legend(loc='best')
    plt.title("Stratified Buffervessel Simulation")
    plt.show()

