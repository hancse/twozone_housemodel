from scipy.integrate import solve_ivp  # ODE solver
import numpy as np  # linear algebra
import matplotlib

matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

class StratifiedBuffer():
    """parent class for cylindrical stratified buffer vessel

    """
    def __init__(self, n_layers: int, hot_node, cold_node, volume, height, u):
        # super.__init__()
        self.n_layers = n_layers
        self.hot_node = hot_node    # anchor point of hot water supply to house model
        self.cold_node = cold_node  # anchor point of cold water return from house model
        self.volume = volume
        self.height = height
        self.uwall = u
        self.layer_height = self.height / self.n_layers
        self.radius = np.sqrt(self.volume / (self.height * np.pi))
        self.ratio = self.height / (self.radius * 2)
        self.Awall = 2 * np.pi * self.radius * self.height
        self.Awall_layer = self.Awall / n_layers
        self.Abase = self.radius**2 * np.pi
        self.lambda_water = 0.644  # W/mK
        self.cp_water = 4190       # J/kgK
        self.rho = 1000            # kg/m^3
        self.temperatures = None
        self.mass_water = 1000

    def set_volume(self, vol):
        self.volume = vol

    def set_uwall(self, u):
        self.uwall = u

    def set_temperatures(self, T):
        T = np.asarray(T)
        # check if length of array equals n_layers!
        self.temperatures = T

    def set_lateral_surface_area(self):
        self.Awall = 2*np.pi*self.radius * self.height

    def model_buffervessel(self, t, x, Pin, T_amb, flux, Twaterin):
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

        t_buffervesseldt = (Pin + self.u * self.Awall * (T_amb - x[0]) + (self.rho * self.cp_water * flux * (Twaterin - x[0]))) / (self.rho * self.volume * self.cp_water)

        return t_buffervesseldt

    def model_stratified_buffervessel(self, t, x, Tamb, Tsupply, Treturn, mdots, mdotd):

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




        dT1 = ((mdots * self.cp_water * (Tsupply - x[0])) + (mdote *self.cp_water*(x[0] - x[1]) * deltaMinus) - (self.uwall * (self.Abase + self.Awall_layer) * (x[0]- Tamb)) + ((self.Abase * self.lambda_water) / self.layer_height) * (x[0] - x[1])) / (self.mass_water*self.cp_water)
        dT2 = ((mdote *self.cp_water*(x[0] - x[1]) * deltaPlus) + (mdote *self.cp_water*(x[1] - x[2]) * deltaMinus) - (self.uwall * self.Awall_layer * (x[1]- Tamb)) + ((self.Abase * self.lambda_water) / self.layer_height) * (x[0] + x[2] - (2*x[1]))) / (self.mass_water*self.cp_water)
        dT3 = ((mdote *self.cp_water*(x[1] - x[2]) * deltaPlus) + (mdote *self.cp_water*(x[2] - x[3]) * deltaMinus) - (self.uwall * self.Awall_layer * (x[2]- Tamb)) + ((self.Abase * self.lambda_water) / self.layer_height) * (x[1] + x[3] - (2*x[2]))) / (self.mass_water*self.cp_water)
        dT4 = ((mdote *self.cp_water*(x[2] - x[3]) * deltaPlus) + (mdote *self.cp_water*(x[3] - x[4]) * deltaMinus) - (self.uwall * self.Awall_layer * (x[3]- Tamb)) + ((self.Abase * self.lambda_water) / self.layer_height) * (x[2] + x[4] - (2*x[3]))) / (self.mass_water*self.cp_water)
        dT5 = ((mdote *self.cp_water*(x[3] - x[4]) * deltaPlus) + (mdote *self.cp_water*(x[4] - x[5]) * deltaMinus) - (self.uwall * self.Awall_layer * (x[4]- Tamb)) + ((self.Abase * self.lambda_water) / self.layer_height) * (x[3] + x[5] - (2*x[4]))) / (self.mass_water*self.cp_water)
        dT6 = ((mdote *self.cp_water*(x[4] - x[5]) * deltaPlus) + (mdote *self.cp_water*(x[5] - x[6]) * deltaMinus) - (self.uwall * self.Awall_layer * (x[5]- Tamb)) + ((self.Abase * self.lambda_water) / self.layer_height) * (x[4] + x[6] - (2*x[5]))) / (self.mass_water*self.cp_water)
        dT7 = ((mdote *self.cp_water*(x[5] - x[6]) * deltaPlus) + (mdote *self.cp_water*(x[6] - x[7]) * deltaMinus) - (self.uwall * self.Awall_layer * (x[6]- Tamb)) + ((self.Abase * self.lambda_water) / self.layer_height) * (x[5] + x[7] - (2*x[6]))) / (self.mass_water*self.cp_water)
        dT8 = ((mdotd * self.cp_water * (Treturn - x[7])) + (mdote * self.cp_water * (x[6] - x[7]) * deltaPlus) - (self.uwall * self.Awall_layer * (x[7] - Tamb)) + ((self.Abase * self.lambda_water) / self.layer_height) * (x[6] - x[7])) / (self.mass_water*self.cp_water)

        return [dT1, dT2, dT3, dT4, dT5, dT6, dT7, dT8]


#def buffervessel(Pin, U, A, T_amb, rho, volume, cp, flux, Twaterin):
    """Compute air and wall tempearature inside the house.

    :param T_outdoor:    (array):  Outdoor temperature in degree C
    :param Q_internal:   (array):  Internal heat gain in w.
    :param Q_solar:      (array):  Solar irradiation on window [W]
    :param SP_T:         (array):  Setpoint tempearature from thermostat.
    :param time_sim:     (array)  :  simulation time

    :return:             tuple :  Tuple containing (Tbuffeervessel):

    """
    # initial values for odeint
#    inputs = (Pin, U, A, T_amb, rho, volume, cp, flux, Twaterin)
#    result = solve_ivp(model_buffervessel, [0, 1000], [15], args=inputs)
#    Tbuffervessel = result.y[0, :]
#    return [result.t, Tbuffervessel]

#def stratified_buffervessel(U, As, Aq, Tamb, Tsupply, Treturn, cpwater, lamb, mdotsupply, mdotd, mass_water, z):
    """Compute air and wall tempearature inside the house.

    :param T_outdoor:    (array):  Outdoor temperature in degree C
    :param Q_internal:   (array):  Internal heat gain in w.
    :param Q_solar:      (array):  Solar irradiation on window [W]
    :param SP_T:         (array):  Setpoint tempearature from thermostat.
    :param time_sim:     (array)  :  simulation time

    :return:             tuple :  Tuple containing (Tbuffeervessel):

    """
#    # initial values for odeint
#    inputs = (U, As, Aq, Tamb, Tsupply, Treturn, cpwater, lamb, mdotsupply, mdotd, mass_water, z)
#    result = solve_ivp(model_stratified_buffervessel, [0, 3600*2], [80, 80, 80, 80, 80, 80, 80, 80], args=inputs)
#    return [result.t, result.y[0, :], result.y[1, :], result.y[2, :], result.y[3, :], result.y[4, :], result.y[5, :], result.y[6, :], result.y[7, :]]


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
    plt.show()


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
    plt.show()"""

    test = StratifiedBuffer(8, 0, 7, 0.200,  2, 0.12)
    print(test.height, test.radius, test.ratio, test.Awall, test.Abase)
    Tamb = 20
    Tsupply = 80
    Treturn = 20
    mdots = 0
    mdotd = 0
    inputs = (Tamb, Tsupply, Treturn, mdots, mdotd)
    result = solve_ivp(test.model_stratified_buffervessel, [0, 3600 * 2], [80, 80, 80, 80, 80, 80, 80, 80], args=inputs)
    
    plt.figure(figsize=(15, 5))
    plt.plot(result.t, result.y[0, :], label='T1')
    plt.plot(result.t, result.y[1, :], label='T2')
    plt.plot(result.t, result.y[2, :], label='T3')
    plt.plot(result.t, result.y[3, :], label='T4')
    plt.plot(result.t, result.y[4, :], label='T5')
    plt.plot(result.t, result.y[5, :], label='T6')
    plt.plot(result.t, result.y[6, :], label='T7')
    plt.plot(result.t, result.y[7, :], label='T8')
    plt.legend(loc='best')
    plt.title("Stratified Buffervessel Simulation")
    plt.show()



