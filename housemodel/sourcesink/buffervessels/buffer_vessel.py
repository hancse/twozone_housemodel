from scipy.integrate import solve_ivp  # ODE solver
import numpy as np  # linear algebra
import matplotlib

matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from housemodel.sourcesink.buffervessels import cp_water, rho_water, lambda_water

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
        self.temperatures = None
        self.mass_water_layer = (self.volume / self.n_layers) * rho_water

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

        mdote = mdots - mdotd

        if mdote > 0:
            deltaPlus = 1
        else:
            deltaPlus = 0

        if mdote < 0:
            deltaMinus = 1
        else:
            deltaMinus = 0

        dT1 = ((mdots * cp_water * (Tsupply - x[0])) + (mdote * cp_water *(x[0] - x[1]) * deltaMinus) - (self.uwall * (self.Abase + self.Awall_layer) * (x[0]- Tamb)) + ((self.Abase * lambda_water) / self.layer_height) * (x[0] - x[1])) / (self.mass_water_layer*cp_water)
        dT2 = ((mdote * cp_water * (x[0] - x[1]) * deltaPlus) + (mdote * cp_water * (x[1] - x[2]) * deltaMinus) - (self.uwall * self.Awall_layer * (x[1]- Tamb)) + ((self.Abase * lambda_water) / self.layer_height) * (x[0] + x[2] - (2*x[1]))) / (self.mass_water_layer*cp_water)
        dT3 = ((mdote * cp_water * (x[1] - x[2]) * deltaPlus) + (mdote * cp_water * (x[2] - x[3]) * deltaMinus) - (self.uwall * self.Awall_layer * (x[2]- Tamb)) + ((self.Abase * lambda_water) / self.layer_height) * (x[1] + x[3] - (2*x[2]))) / (self.mass_water_layer*cp_water)
        dT4 = ((mdote * cp_water * (x[2] - x[3]) * deltaPlus) + (mdote * cp_water * (x[3] - x[4]) * deltaMinus) - (self.uwall * self.Awall_layer * (x[3]- Tamb)) + ((self.Abase * lambda_water) / self.layer_height) * (x[2] + x[4] - (2*x[3]))) / (self.mass_water_layer*cp_water)
        dT5 = ((mdote * cp_water * (x[3] - x[4]) * deltaPlus) + (mdote * cp_water * (x[4] - x[5]) * deltaMinus) - (self.uwall * self.Awall_layer * (x[4]- Tamb)) + ((self.Abase * lambda_water) / self.layer_height) * (x[3] + x[5] - (2*x[4]))) / (self.mass_water_layer*cp_water)
        dT6 = ((mdote * cp_water * (x[4] - x[5]) * deltaPlus) + (mdote * cp_water * (x[5] - x[6]) * deltaMinus) - (self.uwall * self.Awall_layer * (x[5]- Tamb)) + ((self.Abase * lambda_water) / self.layer_height) * (x[4] + x[6] - (2*x[5]))) / (self.mass_water_layer*cp_water)
        dT7 = ((mdote * cp_water * (x[5] - x[6]) * deltaPlus) + (mdote * cp_water * (x[6] - x[7]) * deltaMinus) - (self.uwall * self.Awall_layer * (x[6]- Tamb)) + ((self.Abase * lambda_water) / self.layer_height) * (x[5] + x[7] - (2*x[6]))) / (self.mass_water_layer*cp_water)
        dT8 = ((mdotd * cp_water * (Treturn - x[7])) + (mdote * cp_water * (x[6] - x[7]) * deltaPlus) - (self.uwall * self.Awall_layer * (x[7] - Tamb)) + ((self.Abase * lambda_water) / self.layer_height) * (x[6] - x[7])) / (self.mass_water_layer*cp_water)

        return [dT1, dT2, dT3, dT4, dT5, dT6, dT7, dT8]

if __name__ == "__main__":

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



