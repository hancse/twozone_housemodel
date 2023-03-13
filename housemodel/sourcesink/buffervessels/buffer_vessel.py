from scipy.integrate import solve_ivp  # ODE solver
import numpy as np  # linear algebra
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

cp_water = 4190
rho_water = 1000
lambda_water = 0.644


class StratifiedBuffer:
    """parent class for cylindrical stratified buffer vessel

    """

    def __init__(self, volume, height, n_layers=5, hot_node=0, u=0.12, Tamb=20):
        # super.__init__()
        self.n_layers = n_layers
        self.hot_node = hot_node  # anchor point of hot water supply to house model
        self.cold_node = n_layers  # anchor point of cold water return from house model
        self.volume = volume
        self.height = height
        self.uwall = u
        self.Tamb = Tamb
        self.layer_height = self.height / self.n_layers
        self.radius = np.sqrt(self.volume / (self.height * np.pi))
        self.ratio = self.height / (self.radius * 2)
        self.Awall = 2 * np.pi * self.radius * self.height
        self.Awall_layer = self.Awall / n_layers
        self.Abase = self.radius ** 2 * np.pi
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
        self.Awall = 2 * np.pi * self.radius * self.height

    def model_stratified_buffervessel(self, t, x, Tsupply, Treturn, mdots, mdotd):
        """

        Args:
            t:
            x:
            Tsupply:
            Treturn:
            mdots:
            mdotd:

        Returns:
            object:
        """
        mdote = mdots - mdotd
        n = self.n_layers - 1

        if mdote > 0:
            deltaPlus = 1
        else:
            deltaPlus = 0

        if mdote < 0:
            deltaMinus = 1
        else:
            deltaMinus = 0

        dT = np.zeros(len(x))

        # Equation for the top layer of the buffervessel
        dT[0] = ((mdots * cp_water * (Tsupply - x[0])) + (mdote * cp_water * (x[0] - x[1]) * deltaMinus) - (
                    self.uwall * (self.Abase + self.Awall_layer) * (x[0] - self.Tamb)) + (
                             (self.Abase * lambda_water) / self.layer_height) * (x[0] - x[1])) / (
                            self.mass_water_layer * cp_water)

        # Equation for the middle layers of the buffervessel
        for i in range(len(dT) - 2):
            dT[i + 1] = ((mdote * cp_water * (x[i] - x[i + 1]) * deltaPlus) + (
                        mdote * cp_water * (x[i + 1] - x[i + 2]) * deltaMinus) - (
                                 self.uwall * self.Awall_layer * (x[i + 1] - self.Tamb)) + (
                                 (self.Abase * lambda_water) / self.layer_height) * (
                                     x[i] + x[i + 2] - (2 * x[i + 1]))) / (
                                self.mass_water_layer * cp_water)

        # Equation for the bottom layer of the buffervessel
        dT[n] = ((mdotd * cp_water * (Treturn - x[n])) + (mdote * cp_water * (x[n - 1] - x[n]) * deltaPlus) - (
                    self.uwall * self.Awall_layer * (x[n] - self.Tamb)) + (
                             (self.Abase * lambda_water) / self.layer_height) * (x[n - 1] - x[n])) / (
                            self.mass_water_layer * cp_water)

        return dT

    def model2_stratified_buffervessel(self, t, x, Tsupply, Treturn, mdots, mdotd):
        """arranged differently

        """
        mdote = mdots - mdotd
        # changed definition of index n
        n = self.n_layers

        if mdote > 0:
            deltaPlus = 1
        else:
            deltaPlus = 0

        if mdote < 0:
            deltaMinus = 1
        else:
            deltaMinus = 0

        dT = np.zeros(len(x))

        supply_flow = mdots * cp_water
        between_flow = mdote * cp_water
        demand_flow = mdotd * cp_water
        leak_amb_tb = self.uwall * (self.Abase + self.Awall_layer)
        leak_amb_m = self.uwall * self.Awall_layer
        cond_between = self.Abase * lambda_water / self.layer_height
        cap_lay = self.mass_water_layer * cp_water

        # Equation for the top layer of the buffervessel
        dT[0] = supply_flow * (Tsupply - x[0])
        dT[0] += between_flow * (x[0] - x[1]) * deltaMinus
        dT[0] -= leak_amb_tb * (x[0] - self.Tamb)
        # corrected error in next line
        dT[0] += cond_between * (x[1] - x[0])
        dT[0] /= cap_lay

        # Equation for the middle layers of the buffervessel
        # changed loop range
        for i in range(1, n-1):
            dT[i] = between_flow * (x[i-1] - x[i]) * deltaPlus
            dT[i] += between_flow * (x[i] - x[i+1]) * deltaMinus
            dT[i] -= leak_amb_m * (x[i] - self.Tamb)
            dT[i] += cond_between * (x[i-1] + x[i+1] - (2 * x[i]))
            dT[i] /= cap_lay

        # Equation for the bottom layer of the buffervessel
        dT[n-1] = demand_flow * (Treturn - x[n-1])
        dT[n-1] += (between_flow * (x[n-2] - x[n-1]) * deltaPlus)
        dT[n-1] -= leak_amb_tb * (x[n-1] - self.Tamb)
        dT[n-1] += cond_between * (x[n-2] - x[n-1])
        dT[n-1] /= cap_lay

        return dT


if __name__ == "__main__":
    test = StratifiedBuffer(5, 2.5, 8)
    As = test.Awall_layer
    Aq = test.Abase
    mass_water = test.mass_water_layer
    z = test.layer_height

    Tsupply = 62.5
    Treturn = 50
    mdots = 0
    mdotd = 1  # kg/s

    leak_to_amb_top = test.uwall*(test.Abase+test.Awall_layer)
    leak_to_amb_mid = test.uwall*test.Awall_layer
    layer_to_layer = test.Abase*lambda_water/test.layer_height
    cap_layer = test.mass_water_layer*cp_water
    print(f"{leak_to_amb_top}, {leak_to_amb_mid}, {layer_to_layer}, {cap_layer}, "
          f"{mdots*cp_water}, {mdotd*cp_water}")

    initial_condition = np.ones(test.n_layers) * 80
    inputs = (Tsupply, Treturn, mdots, mdotd)
    result = solve_ivp(test.model2_stratified_buffervessel, [0, 3600 * 2], initial_condition, args=inputs)

    plt.figure(figsize=(10, 5))
    for i in range(len(result.y)):
        plt.plot(result.t, result.y[i, :], label=f'$T_{i}$')
    plt.legend(loc='best')
    plt.title("Stratified Buffervessel Simulation")
    plt.show()
