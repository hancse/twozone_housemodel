import numpy as np
import CoolProp.CoolProp as CP
from scipy import optimize
from housemodel.tools.radiator_performance.TemperatureDifference import LMTD
import math
import numpy as np


CP.set_config_bool(CP.DONT_CHECK_PROPERTY_LIMITS, True)


class FrostModel:
    def __init__(self, K, A_u, fin_separation, T_0):
        self.K = K  # Heat transfer coefficient
        self.A_u = A_u  # Evaporator surface area in square meters
        self.fin_separation = fin_separation  # Fin separation in mm
        self.T_0 = T_0  # Initial guess for T_0 based on outside temperature
        self.P = 101325  # air pressure in  Pascal
        self.Total_frost = 0 # Total frost at current time instance
        self.Total_frost_t = []
        self.mass_per_step = []
        self.timesteps_frost_buildup = 0
        self.total_volume_frost = 0
        self.cumulative_density_frost = 0
        self.average_density_frost = 0

    def calculate_humidity_ratio(self, T_outside, RV, P):
        """
        Calculates the amount of moisture in the air (humidity ratio) using CoolProp.

        Parameters:
        T_outside : float
            Dry bulb temperature in Kelvin.
        RV : float
            Relative humidity as a fraction (e.g., 0.5 for 50%).
        P : float
            Atmospheric pressure in Pascals.

        Returns:
        x : float
            Humidity ratio in kg of water/kg of dry air.
        """
        # Calculate the saturation pressure of water vapor at the given temperature
        x = CP.HAPropsSI('W', 'T', T_outside, 'P', P, 'R', RV)
        return x

    def calculate_enthalpy_moist_air(self, T, P, x):
        """
            Calculates the enthalpy of moist air using CoolProp.

            Parameters:
            T : float
                Dry bulb temperature in Kelvin.
            P : float
                Atmospheric pressure in Pascals.
            x : float
                Humidity ratio

            Returns:
            h : float
                Enthalpy of moist air in kJ/kg.
            """
        # Calculate the enthalpy of the air-vapor mixture
        # Using HAPropsSI function with inputs: "H" for enthalpy, "T" for temperature,
        # "P" for pressure, "W" for humidity ratio
        h = 0.001 * CP.HAPropsSI("H", "T", T, "P", P, "W", x)
        return h

    def calculate_enthalpy_change(self, Q_r, mdot_a):
        """

        Args:
            Q_r: Power extracted from the evaporator by the refrigerant
            mdot_a: Flow rate of the moist air

        Returns:

        """
        return Q_r / mdot_a

    def calculate_average_cp(self, h_ai, h_0, T_a):
        """

        Args:
            h_ai: Enthalpy of outside air in kJ/kg
            h_0:  Enthalpy of air at evaporator surface kJ/kg
            T_a:  Ambient temperature in K

        Returns:
            Average Cp [kJ/(KgK)]
        """
        return (h_ai - h_0) / (T_a - self.T_0)

    def calculate_log_mean_enthalpy_matlab(self, h_ai, h_ao, h_0):
        """

        Args:
            h_ai: Enthalpy of outside air in kJ/kg
            h_ao: Enthalpy of air leaving the evaporator kJ/kg
            h_0: Enthalpy of air at evaporator surface kJ/kg

        Returns:
            The log mean temperature difference in K
        """
        try:
            delta_h1 = h_ai - h_0
            delta_h2 = h_ao - h_0
            # Calculate the logarithmic mean enthalpy difference
            if(not(delta_h1 <= 0 or delta_h2 <=0)):
                lnH = (delta_h1 - delta_h2) / math.log(delta_h1 / delta_h2)
            else:
                lnH = 0
        except Exception as e:
            print(f"An error occurred: {e}")
            raise
        return lnH

    def calculate_power(self, delta_h_ln, average_cp):
        """

        Args:
            delta_h_ln: Logarithmic mean temperature difference in K
            average_cp: Average specific heat capacity in kJ/KgK

        Returns:
            Power taken from the outside air in W
        """
        return (self.K * self.A_u * delta_h_ln) / average_cp

    def find_evaporator_temperature(self, Q_r, h_ai, h_ao):
        def objective_function(T_0):
            self.T_0 = float(T_0)
            enthalpy_air = h_ai
            enthalpy_air_out = h_ao
            humidity_ratio_at_surface = self.calculate_humidity_ratio(self.T_0, 1, P)
            P_sat = CP.PropsSI('P', 'T', self.T_0, 'Q', 0, 'Water')
            enthalpy_at_surface = 0.001 * CP.HAPropsSI("H", "T", self.T_0, "P", P_sat, "W",
                                                       humidity_ratio_at_surface)
            average_cp = self.calculate_average_cp(enthalpy_air, enthalpy_at_surface, T_outside)

            ln_enthalpy_difference = self.calculate_log_mean_enthalpy_matlab(enthalpy_air, enthalpy_air_out,
                                                                           enthalpy_at_surface)
            Q_estimated = ((self.K * self.A_u * ln_enthalpy_difference) / average_cp) / 1000
            estimate = Q_r - Q_estimated
            return estimate

        initial_guess = self.T_0
        result = optimize.newton(objective_function, initial_guess, fprime=None)
        asdf = optimize.fsolve(objective_function, [-5+273])
        humidity_ratio_at_surface = self.calculate_humidity_ratio(self.T_0, 1, P)
        P_sat = CP.PropsSI('P', 'T', self.T_0, 'Q', 0, 'Water')
        enthalpy_at_surface = 0.001 * CP.HAPropsSI("H", "T", self.T_0, "P", P_sat, "W",
                                                   humidity_ratio_at_surface)
        ln_enthalpy_difference = self.calculate_log_mean_enthalpy_matlab(h_ai, h_ao,
                                                                       enthalpy_at_surface)
        Q_estimated = ((self.K * self.A_u * ln_enthalpy_difference) / average_cp) / 1000
        return result, Q_estimated

    def find_evaporator_temperature_matlab(self, Q_r, h_ai, h_ao, T_outside):
        """

        Args:
            Q_r: Power taken from the evaporator in kW
            h_ai: Enthalpy of outside air in kJ/kg
            h_ao: Enthalpy of air leaving the evaporator kJ/kg
        Returns:
            Evaporator temperature in K

        """
        enthalpy_air = h_ai
        enthalpy_air_out = h_ao

        # Loop control variables
        tolerance = 0.1
        step_size = 0.001
        max_iterations = 10000  # Prevent infinite loop
        iteration = 0

        while iteration < max_iterations:
            P_sat = CP.PropsSI('P', 'T', self.T_0, 'Q', 0, 'Water')
            humidity_ratio_at_surface = self.calculate_humidity_ratio(self.T_0, 1, self.P)
            enthalpy_at_surface = 0.001 * CP.HAPropsSI("H", "T", self.T_0, "P", P_sat, "W",
                                                       humidity_ratio_at_surface)
            average_cp = self.calculate_average_cp(enthalpy_air, enthalpy_at_surface, T_outside)
            ln_enthalpy_difference = self.calculate_log_mean_enthalpy_matlab(enthalpy_air, enthalpy_air_out,
                                                                             enthalpy_at_surface)
            Q_estimated = ((self.K * self.A_u * ln_enthalpy_difference) / average_cp) / 1000

            if (Q_r - tolerance) <= Q_estimated <= (Q_r + tolerance):
                break

            # Adjust T_0 with a certain step size
            direction = -1 if Q_r > Q_estimated else 1
            self.T_0 += direction * step_size

            iteration += 1

        if iteration >= max_iterations:
            print("Max iterations reached, solution may not be accurate. Expected value = %.3f, calculated value is %.3f" % (Q_r, Q_estimated))

        return self.T_0, Q_estimated

    def calculate_x_air_leaving(self, P, h_ao, RV, T_outside):
        """

        Args:
            P: Pressure in Pa
            h_ao: Enthalpy of the leaving air in kJ/kg
            RV: Relative humidity in fraction from 0-1
            T_outside: Outside temperature in K

        Returns:
            Enthaply of the air leaving the evaporator

        """
        x_at_surface = self.calculate_humidity_ratio(self.T_0, 1, P)
        x_now = self.calculate_humidity_ratio(T_outside, RV, P)
        P_sat = CP.PropsSI('P', 'T', self.T_0, 'Q', 0, 'Water')
        enthalpy_at_surface = 0.001 * CP.HAPropsSI("H", "T", self.T_0, "P", P_sat, "W", x_at_surface)
        enthalpy_air_now = self.calculate_enthalpy_moist_air(T_outside, P, x_now)
        return x_at_surface + (h_ao - enthalpy_at_surface) * (
                (x_now - x_at_surface) / (enthalpy_air_now - enthalpy_at_surface))

    def calculate_mrijp(self, x_now, x_air_leaving, mdotair, timeinterval):
        return ((x_now - x_air_leaving) * mdotair * timeinterval)

    def calculate_t_dew_quick(self, t, RV):
        return t - ((100 - RV) / 5)

    # Calculate dew point
    def calculate_dew_point(self, T_outside, pressure, relative_humidity):
        return CP.HAPropsSI('D', 'T', T_outside, 'P', pressure, 'R', relative_humidity)

    def calculate_dew_point_quick(self, T_outside, relative_humidity):
        return T_outside - ((1 - relative_humidity) * 100 / 5)

    def calculate_rho_rijp(self, T_dew):
        return 260 - (15 * (T_dew - (self.T_0 - 273.15)))

    def update(self, T_outside, RV, evaporator_power, massflow_air):
        if evaporator_power > .5:
            if T_outside < 273.15 + 10:
                # First, the enthalpy of the air entering the evaporator (outside air), and the enthalpy leaving the evaporator need to be determined
                humidity_ratio_outside = self.calculate_humidity_ratio(T_outside, RV, self.P)
                enthalpy_air_outside = self.calculate_enthalpy_moist_air(T_outside, self.P, humidity_ratio_outside)
                extracted_enthalpy = self.calculate_enthalpy_change(evaporator_power, massflow_air)
                enthalpy_air_leaving = enthalpy_air_outside - extracted_enthalpy
                self.find_evaporator_temperature_matlab(evaporator_power, enthalpy_air_outside, enthalpy_air_leaving, T_outside)

                if(self.T_0 < 273.15):
                    self.timesteps_frost_buildup += 1
                    x_air_leaving = self.calculate_x_air_leaving(self.P, enthalpy_air_leaving, RV, T_outside)
                    mass_rijp = self.calculate_mrijp(humidity_ratio_outside, x_air_leaving, massflow_air, 600)
                    self.Total_frost += mass_rijp
                    self.Total_frost_t.append(self.Total_frost)
                    self.mass_per_step.append(mass_rijp)
                    dewpoint = self.calculate_dew_point_quick(T_outside - 273.15, RV)
                    rho_rijp = self.calculate_rho_rijp(dewpoint)
                    volume_frost = mass_rijp / rho_rijp
                    self.total_volume_frost += volume_frost
                    self.cumulative_density_frost += rho_rijp
                    self.average_density_frost = self.cumulative_density_frost / self.timesteps_frost_buildup
                else:
                    if (self.Total_frost > 0):
                        energy_ice = self.Total_frost * 334  # kJ
                        energy_from_solar = -evaporator_power * 600
                        energy_from_air = 0
                        if (T_outside > self.T_0):
                            energy_from_air = (44 * self.A_u * (T_outside - self.T_0) * 600) / 1000
                        energy_ice_new = energy_ice - energy_from_solar - energy_from_air
                        self.Total_frost = energy_ice_new / 334
                        if (self.Total_frost < 0):
                            self.Total_frost = 0
                        self.Total_frost_t.append(self.Total_frost)
                        self.mass_per_step.append(0)
                        self.total_volume_frost = self.Total_frost / self.average_density_frost

            else:
                self.T_0 = T_outside
                self.Total_frost_t.append(self.Total_frost)
                self.mass_per_step.append(0)
        elif evaporator_power > 0:
            self.T_0 = T_outside - 5
            self.Total_frost_t.append(self.Total_frost)
            self.mass_per_step.append(0)

        else:
            if(self.Total_frost > 0):
                energy_ice = self.Total_frost * 334 # kJ
                energy_from_solar = -evaporator_power * 600
                energy_from_air = 0
                if(T_outside > self.T_0):
                    energy_from_air = (44 * self.A_u * (T_outside-self.T_0) * 600)/1000
                energy_ice_new = energy_ice - energy_from_solar - energy_from_air
                self.Total_frost = energy_ice_new / 334
                if(self.Total_frost<0):
                    self.Total_frost = 0
                self.Total_frost_t.append(self.Total_frost)
                self.mass_per_step.append(0)
                self.total_volume_frost = self.Total_frost / self.average_density_frost





if __name__ == "__main__":
    # Inputs to the frost model
    P = 101325
    T_outside = 4 + 273.15
    RV = 0.7
    K = 44
    A_u = 30.0
    fin_separation = 4
    evaporator_power = 5
    massflow_air = 0.5
    time_interval = 600

    FM = FrostModel(K, A_u, fin_separation, 273.15 + 3)
    FM.update(T_outside, RV, evaporator_power, massflow_air)
    print(f'Estimated frost buildup: {FM.total_volume_frost} m^3')
    print(f'Estimated panel temperature: {FM.T_0 - 273.15} C')

    T_outside_step = 1
    RV_step = 0.05

    for i in range(5):
        # Calculate new values for each iteration
        T_outside_iter = T_outside - (i * T_outside_step)
        RV_iter = RV + (i * RV_step)

        # Update the frost model with new values
        FM.update(T_outside_iter, RV_iter, evaporator_power, massflow_air)

        # Optionally, print the parameters for this iteration to see the changes
        print(f"Iteration {i + 1}: T_outside={T_outside_iter - 273.15}, T_PVT={FM.T_0 - 273.15}, RV={RV_iter}, evaporator_power={evaporator_power}, frost buildup={FM.mass_per_step[i+1]}")

