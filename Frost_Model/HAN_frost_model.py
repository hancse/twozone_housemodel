
import numpy as np
import CoolProp.CoolProp as CP
from scipy.optimize import minimize
from housemodel.tools.radiator_performance.TemperatureDifference import LMTD
import math

CP.set_config_bool(CP.DONT_CHECK_PROPERTY_LIMITS, True)


class FrostModel:
    def __init__(self, K, A_u, fin_separation):
        self.K = K  # Heat transfer coefficient
        self.A_u = A_u  # Evaporator surface area in square meters
        self.fin_separation = fin_separation  # Fin separation in mm
        self.T_0 = T_outside - 20  # Initial guess for T_0 based on outside temperature

    def calculate_moisture_content(self, T_outside, RV, P):
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

    def calculate_average_cp(self, h_ai, h_0, T_a, T_0):
        return (h_ai - h_0) / (T_a - T_0)

    def calculate_log_mean_enthalpy_matlab(self, h_ai, h_ao, h_0):
        delta_h1 = h_ai - h_0
        delta_h2 = h_ao - h_0
        asdf1 = (delta_h1/ delta_h2)
        asdf2 = math.log(delta_h1 / delta_h2)
        # Calculate the logarithmic mean enthalpy difference
        lnH = (delta_h1 - delta_h2) / math.log(delta_h1/ delta_h2)
        return lnH

    def calculate_power(self, delta_h_ln, average_cp):
        return (self.K * self.A_u * delta_h_ln) / average_cp

    def find_evaporator_temperature(self, Q_r, h_ai, h_ao):
        def objective_function(T_0):
            self.T_0 = T_0
            delta_h_a = self.calculate_enthalpy_change(Q_r, m_a)
            h_ao = h_ai - delta_h_a
            humidity_ratio_at_surface = FM.calculate_moisture_content(T_0, 1, P)
            h_0 = self.calculate_enthalpy_moist_air(T_0, P, humidity_ratio_at_surface)
            cp = self.calculate_average_cp(h_ai, h_0, T_a, T_0)
            delta_h_ln = self.calculate_log_mean_enthalpy_matlab(h_ai, h_ao, h_0)
            Q_calculated = self.calculate_power(delta_h_ln, cp)
            asdf = abs(Q_calculated - Q_r)*1000
            return asdf

        initial_guess = self.T_0
        result = minimize(objective_function, initial_guess, method='Nelder-Mead')

        if result.success:
            self.T_0 = result.x[0]
        else:
            raise ValueError("Optimization did not converge")
        return self.T_0

    def find_evaporator_temperature_matlab(self, Q_r, h_ai, h_ao):
        while True:
            enthalpy_air = h_ai
            enthalpy_air_out = h_ao
            humidity_ratio_at_surface = FM.calculate_moisture_content(self.T_0, 1, P)
            P_sat = CP.PropsSI('P', 'T', self.T_0, 'Q', 0, 'Water')
            enthalpy_at_surface = 0.001 * CP.HAPropsSI("H", "T", self.T_0, "P", P_sat, "W",
                                                       humidity_ratio_at_surface)
            average_cp = FM.calculate_average_cp(enthalpy_air, enthalpy_at_surface, T_outside, self.T_0)

            ln_enthalpy_difference = FM.calculate_log_mean_enthalpy_matlab(enthalpy_air, enthalpy_air_out,
                                                                           enthalpy_at_surface)
            Q_estimated = ((self.K * self.A_u * ln_enthalpy_difference) / average_cp)/1000

            if (Q_r - 0.010) <= Q_estimated <= (Q_r + 0.010):
                break

            if Q_r > Q_estimated:
                self.T_0 = self.T_0 - 0.001
            else:
                self.T_0 = self.T_0 + 0.001
        return self.T_0, Q_estimated

    def calculate_x_air_leaving(self, P, enthalpy_air_leaving,  RV, T_outside):
        x_at_surface = FM.calculate_moisture_content(self.T_0, 1, P)
        x_now = self.calculate_moisture_content(T_outside, RV, P)
        P_sat = CP.PropsSI('P', 'T', self.T_0, 'Q', 0, 'Water')
        enthalpy_at_surface = 0.001 * CP.HAPropsSI("H", "T", self.T_0, "P", P_sat, "W", x_at_surface)
        enthalpy_air_now = self.calculate_enthalpy_moist_air(T_outside, P, x_now)
        return x_at_surface + (enthalpy_air_leaving - enthalpy_at_surface) * (
                    (x_now - x_at_surface) / (enthalpy_air_now - enthalpy_at_surface))

    def calculate_mrijp(self, x_now, x_air_leaving, mdotair, timeinterval):
        return ((x_now - x_air_leaving) * mdotair * timeinterval)

    def calculate_t_dew_quick(self, t, RV, t0):
        return t - ((100 - RV) / 5)

    # Calculate dew point
    def calculate_dew_point(self, T_outside, pressure, relative_humidity):
       return  CP.HAPropsSI('D', 'T', T_outside, 'P', pressure, 'R', relative_humidity)

    def calculate_dew_point_quick(self, T_outside, relative_humidity):
       return  T_outside-((1-relative_humidity)*100/5)

    def calculate_rho_rijp(self, T_dew):
        return 260 - (15 * (T_dew - (self.T_0-273.15)))


if __name__ == "__main__":
    # Example usage
    P = 101325
    T_outside = 6 + 273.15
    RV = 0.8
    K = 44
    A_u = 16.0
    fin_separation = 4
    evaporator_power = 3.7
    massflow_air = 1928/3600
    time_interval = 3600

    FM = FrostModel(K, A_u, fin_separation)
    humidity_ratio = FM.calculate_moisture_content(T_outside, RV, P)
    print(f'Humidity Ratio: {humidity_ratio} kg_water/kg_dry_air')

    enthalpy_air = FM.calculate_enthalpy_moist_air(T_outside, P, humidity_ratio)
    print(f'Enthalpy air: {enthalpy_air} kJ/kg')

    extracted_enthalpy = FM.calculate_enthalpy_change(evaporator_power, massflow_air)
    print(f'Extracted enthalpy: {extracted_enthalpy} kJ/kg')

    enthalpy_air_out = enthalpy_air - extracted_enthalpy
    print(f'Enthalpy out: {enthalpy_air_out} kJ/kg')

    humidity_ratio_at_surface = FM.calculate_moisture_content(FM.T_0, 1, P)
    print(f'Humidity Ratio at surface: {humidity_ratio_at_surface} kg_water/kg_dry_air')

    P_sat = CP.PropsSI('P', 'T', T_outside, 'Q', 0, 'Water')
    print(f'P_sat: {P_sat} Pascal')
    enthalpy_at_surface = 0.001 * CP.HAPropsSI("H", "T", FM.T_0, "P", P_sat, "W", humidity_ratio_at_surface)
    print(f'Enthalpy at surface: {enthalpy_at_surface} kJ/kg')

    average_cp = FM.calculate_average_cp(enthalpy_air, enthalpy_at_surface, T_outside, T_outside-5)
    print(f'Avergae Cp: {average_cp} kJ/kgK')

    ln_enthalpy_difference = FM.calculate_log_mean_enthalpy_matlab(enthalpy_air, enthalpy_air_out, enthalpy_at_surface)
    print(f'LMED: {ln_enthalpy_difference} kJ/kg')

    [estimated_evaporator_temperature, estimated_power] = FM.find_evaporator_temperature_matlab(evaporator_power, enthalpy_air, enthalpy_air_out)
    print(f'Estimated Power: {estimated_power} kW')
    print(f'Estimated evaporator temperature: {estimated_evaporator_temperature} K')

    print(f'Estimated evaporator temperature: {estimated_evaporator_temperature-273.15} C')

    x_air_leaving = FM.calculate_x_air_leaving(P, enthalpy_air_out, RV, T_outside)
    print(f'Humidity ratio air out: {x_air_leaving} kg/s')

    mass_rijp = FM.calculate_mrijp(humidity_ratio, x_air_leaving, massflow_air, time_interval)
    print(f'Kg rijp: {mass_rijp} kg')

    dewpoint = FM.calculate_dew_point_quick(T_outside - 273.15, RV)
    print(f'Dewpoint: {dewpoint} K')

    rho_rijp = FM.calculate_rho_rijp(dewpoint)
    print(f'Rho rijp: {rho_rijp} Kg/m^3')


