import numpy as np
import CoolProp.CoolProp as CP

class FrostModel:
    def __init__(self, K, A_u, fin_separation):
        self.K = K  # Heat transfer coefficient
        self.A_u = A_u  # Evaporator surface area in square meters
        self.fin_separation = fin_separation  # Fin separation in mm
        self.T_0 = T_outside - 5  # Initial guess for T_0 based on outside temperature

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
        P_sat = CP.PropsSI('P', 'T', T_outside, 'Q', 0, 'Water')

        # Calculate the partial pressure of water vapor
        P_vapor = RV * P_sat

        # Using the formula for humidity ratio: x = 0.62198 * (P_vapor / (P - P_vapor))
        # Where 0.62198 is the ratio of the molecular weight of water to dry air
        x = 0.62198 * (P_vapor / (P - P_vapor))
        return x

    def calculate_cp(self, T, P, x):
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
        h = 0.001* CP.HAPropsSI("H", "T", T, "P", P, "W", x)
        return h

if __name__ == "__main__":
    # Example usage
    P = 101325
    T_outside = 5 + 273.15
    RV = 0.70
    K = 0.01
    A_u = 1.0
    fin_separation = 4

    FM = FrostModel(K, A_u, fin_separation)
    humidity_ratio = FM.calculate_moisture_content(T_outside, RV, P)
    print(f'Humidity Ratio: {humidity_ratio} kg_water/kg_dry_air')

    enthalpy_air = FM.calculate_cp(T_outside, P, humidity_ratio)
    print(f'Enthalpy air: {enthalpy_air} kJ/kg')
