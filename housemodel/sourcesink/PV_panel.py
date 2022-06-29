
from math import exp

import numpy as np


class PVPanel:
    """"
        class for modeling a PV-panel
    """

    def __init__(self, loc_long, loc_lat, orient_azimuth,orient_incl, area,
                 nom_efficiency, nom_temp, temp_coeff):
        """

        Args:
            loc_long: (float) longitude of the panel location [rad between 0 and pi/2]
            loc_lat: (float) latitude of the panel location [rad between -pi and pi]
            orient_azimuth: (float) azimuth angle of the panel normal vector [rad between -pi and pi, 0=S pi/2 = W]
            orient_incl: (float) inclination angle of the panel [rad, between 0 and pi/2]
            area: (float) surface area in [m2]
            nom_efficiency: (float) nominal efficiency value [# between 0 and 1]
            nom_temp
            temp_coeff
            work_temp
            work_efficiency
        """
        self.loc_long = loc_long
        self.loc_lat = loc_lat
        self.orient_azimuth = orient_azimuth
        self.orient_incl = orient_incl
        self.area = area
        self.nom_efficiency = nom_efficiency
        self.nom_temp = nom_temp
        self.temp_coeff = temp_coeff


    def compute_output_power(self, insol_global, T_amb, wind_speed):
        """

        Args:
            insol_global: global insolation on the oriented surface (sum of direct on the surface and diffuse on the
            surface)
            T_amb: measured ambient temp (as by NEN)
            wind_speed: measured wind speed (as by NEN)

        Returns:

        """
        insol_global = np.asarray(insol_global)
        T_amb = np.asarray(T_amb)
        wind_speed = np.asarray(wind_speed)

        scalar_input = False  # assume array input, unless...
        if insol_global.ndim == 0:  # first input was converted to a 0D array (= scalar)
            insol_global = insol_global[np.newaxis]  # then, convert oD array to 1D array
            scalar_input = True  # and set the scalar flag!

        if scalar_input:  # if the first argument was a scalar...
            T_amb = T_amb[np.newaxis]  # convert the other 0D arrays to 1D arrays
            wind_speed = wind_speed[np.newaxis]

        pv_temp = self.compute_temp(insol_global, T_amb, wind_speed)
        pv_efficiency = self.compute_efficiency(pv_temp)
        out_power = pv_efficiency*insol_global

        if scalar_input:
            return out_power.item()
        return out_power

    def compute_temp(self, insol_global, T_amb, wind_speed):
        """

        Args:
            insol_global: global insolation in [W/m2]
            T_amb: ambient temperature in [degrees centigrade]
            wind_speed: wind speed in [m/s]
            All inputs need to be of same length.

        Returns:
            work_temp
        """
        # convert required input (scalar, list or numpy array) to numpy array
        insol_global = np.asarray(insol_global)
        T_amb = np.asarray(T_amb)
        wind_speed = np.asarray(wind_speed)

        scalar_input = False  # assume array input, unless...
        if insol_global.ndim == 0:  # first input was converted to a 0D array (= scalar)
            insol_global = insol_global[np.newaxis]  # then, convert oD array to 1D array
            scalar_input = True  # and set the scalar flag!

        if scalar_input:  # if the first argument was a scalar...
            T_amb = T_amb[np.newaxis]  # convert the other 0D arrays to 1D arrays
            wind_speed = wind_speed[np.newaxis]


        exponent = -0.61 * np.power(wind_speed,0.63)
        work_temp = T_amb + 43.3 * (np.exp(exponent) + 2.1) * (insol_global/1000)

        if scalar_input:
            # return np.asscalar(work_temp)
            return work_temp.item()
        return work_temp

    def compute_efficiency(self, work_temp):
        work_temp = np.asarray(work_temp)
        scalar_input = False
        if work_temp.ndim == 0:  # first input was converted to a 0D array (= scalar)
            work_temp = work_temp[np.newaxis]  # then, convert oD array to 1D array
            scalar_input = True

        delta_temp = work_temp - self.nom_temp
        factor = (1 + self.temp_coeff * delta_temp)
        work_efficiency = self.nom_efficiency * factor

        if scalar_input:
            # return np.asscalar(work_efficiency)
            return work_efficiency.item()
        return work_efficiency

if __name__ == "__main__":
    longitude = 0
    latitude = 0.1
    azimuth = 0.2
    inclination = 0.3
    surf_area = 4
    nominal_efficiency = 0.5
    nominal_temp = 16
    temp_coeff = -0.0041

    pv = PVPanel(longitude, latitude, azimuth, inclination, surf_area, nominal_efficiency, nominal_temp, temp_coeff)
    global_insol = 15
    amb_temp = 24
    wind_speed = 3
    work_temp = pv.compute_temp(global_insol, amb_temp, wind_speed)
    print(work_temp)
    work_efficiency = pv.compute_efficiency(work_temp)
    print(work_efficiency)
    out_power = pv.compute_output_power(global_insol, amb_temp, wind_speed)
    print(out_power)

    global_insol = [15, 20]
    amb_temp = [24 , 30]
    wind_speed = [3, 3.5]
    work_temp = pv.compute_temp(global_insol, amb_temp, wind_speed)
    print(work_temp)
    work_efficiency = pv.compute_efficiency(work_temp)
    print(work_efficiency)
    out_power = pv.compute_output_power(global_insol, amb_temp, wind_speed)
    print(out_power)
