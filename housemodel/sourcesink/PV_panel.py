
from math import exp

class PVPanel:
    """"
        class for modeling a PV-panel
    """

    def __init__(self, loc_long, loc_lat, orient_azimuth,orient_incl, area,
                 nom_efficiency, nom_temp, temp_coeff, temp, work_efficiency):
        """

        Args:
            loc_long: (float) longitude of the panel location [rad between 0 and pi/2]
            loc_lat: (float) latitude of the panel location [rad between -pi and pi]
            orient_azimuth: (float) azimuth angle of the panel normal vector [rad between -pi and pi, 0=S pi/2 = W]
            orient_incl: (float) inclination angle of the panel [rad, between 0 and pi/2]
            area: (float) surface area in [m2]
            nom_efficiency: (float) nominal efficiency value [# between 0 and 1]
            nom_temp
            T_coeff
            temp
        """
        self.loc_long = loc_long
        self.loc_lat = loc_lat
        self.orient_azimuth = orient_azimuth
        self.orient_incl = orient_incl
        self.area = area
        self.nom_efficiency = nom_efficiency
        self.nom_temp = nom_temp
        self.temp_coeff = temp_coeff
        self.work_temp = temp
        self.work_efficiency = work_efficiency

    def compute_output_power(self, insol_global, T_amb, wind_speed):
        self.update_temp(insol_global,T_amb,wind_speed)
        self.update_efficiency()

        return out_power

    def update_temp(self, insol_global, T_amb, wind_speed):
        """

        Args:
            insol_global: global insolation in [W/m2]
            T_amb: ambient temperature in [degrees centigrade]
            wind_speed: wind speed in [m/s]

        Returns:

        """

        exponent = -0.61 * pow(wind_speed,0.63)
        self.work_temp = T_amb + 43.3 * (exp(exponent) + 2.1) * (insol_global/1000)

    def update_efficiency(self):
        delta_temp = self.work_temp - self.nom_temp
        factor = (1 + self.temp_coeff * delta_temp)
        self.work_efficiency = self.nom_efficiency * factor


if __name__ == "__main__":
    longitude = 0
    latitude = 0.1
    azimuth = 0.2
    inclination = 0.3
    area = 4
    nominal_efficiency = 0.5
    nominal_temp = 16
    temp_coeff = -0.0041
    work_temp = 16
    work_eff = 0.5


    pv = PVPanel(longitude, latitude, azimuth, inclination, area, nominal_efficiency, nominal_temp, temp_coeff, work_temp, work_eff)

    print(pv.work_temp)
    print(pv.work_efficiency)
    pv.update_temp(15,24,3)
    print(pv.work_temp)
    print(pv.work_efficiency)
    pv.update_efficiency()
    print(pv.work_efficiency)