
"""
SYNTAX: E=irrad(Dh,En,iday,LST,gamma,beta,rground)
OUTPUT: global irradiation on a surface
         
Splitted irradiation

E(1)= diffuse solar irradiation on an inclined surface
E(2)= direct solar irradiation on an inclined surface
E(3)= total solar irradiation on an inclined surface
E(4)= total solar irradiation on a horizontal surface

INPUT:


east:gamma = -90, west:gamma = 90
south:gamma = 0, north:gamma = 180

horizontal: beta=0, vertical: beta=90
default geographical position: De Bilt

 
EXAMPLE: E=irrad(800,200,201,12,0,45)
ANSWER: E=1.0e+003 *
0.8569 0.1907 1.0759 0.9684

REF: Perez (zie Solar Energy volume 39 no. 3)

Adapted version from Eindhoven University of Technology: JvS feb 2002
"""
import numpy as np


def solar_check(tclim):
    """solar_check Calculates the irradiation on an outer surface
       (= adaptation of qsun routine)

    Args:
        tclim:   seconds since 1st of January 00:00:00 (naive time)
        Dh:      (scalar) Dh = diffuse horizontal irradiation[W/m2]
        En:      (scalar) En = direct normal irradiation [W/m2]
        gamma:   (scalar) gamma = azimuth angle of the surface
        beta:    (scalar) beta = inclination angle of the surface
        rground: default ground reflectivity (albedo): 0.2

    Returns:
        ET:  equation of time
        delta: declination angle
        h:     hour angle
        zet:   zenith angle
        teta:  incident angle of the tilt surface
        Eon:   extraterrestrial radiation
    """
    iday = 1 + np.floor(tclim / (24 * 3600))  # day of the year (1-365)
    LST = np.floor((tclim / 3600) % 24)       # Local Standard time (0 - 23) [hour]

    # L = Latitude [graden]
    L = 52.1
    # LON = Local Longitude [graden] oost is positief
    LON = 5.1
    # LSM = Local Standard time Meridian [graden] oost is positief
    LSM = 15 * 2
    r = np.pi / 180
    L = L * r  # conversion degrees-> radians
    theta = 2 * np.pi * (iday - 1) / 365.25
    el = 4.901 + 0.033 * np.sin(-0.031 + theta) + theta  # elevation

    # declination
    delta = np.arcsin(np.sin(23.442 * r) * np.sin(el))
    q1 = np.tan(4.901 + theta)
    q2 = np.cos(23.442 * r) * np.tan(el)

    # equation of time
    ET = (np.arctan((q1 - q2) / (q1 * q2 + 1))) * 4 / r
    AST = LST + ET / 60 + (4 / 60) * (LSM - LON)  # change from minus to plus
    h = (AST - 12) * 15 * r

    # hai=sin(solar altitude) # length of incline surface??? check sun position with the surface
    # hai = np.cos(L) * np.cos(delta) * np.cos(h) + np.sin(L) * np.sin(delta)

    # E = np.zeros((1, 4))
    # teta = incident angle on the tilted surface
    # teta = 0
    # determination of zet = solar zenith angle (pi/2 - solar altitude).
    # zet = 0
    # calculation of extraterrestrial radiation
    Eon = 0
    # if hai > 0:
    # salt=solar altitude
    #   salt = np.asin(hai)
    #   phi = np.acos((hai * np.sin(L) - np.sin(delta)) / (np.cos(salt) * np.cos(L))) * np.sign(h)

    # calculation of extraterrestrial radiation
    # Eon = 1370 * (1 + 0.033 * np.cos(2 * np.pi * (iday - 3) / 365))
    # Eon = 1366.1 * (1 + 0.033 * np.cos(2 * np.pi * (iday - 3) / 365))
    E_naught = 1366.1  # [W/m^2]
    b = 2.0 * np.pi * iday / 365.0  # radians
    b = 2.0 * np.pi * (iday - 1) / 365.0  # radians
    Eon = E_naught * (1.00011 + 0.034221 * np.cos(b) + 0.001280 * np.sin(b)
                      + 0.000719 * np.cos(2 * b) + 0.000077 * np.sin(2 * b))

    dummy1 = 0
    dummy2 = 0
    return ET, delta, h, dummy1, dummy2, Eon

    # ET:  equation of time
    # delta: declination angle
    # h:     hour angle
    # zet:   zenith angle
    # teta:  incident angle of the tilt surface
    # Eon:   extraterrestrial radiation
