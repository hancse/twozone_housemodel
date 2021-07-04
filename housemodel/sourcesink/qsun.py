"""
qsun: Calculates the irradiation on an inclined (outer) surface

    OLD SYNTAX: E = irrad(Dh, En, iday, LST, gamma, beta, rground)
    OUTPUT: irradiation on a surface

Irradiation output is split into the following components

    E(0) = diffuse solar irradiation on an inclined surface
    E(1) = direct solar irradiation on an inclined surface
    E(2) = total solar irradiation on an inclined surface
    E(2) = total solar irradiation on a horizontal surface

INPUT:
    (scalar) Dh = diffuse horizontal irradiation[W/m2]
    (scalar) En = direct normal irradiation [W/m2]
    (scalar) t = time in seconds after 1 January 00:00
    (scalar) gamma = azimuth angle of the surface
    east: gamma = -90, west:gamma = 90
    south: gamma = 0, north:gamma = 180
    (scalar) beta = inclination angle of the surface,
    horizontal: beta=0, vertical: beta=90
    default geographical position: De Bilt
    default ground reflectivity (albedo): 0.2
    WA: Window area
    WZTA: Windor solar factor (zontoetredingsfactor)

EXAMPLE:
    E=irrad(800,200,201,12,0,45)

ANSWER:
    E=1.0e+003 * [0.8569 0.1907 1.0759 0.9684]

REFERENCES:
    Perez et al., Solar Energy volume 39 no. 3

    Adapted version from Eindhoven University of Technology: JvS feb 2002
"""
# import math        all math functions replaced by numpy
import numpy as np

"""
A problem with the current qsun routine is that the calling sequence has become a bit awkward.
In Python function calls, the required input parameters must come before the optional parameters.
The only optional parameter is rground, which is now in the middle.
A new and better calling convention would be:
def qsun(iday, LST, Dh, En, gamma, beta, rground=0.0):
We should determine a good moment to carry out this change, which will require adaptation of all usage of qsun
The change is therefore postponed to that suitable moment.
"""

def qsun(Dh, En, gamma, beta, rground, iday, LST):
    """ calculate direct, diffuse and reflected sunlight on a surface

    Args:
        (scalar) Dh     : diffuse horizontal irradiation [W/m2]
        (scalar) En     : direct normal irradiation [W/m2]
        (scalar) t      : time in seconds after 1 January 00:00
        (scalar) gamma  : azimuth angle of the surface,
            - east			: gamma = -90, west:gamma = 90
            - south			: gamma = 0, north:gamma = 180
        (scalar) beta 	 : inclination angle of the surface
            - horizontal	: beta=0
            - vertical      : beta=90
        (scalar) rground : default ground reflectivity (albedo): 0.2
        (scalar) iday    : day of the year (1-365)
        (scalar) LST     : Local Standard time (0 - 23) [hour]

        default geographical position: De Bilt

    Returns:

        E[0,0]	: diffuse radiation
        E[0,1]	: direct radiation
        E[0,2]	: Total radiation
        E[0,3]	: global radiation
    """

    # (scalar) iday = day of the year (1-365)
    # (scalar) LST = Local Standard time (0 - 23) [hour]

    # iday = 1 + np.floor(tclim / (24 * 3600))
    # LST  = np.floor((tclim / 3600) % 24)

    # L = Latitude [graden]
    L = 52.1
    # LON = Local Longitude [graden] oost is positief
    LON = 5.1
    # LSM = Local Standard time Meridian [graden] oost is positief
    LSM = 15 * 1

    # rground = albedo
    # rground = 0.2
    r = np.pi / 180
    L = L * r              # conversion degrees-> radians
    beta = beta * r
    theta = 2 * np.pi * (iday - 1) / 365.25  # angle in circular earth orbit around sun

    el = 4.901 + 0.033 * np.sin(-0.031 + theta) + theta  # elevation

    # declination
    delta = np.arcsin(np.sin(23.442 * r) * np.sin(el))
    q1 = np.tan(4.901 + theta)
    q2 = np.cos(23.442 * r) * np.tan(el)

    # equation of time
    ET = (np.arctan((q1 - q2) / (q1 * q2 + 1))) * 4 / r
    AST = LST + ET / 60 + (4 / 60) * (LSM - LON)  # change from minus to plus
    h = (AST - 12) * 15 * r

    # hai=sin(solar altitude) # length of inclined surface??? check sun position with the surface
    hai = np.cos(L) * np.cos(delta) * np.cos(h) + np.sin(L) * np.sin(delta)

    E = np.zeros((1, 4))
    # teta = incident angle on the tilted surface
    teta = 0
    # determination of zet = solar zenith angle (pi/2 - solar altitude).
    zet = 0
    # calculation of extraterrestrial radiation
    Eon = 0
    # print(hai)

    if hai > 0:
        # salt=solar altitude
        salt = np.arcsin(hai)
        phi = np.arccos((hai * np.sin(L) - np.sin(delta)) / (np.cos(salt) * np.cos(L))) * np.sign(h)
        gam = phi - gamma * r
        # cai=cos(teta)
        cai = np.cos(salt) * np.cos(abs(gam)) * np.sin(beta) + hai * np.cos(beta)

        # teta = incident angle on the tilted surface
        teta = np.arccos(cai)

        # salts=solar altitude for an inclined surface
        salts = np.pi / 2 - teta

        # Perez (zie Solar Energy volume 39 no. 3)
        # berekening van de diffuse straling op een schuin vlak
        # Approximation of A and C, the solid angles occupied by the circumsolar region,
        # weighed by its average incidence on the slope and horizontal respectively.
        # In the expression of diffuse on inclined surface the quotient of A/C is
        # reduced to XIC/XIH. A=2*(1-cos(beta))*xic, C=2*(1-cos(beta))*xih
        # gecontroleerd okt 1996 martin de wit

        # alpha= the half-angle circumsolar region
        alpha = 25 * r
        if salts < -alpha:
            xic = 0
        elif salts > alpha:
            xic = cai
        else:
            xic = 0.5 * (1 + salts / alpha) * np.sin((salts + alpha) / 2)

        if salt > alpha:
            xih = hai
        else:
            xih = np.sin((alpha + salt) / 2)

        epsint = [1.056, 1.253, 1.586, 2.134, 3.23, 5.98, 10.08, 999999]
        f11acc = [-0.011, -0.038, 0.166, 0.419, 0.710, 0.857, 0.734, 0.421]
        f12acc = [0.748, 1.115, 0.909, 0.646, 0.025, -0.370, -0.073, -0.661]
        f13acc = [-0.080, -0.109, -0.179, -0.262, -0.290, -0.279, -0.228, 0.097]
        f21acc = [-0.048, -0.023, 0.062, 0.140, 0.243, 0.267, 0.231, 0.119]
        f22acc = [0.073, 0.106, -0.021, -0.167, -0.511, -0.792, -1.180, -2.125]
        f23acc = [-0.024, -0.037, -0.050, -0.042, -0.004, 0.076, 0.199, 0.446]

        # determination of zet = solar zenith angle (pi/2 - solar altitude).
        zet = np.pi / 2 - salt

        # determination of inteps with eps
        inteps = 0
        if Dh > 0:
            eps = 1 + En / Dh
            inteps = 7  # give big random number for starting point
            for i in range(len(epsint)):
                if epsint[i] >= eps:
                    temp_i = i
                    inteps = min(temp_i, inteps)
            # print(inteps)
            # inteps=min(i)

        # calculation of inverse relative air mass
        airmiv = hai
        if salt < 10 * r:
            airmiv = hai + 0.15 * (salt / r + 3.885) ** (-1.253)  # change ^ to **

        # calculation of extraterrestrial radiation, varying due to (Rav/R)^2
        # the square of the sun-earth distance
        Eon = 1370 * (1 + 0.033 * np.cos(2 * np.pi * (iday - 3) / 365))
        # Eon = 1366.1 * (1 + 0.033 * np.cos(2 * np.pi * (iday - 3) / 365))
        # E_naught = 1366.1 # [W/m^2]
        #b = 2.0 * np.pi * iday / 365.0 # radians
        # b = 2.0 * np.pi * (iday-1) / 365.0 # radians
        # Eon = E_naught * ( 1.00011 + 0.034221 * np.cos(b) + 0.001280 * np.sin(b) \
                           # + 0.000719 * np.cos(2*b) + 0.000077 * np.sin(2*b))

        # delta is "the new sky brightness parameter"
        delta = Dh / (airmiv * Eon)

        # determination of the "new circumsolar brightness coefficient
        # (f1acc) and horizon brightness coefficient (f2acc)"// Why use circumsolar

        f1acc = f11acc[inteps] + f12acc[inteps] * delta + f13acc[inteps] * zet
        f2acc = f21acc[inteps] + f22acc[inteps] * delta + f23acc[inteps] * zet

        # determination of the diffuse radiation on an inclined surface
        E[0, 0] = Dh * (0.5 * (1 + np.cos(beta)) * (1 - f1acc) + f1acc * xic / xih + f2acc * np.sin(beta))

        if E[0, 0] < 0:
            E[0, 0] = 0

        # horizontal surfaces treated separately
        # beta=0 : surface facing up, beta=180(pi) : surface facing down
        # 3/22/19 10:05 AM E:\NEN5060\qsun.m 4 of 4

        if -0.0001 < beta < 0.0001:
            E[0, 0] = Dh

        if (np.pi - 0.0001) < beta < (np.pi + 0.0001):
            E[0, 0] = 0

        # Isotropic sky
        # E(1)=0.5*(1+cos(beta))*Dh;

        # direct solar radiation on a surface

        E[0, 1] = En * cai  # beam  aoi projection, beam_component

        if E[0, 1] < 0.0:
            E[0, 1] = 0

        # the ground reflected component: assume isotropic
        # ground conditions.

        Eg = 0.5 * rground * (1 - np.cos(beta)) * (Dh + En * hai)

        # global irradiation
        E[0, 3] = Dh + En * hai

        # total irradiation on an inclined surface
        E[0, 2] = E[0, 0] + E[0, 1] + Eg

    return E[0, 0], E[0, 1], E[0, 2], E[0, 3]

# E[0,0] diffuse radiation on an inclined surface
# E[0,1] direct solar radiation on a surface
# E[0,2] total irradiation
# E[0,3]  global irradiation
