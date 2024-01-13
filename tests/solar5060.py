import numpy as np

sol_const_5060 = 1370


def extra_terrestrial_5060(nday):
    """calculates extraterrestrial irradiance G sol,ext (D.18)

    Args:
        Nday:	is	het	volgnummer	van	de	dag	in	het	jaar.

    Returns:
        extraterrestrial radiation

    See: https://stackoverflow.com/questions/54148787/minor-mistake-in-calculation-of-extraterrestrial-radiation-method-asce
    """
    B = (360/365) * nday # not (nday-1)!
    return sol_const_5060* (1.0 + 0.033 * np.cos(B))  # in [W/m^2]


def H_sol_5060(decl, lat, ha):
    """

    Args:
        decl:
        lat:
        ha:

    Returns:
    """
    s = np.sin(decl)
    return