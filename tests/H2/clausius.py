import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")


# Clausius-Clapeyron
def clausius(T, T_ref=373.15, p_ref=101.325):
    """

     Args:
          T:
          T_ref:
          p_ref:

     Returns:
     """
    DH = 40657  # latent heat in kJ/ mol
    R = 8.314472  # J/kg /mol gas constant
    exponent = -(DH / R) * (np.reciprocal(T) - np.reciprocal(T_ref))
    pcc = p_ref * np.exp(exponent)
    return pcc


def arm(Tc):
    exponent = (17.625 * Tc) / (Tc + 243.04)
    parm = 0.61094 * np.exp(exponent)
    return parm


def tetens_above(Tc):
    """Tetens equation for vapour pressure
    valid for T >= 0 C

    Args:
        Tc: temperature in C

    Returns:
       ptet (array): pressure in kPa
    """
    exponent = (17.27 * Tc) / (Tc + 237.3)
    ptet = 0.61078 * np.exp(exponent)
    return ptet


def tetens_below(Tc):
    """Tetens equation for vapour pressure
    valid for T <= 0 C (Murray 1967)

    Args:
        Tc: temperature in C

    Returns:
       ptet (array): pressure in kPa
    """
    exponent = (21.875 * Tc) / (Tc + 265.5)
    ptet = 0.61078 * np.exp(exponent)
    return ptet


def august(Tk):
    """calculate saturation pressure of water with August formula.

    Args:
        Tk: temperature in K

    Returns:
       p_sat_Pa (array): saturation pressure in Pa
    """
    exponent = 20.386 - (5132 / Tk)
    p_mmHg = np.exp(exponent)
    p_sat_Pa = 133.3224 * p_mmHg
    return p_sat_Pa


if __name__ == "__main__":
    T_C = np.arange(0.0, 101.0, 5.0)
    T_K = T_C + 273.15
    plist = [0.6113, 0.8726, 1.2281, 1.7056, 2.3388, 3.1690, 4.2455, 5.6267, 7.3814, 9.5898, 12.3440,
             15.7520, 19.9320, 25.0220, 31.1760, 38.5630, 47.3730, 57.8150, 70.1170, 84.5290, 101.3200]
    p = np.array(plist)

    p_cc = clausius(T_K, 373.15, 101.325)
    p_arm = arm(T_K - 273.15)
    p_tet = tetens_above(T_K - 273.15)
    p_aug = august(T_K)

    critical_pressure_cc = clausius(647.0)
    critical_pressure_arm = arm(647.0 - 273.15)
    critical_pressure_tetens = tetens_above(647.0 - 273.15)
    critical_pressure_aug = august(647)

    print(f"Critical pressure: {critical_pressure_cc}   {critical_pressure_arm}   "
          f"{critical_pressure_tetens}   {critical_pressure_aug} (22064) kPa")

    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax[0].plot(T_K, p, 'bo-', label="vapour pressure")
    ax[0].plot(T_K, p_cc, '-r.', label="CC")
    ax[0].plot(T_K, p_arm, '-g.', label="ARM")
    ax[0].plot(T_K, p_tet, '-k.', label="Tetens")
    ax[0].plot(T_K, p_aug, '-m.', label="August")
    ax[0].legend(loc='best')
    ax[0].set_xlabel("temperature (K)")
    ax[0].set_ylabel("vapour pressure (kPa)")

    ax[1].plot(T_K, p_cc - p, 'r.-', label="CC")
    ax[1].plot(T_K, p_arm - p, '-g.', label="ARM")
    ax[1].plot(T_K, p_tet - p, '-k.', label="Tetens")
    ax[1].plot(T_K, p_aug - p, '-m.', label="August")
    ax[1].legend(loc='best')
    ax[1].set_xlabel("temperature (K)")
    ax[1].set_ylabel("Difference in vapour pressure (kPa)")
    ax[1].set_xlim(270, 380)
    ax[1].set_ylim(-3, 3)
    plt.tight_layout()
    plt.show()
