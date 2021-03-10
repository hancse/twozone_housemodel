# This is described in appendix Q3.8 of the NTA 8800 on page 897
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def defrost8800(T_evap, COP_par, COP_A2W35=None ):
    """
    calculate reduction in COP by frosting of heat pump at evaporator

    :param T_evap:     temperature at evaporator side ( outdoor temperature
    :param COP_par:    tuple of 3 coefficients for COP of heat pump model
    :param COP_A2W35:  COP measured at Tevap = 2 degrees and Tcond = 35 degrees, if available
    :return: frost_factor:

    The  COP then becomes:
    COP_HP(T_evap, T_cond) = COP_HP_no_frost(T_evap, T_cond) * frost factor(T_evap)
    The Power becomes:
    Power_HP(T_evap, T_cond)  = Power_HP_no_frost(T_evap, T_cond)  * frost factor(T_evap)
    Remark: no separate frost factor for the power is specified. This is an approximation!
    These measurements are valid for full power
    """
    Tevap_ref = 2 # degrees Celsius
    Tcond_ref = 35 # degrees Celsius
    COP_nofrost = COP_par[0] + COP_par[1] * Tevap_ref + COP_par[2] * Tcond_ref
    if COP_A2W35:
        frost_factor_0 = COP_A2W35 /  COP_nofrost
    else:
        frost_factor_0 = 0.75

    # The correction for frosting Frost_factor( T_air )  now becomes:
    if (T_evap <= -7) or (T_evap >= 7):
        frost_factor = 1.0
    elif T_evap <= 2:
        frost_factor = ((frost_factor_0 - 1.0) / 9.0) *T_evap + (( 7 * frost_factor_0 + 2.0) / 9.0)
    else:
        frost_factor = ((1.0 - frost_factor_0) / 5.0) *T_evap + (( 7 * frost_factor_0 - 2.0) / 5.0)

    return frost_factor


if __name__ == "__main__":
    par = np.c_[[5.0, 0.10, -0.05]]
    Tair = np.arange(-10, 11, 1.0)
    # ff = np.zeros(np.shape(Tair))
    ff = np.empty(np.shape(Tair))
    list = []
    for T in Tair:
        list.append(defrost8800(T, par))

    fig = plt.figure()
    plt.suptitle("COP reduction factor due to defrosting of heat pump evaporator")
    plt.title("according to NTA8800:2020, Appendix Q (Q.18-Q.19)")
    plt.plot(Tair, list)
    plt.xlim(-10, 10)
    plt.xticks(np.arange(-10, 12, step=2.0))
    plt.grid()
    plt.show()