# This is described in appendix Q3.8 of the NTA 8800 on page 897
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def frost_factor_8800(T_evap, max_ff_8800=0.75):
    """calculate reduction in COP by frosting of heat pump at evaporator

    Args:
        T_evap:                 temperature at evaporator side ( outdoor temperature )
        max_frost_factor_8800:  COP reduction factor @ A2W35

    Returns: (array): frost factor for all input temperatures
                      according to NTA8800 Appendix Q

    The COP then becomes:
    COP_HP(T_evap, T_cond) = COP_HP_no_frost(T_evap, T_cond) * frost factor(T_evap)

    The Power becomes:
    Power_HP(T_evap, T_cond)  = Power_HP_no_frost(T_evap, T_cond)  * frost factor(T_evap)

    Notes: - no separate frost factor for the power is specified.
           - this is an approximation!
           - these measurements are valid for full power
    """
    # The correction for frosting Frost_factor( T_air ) now becomes:
    T = np.array(T_evap)   # input type can be list or numpy array
    ff = np.ones_like(T)
    index1 = np.where((T > -7.0) & (T <= 2.0))
    ff[index1] = frost_factor = ((max_ff_8800 - 1.0) / 9.0) *T[index1] + (( 7 * max_ff_8800 + 2.0) / 9.0)
    index2 = np.where((T > 2.0) & (T < 7.0))
    ff[index2] = ((1.0 - max_ff_8800) / 5.0) *T[index2] + (( 7 * max_ff_8800 - 2.0) / 5.0)
    return ff


def maximum_frost_factor_8800(COP_par, COP_A2W35=None ):
    """    calculate reduction in COP by frosting of heat pump at evaporator

    Args:
        COP_par:    tuple of 3 coefficients for linear fit
                    of COP in heat pump model NTA8800:2020 Appendix Q.9
        COP_A2W35:  COP measured at Tevap = 2 degrees and Tcond = 35 degrees,
                    if available

    Returns:        maximum frost_factor @A2W35 ( default =0.75 )

    """
    if COP_A2W35 is not None:
        # calculate COP @ A2W35 according to NTA880 model coefficients
        Tevap_ref = 2 # degrees Celsius
        Tcond_ref = 35 # degrees Celsius
        COP_nofrost = COP_par[0] + COP_par[1] * Tevap_ref + COP_par[2] * Tcond_ref

        # calculate maximum frost factor, occurring for @ A2W35
        return COP_A2W35 /  COP_nofrost
    else:
        return 0.75 # default value from NTA8800


if __name__ == "__main__":
    par = np.c_[[5.0, 0.10, -0.05]] # make column vector of
    ff_max = maximum_frost_factor_8800(par)
    print(f"maximum frost factor @ A2W35 {ff_max}")

    T_air = list(np.arange(-10, 11, 1.0))
    COP_factor = frost_factor_8800(T_air)

    fig = plt.figure()
    plt.suptitle("COP reduction factor due to defrosting of heat pump evaporator")
    plt.title("according to NTA8800:2020, Appendix Q (Q.18-Q.19)")
    plt.plot(T_air, COP_factor, marker='o')
    plt.xlim(-10, 10)
    plt.xticks(np.arange(-10, 12, step=2.0))
    plt.grid()
    plt.show()