

import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from housemodel.sourcesink.heatpumps.NTA8800_Q.defrost8800 import frost_factor_8800
from housemodel.sourcesink.heatpumps.NTA8800_Q.HPQ9 import (calc_WP_general,
                                                            plot_plane, plot_lines)

class Heatpump_NTA():
    """class for modelling PID-controlled heatpump.

    """

    def __init__(self,):
        """

        """
        self.T_evap = 20
        self.T_cond = 20
        self.cal_T_evap = np.array([7, 7, -7])
        self.cal_T_cond = np.array([35, 55, 35])
        self.cal_COP_val = np.zeros_like(self.cal_T_evap)
        self.cal_Pmax_val = np.zeros_like(self.cal_T_cond)
        self.c_coeff = np.zeros_like(self.cal_T_cond)
        self.p_coeff = np.zeros_like(self.cal_T_cond)
        self.COP_A2W35 = None
        self.Pmax_A2W35 = None
        self.Pmax = None

    def set_cal_val(self, cop_val: list, pmax_val: list):
        self.cal_COP_val = np.array(cop_val)
        self.cal_Pmax_val = np.array(pmax_val)

    def set_A2W35(self, cop_val: float, pmax_val: float):
        self.COP_A2W35 = np.array(cop_val)
        # self.Pmax_A2W35 = np.array(pmax_val)

    def update(self, Te, Tc):
        """

        Args:
            Te:
            Tc:

        Returns:

        """
        cop = self.c_coeff[0] + self.c_coeff[1] * Te + self.c_coeff[2] * Tc
        p_max = self.p_coeff[0] + self.p_coeff[1] * Te + self.p_coeff[2] * Tc

        # COP defrost correction
        frost_factor = frost_factor_8800([Te]) # input = list or 1-dim array
        cop *= frost_factor
        p_max *= frost_factor
        p_max = np.clip(p_max, 0, self.Pmax)

        return cop, p_max


if __name__ == "__main__":
    print("Heat Pump Model (L/W) according to NTA8800:2020, Appendix Q9")
    nta = Heatpump_NTA()
    nta.set_cal_val([4.0, 3.0, 2.5], [6.0, 2.0, 3.0])

    nta.c_coeff = calc_WP_general(nta.cal_T_evap, nta.cal_T_cond,
                                  nta.cal_COP_val, order=1)
    plot_plane(nta.cal_T_evap, nta.cal_T_cond,
               nta.cal_COP_val, nta.c_coeff, 'COP', 1.0, 5.0)

    nta.p_coeff = calc_WP_general(nta.cal_T_evap, nta.cal_T_cond,
                                  nta.cal_Pmax_val, order=1)
    plot_plane(nta.cal_T_evap, nta.cal_T_cond,
               nta.cal_COP_val, nta.c_coeff, 'Power', 0.0, 10.0)

    c, p = nta.update(7, 35)
    print(f"COP: {c}, Pmax: {p}")
    c, p = nta.update(7, 55)
    print(f"COP: {c}, Pmax: {p}")
    c, p = nta.update(-7, 35)
    print(f"COP: {c}, Pmax: {p}")

    Tin_space = np.linspace(-20, 20, 41, endpoint=True)
    COP_35 = [nta.c_coeff[0] + nta.c_coeff[1]*t +nta.c_coeff[2]*35.0 for t in Tin_space]
    COP_45 = [nta.c_coeff[0] + nta.c_coeff[1]*t +nta.c_coeff[2]*45.0 for t in Tin_space]
    COP_55 = [nta.c_coeff[0] + nta.c_coeff[1]*t +nta.c_coeff[2]*55.0 for t in Tin_space]

    P_35 = [nta.p_coeff[0] + nta.p_coeff[1]*t + nta.p_coeff[2]*35.0 for t in Tin_space]
    P_45 = [nta.p_coeff[0] + nta.p_coeff[1]*t+ nta.p_coeff[2]*45.0 for t in Tin_space]
    P_55 = [nta.p_coeff[0] + nta.p_coeff[1]*t + nta.p_coeff[2]*55.0 for t in Tin_space]

    # COP defrost correction
    frost_factor = frost_factor_8800(Tin_space)
    COP_35 = COP_35 * frost_factor
    COP_45 = COP_45 * frost_factor
    COP_55 = COP_55 * frost_factor

    # Pmax defrost correction
    frost_factor = frost_factor_8800(Tin_space)
    P_35 = P_35 * frost_factor
    P_45 = P_45 * frost_factor
    P_55 = P_55 * frost_factor

    plot_lines(Tin_space, COP_35, COP_45, COP_55, P_35, P_45, P_55)





