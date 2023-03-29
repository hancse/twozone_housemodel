# https://learn.openenergymonitor.org/sustainable-energy/building-energy-model/radiatormodel
# https://www.buildingservicesindex.co.uk/entry/136540/AEL-Heating-Solutions-Ltd/How-to-calculate-the-delta-T-for-a-radiator/

import numpy as np

from scipy import interpolate
from scipy.optimize import root
from housemodel.sourcesink.heatexchangers import LMTD

from housemodel.basics.components import (FixedNode)

import matplotlib
import matplotlib.pyplot as plt
import logging

matplotlib.use("Qt5Agg")

# logging.basicConfig(level="DEBUG")
logging.basicConfig(level="INFO")


def LMTD_radiator(T_feed, T_return, T_amb, corrfact=1.0):
    """calculates log mean temperature difference

    representative value in case of varying temperature difference along heat exchanger
    https://checalc.com/solved/LMTD_Chart.html
    Args:
        T_feed:     entry temperature hot fluid or gas
        T_return:    exit temperature hot fluid or gas
        T_amb:    entry temperature cold fluid or gas
        corrfact:    see:     https://checalc.com/solved/LMTD_Chart.html
                              https://cheguide.com/lmtd_charts.html
                              https://excelcalculations.blogspot.com/2011/06/lmtd-correction-factor.html
                              http://fchart.com/ees/heat_transfer_library/heat_exchangers/hs2000.htm
                              https://yjresources.files.wordpress.com/2009/05/4-3-lmtd-with-tutorial.pdf
                              https://www.engineeringtoolbox.com/arithmetic-logarithmic-mean-temperature-d_436.html
    Returns:
        LMTD temperature
        corr_fact * ( Delta T 1 - Delta T 2 ) / ln (Delta t 1 / Delta T 2)
    """
    eps = 1e-9
    DeltaT_fr = T_feed - T_return
    DeltaT_feed = T_feed - T_amb
    DeltaT_ret = T_return - T_amb
    # assert (DeltaT_fr > 0), "Output temperature difference $\Delta T_1$ is negative"
    # assert DeltaT_in > DeltaT_out, "Input temperature difference $\Delta T_1$ is smaller than output "

    denominator = np.log(DeltaT_feed) - np.log(DeltaT_ret)
    nominator = DeltaT_fr
    # assert denominator > eps, "Ratio of input/output temperature difference too large"
    log_mean_diff_temp = corrfact * nominator / denominator
    return log_mean_diff_temp


def GMTD_radiator(T_feed, T_return, T_amb, corrfact=1.0):
    """calculates geometric mean temperature difference

    Args:
        T_feed:     entry temperature hot fluid or gas
        T_return:   exit temperature hot fluid or gas
        T_amb:      entry temperature cold fluid or gas
        corrfact:   correction factor. See: LMTD_radiator
    Returns:
        GMTD temperature      corr_fact * sqrt(T_feed - T_amb) * sqrt(T_ret - T_amb)
    """
    DeltaT_feed = T_feed - T_amb
    DeltaT_ret = T_return - T_amb
    nominator = np.sqrt(DeltaT_feed) * np.sqrt(DeltaT_ret)
    geo_mean_diff_temp = corrfact * nominator
    return geo_mean_diff_temp


def calc_corr_fact(delta_t):
    dt_model = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    cf_model = [0.05, 0.123, 0.209, 0.304, 0.406, 0.515, 0.629, 0.748, 0.872, 1.0, 1.132, 1.267, 1.406, 1.549, 1.694]
    f = interpolate.interp1d(dt_model, cf_model)
    corr_fact = f(delta_t)
    return corr_fact


def calc_log_mean_diff_rad(Tinlet, Treturn, Tamb):
    """

    Args:
        Tinlet:  inlet temperature of radiator
        Treturn:  return temperature of radiator
        Tamb:     ambient temperature

    Returns:
        (float): log mean temp difference of radiator
    """
    lm = LMTD(Tinlet, Treturn, Tamb, Tamb,
              flowpattern='cross')
    return lm


def calc_mean_diff_rad(Tinlet, Treturn, Tamb):
    lm = np.mean([Tinlet, Treturn]) - Tamb
    return lm


class Radiator:
    """ class for general Radiator object."""
    def __init__(self, name="DefaultRadiator", begin_tag=0, exp_rad=1.3):
        self.name = name
        self.begin_tag = begin_tag
        self.num_nodes = 1
        self.exp_rad = exp_rad

        self.end_node = self.begin_tag  # anchor point of cold water return from house model
        self.nodes = []
        # self.edges = []
        # self.num_edges = 0
        self.boundaries = []
        self.return_node = FixedNode
        self.q_dot = 0.0
        # self.ambient = None

        self.tag_list = []
        self.cap_list = []
        self.edge_list = []
        self.cond_list = []

        self.c_inv_mat = None
        self.k_int_mat = None
        self.k_ext_mat = None

        self.q_vec = None  # np.zeros(self.num_nodes, 1)
        self.f_mat = None  # np.zeros(self.num_nodes, self.num_nodes)

        self.rho = 1000      # [kg/m^3]
        self.cp = 4190  # [J/(K kg)]
        self.flow = None     # [m^3/s]
        self.F_rad = None     # heat flow in [W/K] = flow * rho * c_w
        self.T_feed = None
        self.T_amb = None

        # model radiator as in EN442: Q = Km * LMTD ** n
        self.Km = None
        # model radiator as in DTU lit: q/q_0 = (LMTD/LMTD_0) ** n
        self. LMTD_0 = 50
        self. T_feed_zero = 75     # [C]
        self. T_ret_zero = 65      # [C]
        self. T_amb = 20
        self. q_zero = 1000        # [W]

        self.__denominator = None
        self.__gmtd = None
        self.__lmtd = None
        self.calculate_radiator_properties()
        logging.info(f" Radiator object {self.name} created")

    @classmethod
    def from_dict(cls, d):
        """ classmethod to enable constructing an instance from configuration file.
        """
        return cls(name=d["name"], begin_tag=d["begin_tag"], exp_rad=d["exp_rad"])

    def calculate_radiator_properties(self):
        pass

    def set_exponent(self, e):
        self.exp_rad = e
        self.calculate_radiator_properties()

    def boundaries_from_dict(self, lod):
        for n in range(len(lod)):
            node = FixedNode(label=lod[n]["label"],
                             temp=lod[n]["T_ini"],
                             connected_to=lod[n]["connected_to"])
            # append by reference, therefore new node object in each iteration
            self.boundaries.append(node)
        self.return_node = [fn for fn in self.boundaries if fn.label == "return"][0]

    def get_gmtd(self):
        return self.__gmtd

    def get_lmtd(self):
        return self.__lmtd

    def func_rad_gmtd(self, x):
        """model function for scipy.optimize.root().

        Args:
            x: vector with unknowns [self.q_dot, self.T_ret]

        Returns:
            f : vector with model functions evaluated at x
            df : Jacobian (partial derivatives of model functions wrt x
        """
        self.__gmtd = GMTD_radiator(T_feed=self.T_feed, T_return=x[1], T_amb=self.T_amb, corrfact=1.0)
        # set of nonlinear functions for root finding
        f = [x[0] - (self.Km * self.__gmtd ** self.exp_rad),
             x[0] - self.F_rad * (self.T_feed - x[1])]

        DeltaT_feed = self.T_feed - self.T_amb
        dTdt = -1.0 * self.Km * 0.5*self.exp_rad * np.float_power(DeltaT_feed, (0.5*self.exp_rad))
        dTdt *= np.float_power(x[1] - self.T_amb, (0.5*self.exp_rad) - 1)

        df = np.array([[1.0, dTdt],
                       [1.0, self.F_rad]])
        return f, df

    def func_rad_lmtd(self, x):
        """model function for scipy.optimize.root().

        Args:
            x: vector with unknowns [self.q_dot, self.T_ret]

        Returns:
            f : vector with model functions evaluated at x
            df : Jacobian (partial derivatives of model functions wrt x
        """
        self.__lmtd = LMTD_radiator(T_feed=self.T_feed, T_return=x[1], T_amb=20.0, corrfact=1.0)
        # set of nonlinear functions for root finding
        f = [x[0] - (self.Km * self.__lmtd ** self.exp_rad),
             x[0] - self.F_rad * (self.T_feed - x[1])]

        h1 = self.Km * self.exp_rad
        h1 *= self.__lmtd ** (self.exp_rad - 1.0)

        h2 = (self.T_feed - x[1]) / (x[1] - self.T_amb)
        denominator = np.log(self.T_feed - self.T_amb) - np.log(x[1] - self.T_amb)
        h2 -= denominator

        dTdt = -(h1 * h2) / (denominator * denominator)

        df = np.array([[1.0, dTdt],
                       [1.0, self.F_rad]])
        return f, df

    def update(self, func):
        """update roots [self.q_dot, self.T_ret] with model function
           using scipy.optimize.root().

        Returns:
            None
        """
        opt_res = root(func, [self.q_dot, self.T_ret], jac=True, method='hybr')
        self.q_dot = opt_res.x[0]
        self.T_ret = opt_res.x[1]


if __name__ == "__main__":
    deg = u"\u00b0"
    log_mean_rad = LMTD_radiator(80, 50, 20)
    print(f"LMTD_radiator : {log_mean_rad} {deg}C")
    log_mean_rad_zero = LMTD_radiator(75, 65, 20)
    print(f"LMTD_radiator_design : {log_mean_rad_zero} {deg}C")

    lm_ref = calc_log_mean_diff_rad(75, 65, 20)
    print(f"Reference LMTD: {lm_ref}")

    dt_ref = calc_mean_diff_rad(75, 65, 20)
    print(f"Reference Delta_T: {dt_ref}")

    radiator = Radiator(name="Test", exp_rad=1.3)
    radiator.T_feed = 80.0
    T_amb = 20.0
    radiator.T_ret = (radiator.T_feed + T_amb) / 2.0
    radiator.T_amb = 20.0
    radiator.Km = 50
    radiator.flow = 0.05e-3     # 0.05 l/s
    radiator.F_rad = radiator.flow * radiator.rho * radiator.cp
    print(f"Frad: {radiator.F_rad} [W/K]")

    radiator.update(radiator.func_rad_lmtd)
    print(f"Q_dot: {radiator.q_dot}, T_return: {radiator.T_ret}")
    print(f"LMTD {radiator.get_lmtd()}")
    print(f"GMTD {radiator.get_gmtd()}")

    Delta_T = [20, 25, 30, 35, 40, 45, 50]
    cf = [0.3, 0.41, 0.52, 0.63, 0.75, 0.87, 1.0]

    Delta_T_2 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    cf_2 = [0.05, 0.123, 0.209, 0.304, 0.406, 0.515, 0.629, 0.748, 0.872, 1.0, 1.132, 1.267, 1.406, 1.549, 1.694]

    cf_test = calc_corr_fact(dt_ref)
    print(f"correction factor: {cf_test}")
    fig, ax = plt.subplots()
    ax.plot(Delta_T, cf, 'b-')
    ax.plot(Delta_T_2, cf_2, 'r--')
    ax.plot(dt_ref, cf_test, 'og')
    ax.grid(True)
    ax.set_ylim(0, 1.75)
    plt.show()
    
