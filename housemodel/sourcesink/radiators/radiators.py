# https://learn.openenergymonitor.org/sustainable-energy/building-energy-model/radiatormodel
# https://www.buildingservicesindex.co.uk/entry/136540/AEL-Heating-Solutions-Ltd/How-to-calculate-the-delta-T-for-a-radiator/

import numpy as np

from scipy import interpolate
from scipy.optimize import root

from housemodel.tools.new_configurator import load_config
from housemodel.basics.flows import Flow

import matplotlib
import matplotlib.pyplot as plt
import logging

matplotlib.use("Qt5Agg")

# logging.basicConfig(level="DEBUG")
logging.basicConfig(level="INFO")


def LMTD_radiator(T_feed, T_return, T_amb):
    """calculates log mean temperature difference
    representative value in case of varying temperature difference along heat exchanger
    https://checalc.com/solved/LMTD_Chart.html.

    Args:
        T_feed:     entry temperature hot fluid or gas
        T_return:   exit temperature hot fluid or gas
        T_amb:      temperature of surroundings

    Returns:
        LMTD temperature: ( Delta T 1 - Delta T 2 ) / ln (Delta t 1 / Delta T 2)
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
    log_mean_diff_temp = nominator / denominator
    return log_mean_diff_temp


def GMTD_radiator(T_feed, T_return, T_amb):
    """calculates geometric mean temperature difference.

    Args:
        T_feed:     entry temperature hot fluid or gas
        T_return:   exit temperature hot fluid or gas
        T_amb:      temperature of surroundings

    Returns:
        GMTD temperature: sqrt(T_feed - T_amb) * sqrt(T_ret - T_amb)
    """
    DeltaT_feed = T_feed - T_amb
    DeltaT_ret = T_return - T_amb
    nominator = np.sqrt(DeltaT_feed) * np.sqrt(DeltaT_ret)
    geo_mean_diff_temp = nominator
    return geo_mean_diff_temp


def calc_corr_fact(delta_t):
    """corrfact: see:     https://checalc.com/solved/LMTD_Chart.html
                          https://cheguide.com/lmtd_charts.html
                          https://excelcalculations.blogspot.com/2011/06/lmtd-correction-factor.html
                          http://fchart.com/ees/heat_transfer_library/heat_exchangers/hs2000.htm
                          https://yjresources.files.wordpress.com/2009/05/4-3-lmtd-with-tutorial.pdf
                          https://www.engineeringtoolbox.com/arithmetic-logarithmic-mean-temperature-d_436.html
    """
    dt_model = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    cf_model = [0.05, 0.123, 0.209, 0.304, 0.406, 0.515, 0.629, 0.748, 0.872, 1.0, 1.132, 1.267, 1.406, 1.549, 1.694]
    f = interpolate.interp1d(dt_model, cf_model)
    corr_fact = f(delta_t)
    return corr_fact


def plot_corr_fact():
    Delta_T = [20, 25, 30, 35, 40, 45, 50]
    cf = [0.3, 0.41, 0.52, 0.63, 0.75, 0.87, 1.0]

    Delta_T_2 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    cf_2 = [0.05, 0.123, 0.209, 0.304, 0.406, 0.515, 0.629, 0.748, 0.872, 1.0, 1.132, 1.267, 1.406, 1.549, 1.694]

    # dt_ref = calc_mean_diff_rad(75, 65, 20) # arithmetic mean, use Phetteplace
    # print(f"Reference Delta_T: {dt_ref}")
    # cf_test = calc_corr_fact(dt_ref)
    # print(f"correction factor: {cf_test}")

    fig, ax = plt.subplots()
    ax.plot(Delta_T, cf, 'b-')
    ax.plot(Delta_T_2, cf_2, 'r--')
    # ax.plot(dt_ref, cf_test, 'og')
    ax.grid(True)
    ax.set_ylim(0, 1.75)
    plt.show()


class Radiator:
    """ class for general Radiator object."""
    def __init__(self, name="DefaultRadiator", exp_rad=1.3):
        self.name = name
        self.exp_rad = exp_rad
        # self.return_node = FixedNode
        self.q_dot = 0.0

        self.T_supply = None
        self.T_return = None
        self.T_amb = 20
        self.flow = None  # Flow()  # default Flow object
        # self.flow_rate = None
        # self.F_rad = None    # heat flow in [W/K] = flow * rho * c_w  replaced by self.flow.heat_rate

        # model radiator as in EN442: Q = Km * LMTD ** n
        self.Km = None
        # model radiator as in DTU lit: q/q_0 = (LMTD/LMTD_0) ** n
        self.LMTD_0 = 50
        self.T_sup_zero = 75     # [C]
        self.T_ret_zero = 65      # [C]
        self.T_amb_zero = 20
        self.q_zero = 2000        # [W]
        self.m_zero = None

        # self.__denominator = None
        self.__gmtd = None
        self.__lmtd = None
        self.calculate_radiator_properties()
        logging.info(f" Radiator object {self.name} created")

    @classmethod
    def from_dict(cls, d):
        """ classmethod to enable constructing an instance from configuration file.
        """
        return cls(name=d["name"], exp_rad=d["exp_rad"])

    @classmethod
    def from_yaml(cls, config_yaml: str):
        r = cls()
        d = load_config(config_yaml)
        rd = d.get("Radiator")
        r.name = rd.get("name")
        logging.info(f" Radiator '{r.name}' created \n")
        return r

    def calculate_radiator_properties(self):
        self.LMTD_0 = LMTD_radiator(self.T_sup_zero, self.T_ret_zero, self.T_amb_zero)
        self.Km = np.exp(np.log(self.q_zero) - (self.exp_rad*np.log(self.LMTD_0)))
        print(f"LMTD_0 = {self.LMTD_0} \u00b0C     Km = {self.Km}")
        if self.flow:
            self.m_zero = (self.q_zero) / (self.flow.cp * (self.T_sup_zero - self.T_ret_zero))
            print(f"mass_flow_zero: {self.m_zero:6f} [kg/s] ({self.m_zero * (1.0e6/self.flow.density):3f} ml/s)")

    def set_qzero(self, new_qzero: float):
        self.q_zero = new_qzero
        self.calculate_radiator_properties()

    def set_flow(self, new_flow: Flow):
        self.flow = new_flow
        self.calculate_radiator_properties()

    def set_exponent(self, e):
        self.exp_rad = e
        self.calculate_radiator_properties()

    def get_gmtd(self):
        return self.__gmtd

    def get_lmtd(self):
        return self.__lmtd

    def func_rad_gmtd(self, x):
        """model function for scipy.optimize.root().

        Args:
            x: vector with unknowns [self.q_dot, self.T_return]

        Returns:
            f : vector with model functions evaluated at x
            df : Jacobian (partial derivatives of model functions wrt x
        """
        self.__gmtd = GMTD_radiator(T_feed=self.T_supply, T_return=x[1], T_amb=self.T_amb)
        # set of nonlinear functions for root finding
        f = [x[0] - (self.Km * self.__gmtd ** self.exp_rad),
             x[0] - self.flow.heat_rate * (self.T_supply - x[1])]

        DeltaT_feed = self.T_supply - self.T_amb
        dTdt = -1.0 * self.Km * 0.5*self.exp_rad * np.float_power(DeltaT_feed, (0.5*self.exp_rad))
        dTdt *= np.float_power(x[1] - self.T_amb, (0.5*self.exp_rad) - 1)

        df = np.array([[1.0, dTdt],
                       [1.0, self.flow.heat_rate]])
        return f, df

    def func_rad_lmtd(self, x):
        """model function for scipy.optimize.root().

        Args:
            x: vector with unknowns [self.q_dot, self.T_return]

        Returns:
            f : vector with model functions evaluated at x
            df : Jacobian (partial derivatives of model functions wrt x
        """
        self.__lmtd = LMTD_radiator(T_feed=self.T_supply, T_return=x[1], T_amb=20.0)
        # set of nonlinear functions for root finding
        f = [x[0] - (self.Km * self.__lmtd ** self.exp_rad),
             x[0] - self.flow.heat_rate * (self.T_supply - x[1])]

        h1 = self.Km * self.exp_rad
        h1 *= self.__lmtd ** (self.exp_rad - 1.0)

        h2 = (self.T_supply - x[1]) / (x[1] - self.T_amb)
        denominator = np.log(self.T_supply - self.T_amb) - np.log(x[1] - self.T_amb)
        h2 -= denominator

        dTdt = -(h1 * h2) / (denominator * denominator)

        df = np.array([[1.0, dTdt],
                       [1.0, self.flow.heat_rate]])
        return f, df

    def update(self, func):
        """update roots [self.q_dot, self.T_return] with model function
           using scipy.optimize.root().

        Returns:
            None
        """
        if self.T_return is None:
            self.T_return = 0.5*(self.T_supply + self.T_amb)
        opt_res = root(func, [self.q_dot, self.T_return], jac=True, method='hybr')
        self.q_dot = opt_res.x[0]
        self.T_return = opt_res.x[1]


if __name__ == "__main__":
    deg = u"\u00b0"     # degree sign

    radiator = Radiator(name="Test", exp_rad=1.3)
    radiator.T_supply = 75.0                      # = T_top of buffer vessel
    T_bottom = 40                               # = T_bottom of buffer vessel

    radiator.T_amb = 20.0
    radiator.T_return = (radiator.T_supply + radiator.T_amb) / 2.0  # crude quess

    radiator.Km = 12.5
    radiator.flow.set_flow_rate(0.010e-3)  # m^3/s use SETTER to include update of heat-rate!!!!
    print(f"Heat rate: {radiator.flow.heat_rate} [W/K]")

    radiator.update(radiator.func_rad_lmtd)
    print(f"Q_dot: {radiator.q_dot}, T_return: {radiator.T_return}")
    print(f"LMTD {radiator.get_lmtd()}")
    print(f"GMTD {radiator.get_gmtd()}")
    print(f"radiator to room: {radiator.flow.heat_rate * (radiator.T_supply - radiator.T_return)} [W]")
    print(f"radiator to room: {radiator.Km * np.power(radiator.get_lmtd(), radiator.exp_rad)} [W] with Delta t = {radiator.get_lmtd()}")
    print(f"top-bottom: {radiator.flow.heat_rate * (radiator.T_supply - T_bottom)} [W]")
    print(f"back-to-bottom: {radiator.flow.heat_rate * (radiator.T_return - T_bottom)} [W]")

    # plot_corr_fact()
