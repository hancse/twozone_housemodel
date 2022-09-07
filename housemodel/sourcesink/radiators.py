# https://learn.openenergymonitor.org/sustainable-energy/building-energy-model/radiatormodel
# https://www.buildingservicesindex.co.uk/entry/136540/AEL-Heating-Solutions-Ltd/How-to-calculate-the-delta-T-for-a-radiator/

import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate
from housemodel.tools.house_tools import LMTD, LMTD_radiator
import numpy as np

matplotlib.use("Qt5Agg")


def calc_corr_fact(delta_t):
    dt_model = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    cf_model = [0.05, 0.123, 0.209, 0.304, 0.406, 0.515, 0.629, 0.748, 0.872, 1.0, 1.132, 1.267, 1.406, 1.549, 1.694]
    f = interpolate.interp1d(dt_model, cf_model)
    corr_fact = f(delta_t)
    return corr_fact


def calc_log_mean_diff(Tinlet, Treturn, Tamb):
    lm = LMTD(Tinlet, Treturn, Tamb, Tamb,
              flowpattern='cross')
    return lm


def calc_log_mean_diff_rad(Tinlet, Treturn, Tamb):
    lm = LMTD_radiator(Tinlet, Treturn, Tamb)
    return lm


def calc_Q_rad(feed, ret, amb, corr, exp):
    lmtd_rad = calc_log_mean_diff_rad(feed, ret, amb)
    return (corr * lmtd_rad) ** exp


def calc__mean_diff(Tinlet, Treturn, Tamb):
    lm = np.mean([Tinlet, Treturn]) - Tamb
    return lm


if __name__ == "__main__":
    deg = u"\u00b0"
    lm_ref = calc_log_mean_diff(75, 65, 20)
    print(f"Reference LMTD: {lm_ref}")
    dt_ref = calc__mean_diff(75, 65, 20)
    print(f"Reference Delta_T: {dt_ref}")

    dt_rad = calc_log_mean_diff_rad(75, 65, 20)
    print(f"Radiator Delta_T: {dt_rad} {deg}C")

    Delta_T = [20, 25, 30, 35, 40, 45, 50]
    cf = [0.3, 0.41, 0.52, 0.63, 0.75, 0.87, 1.0]

    Delta_T_2 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    cf_2 = [0.05, 0.123, 0.209, 0.304, 0.406, 0.515, 0.629, 0.748, 0.872, 1.0, 1.132, 1.267, 1.406, 1.549, 1.694]

    cf_test = calc_corr_fact(dt_rad)
    print(f"correction factor: {cf_test}")
    fig, ax = plt.subplots()
    ax.plot(Delta_T, cf, 'b-')
    ax.plot(Delta_T_2, cf_2, 'r--')
    ax.plot(dt_rad, cf_test, 'og')
    ax.grid(True)
    ax.set_ylim(0, 1.75)
    plt.show()
