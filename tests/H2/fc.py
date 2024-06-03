

import numpy as np
from H2_constants import *
from clausius import august
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use("Qt5Agg")


def butler_volmer(eta, i0, alpha_a, alpha_c, n, T, ilim=np.inf):
    """Compute the Butler-Volmer equation for a given overpotential
       including a limiting current density.

    Args:
        eta (array): overpotential E - E_0
        i0 (float): exchange current density
        alpha_a (float): anode transfer coefficient
        alpha_c: (float): cathode transfer coefficient
        n (int): number of electrons
        T (float): temperature in K
        ilim (float): limiting current density

    Returns:

    """
    i_a = i0 * np.exp(alpha_a * n * FARADAY * eta / (RGAS * T))
    i_c = i0 * np.exp(-alpha_c * n * FARADAY * eta / (RGAS * T))
    if ilim != np.inf:  # Apply limiting current density only if it is not infinity
        i_c = i_c / (1 + i_c/ilim)
    return i_a - i_c, i_a, i_c


def tafel_slope(alpha, n, T):
    """Compute the Tafel slope.

    Args:
        alpha (float): transfer coefficient
        n (int): number of electrons
        T (float): temperature in K

    Returns:

    """
    return (np.log(10.0) * RGAS * T) / (alpha * n * FARADAY)


class FuelCell():
    def __init__(self):
        self.Tc_cell = 60       # Cell temperature in [C]
        self.p_anode = 130000       # partial pressure anode [Pa]
        self.p_cathode = 200000       # partial pressure cathode [Pa]
        self.alpha = 0.2       # from Tafel eq., ideally 0.5 [-]
        self.n = 2             # number of electrons in reaction [-]
        self.zero_cur_mA = 5   # Current density exchange [mA· cm^-2] (NOT ZERO)
        self.Rint = 0.15       # internal resistance stack[Ohm * cm^2]
        self.C = 0.085          # amplification constant
        self.k = 1.1           # mass transport constant
        self.ilim_A = 1.4        # maximum limit for current of stack [A/cm^2]

        self.cur_mA = np.linspace(1.0, 1400, 51)
        # self.cur2D = np.vstack((self.time, self.cur_mA)).T

        self.p_sat_water = None
        self.p_hydrogen = None
        self.p_oxygen = None
        self.U_nernst = None
        self.U_act = None
        self.U_ohm = None
        self.U_conc = None
        self.U_total = None

    def calc_U_out(self):
        self.p_sat_water = august(self.Tc_cell + 273.15)  # Pa
        self.water2hyd()
        self.water2ox()
        self.nernst()
        self.calc_U_act()
        self.calc_U_ohm()
        self.calc_U_conc()
        self.U_total = self.U_nernst - self.U_act - self.U_ohm - self.U_conc

    def calc_U_act(self):
        Tk = self.Tc_cell + 273.15
        factor = (RGAS * Tk) / (self.alpha * self.n * FARADAY)
        ln_i_over_i_zero = np.log(self.cur_mA / self.zero_cur_mA)
        self.U_act = factor * ln_i_over_i_zero

    def calc_U_ohm(self):
        self.U_ohm = self.cur_mA * 0.001 * self.Rint

    def calc_U_conc(self):
        i_A = self.cur_mA * 0.001
        factor = self.C * np.power(i_A, self.k)
        self.U_conc = -factor * np.log(1.0 - (i_A / self.ilim_A))

    def water2hyd(self):
        Tk = self.Tc_cell + 273.15
        exponent = -(1.635 * self.cur_mA) / np.power(Tk, 1.334)
        self.p_hydrogen = np.exp(exponent) * (self.p_anode/self.p_sat_water)
        self.p_hydrogen += -1.0
        self.p_hydrogen *= 0.5 * self.p_sat_water

    def water2ox(self):
        Tk = self.Tc_cell + 273.15
        exponent = -(4.192 * self.cur_mA) / np.power(Tk, 1.334)
        self.p_oxygen = np.exp(exponent) * (self.p_cathode / self.p_sat_water)
        self.p_oxygen += -1.0
        self.p_oxygen *= self.p_sat_water

    def nernst(self):
        Tk = self.Tc_cell + 273.15
        factor = (RGAS * Tk) / (self.n * FARADAY)
        ln_K = np.log(self.p_hydrogen * np.sqrt(self.p_oxygen))
        self.U_nernst = U_IDEAL - (factor * ln_K)


    def plot_pol_curve(self):
        fig, ax = plt.subplots(2, 1, figsize=(8, 10), sharex='all')
        ax[0].plot(self.cur_mA / 1000, U_IDEAL*np.ones_like(self.cur_mA), 'r.-', label="U_ideal")
        ax[0].plot(self.cur_mA / 1000, self.U_total, 'b.-', label="U_total")
        ax[0].grid()
        # ax[0].legend()
        ax[0].set_ylim([0, 1.3])
        # ax[0].set_xlabel('Current density [A/cm^2]')
        ax[0].set_ylabel('Voltage [V]')

        ax[1].plot(self.cur_mA / 1000, self.U_nernst, 'r.-', label="U_nernst")
        ax[1].plot(self.cur_mA / 1000, self.U_act, 'm.-', label="U_act")
        ax[1].plot(self.cur_mA / 1000, self.U_ohm, 'k.-', label="U_ohmic")
        ax[1].plot(self.cur_mA / 1000, self.U_conc, 'g.-', label="U_conc")
        ax[1].grid()
        ax[1].legend()
        # ax[1].set_ylim([0, 1.3])
        ax[1].set_xlabel('Current density [A/cm^2]')
        ax[1].set_ylabel('Voltage [V]')
        plt.show()

np.arcsinh()

if __name__ == "__main__":
    fc = FuelCell()
    fc.calc_U_out()
    fc.plot_pol_curve()

    # Input parameters in the sidebar
    i0 = 1.0  # Exchange current density (i0) [A/m^2]
    T = 300  # Temperature (T) [K]
    n = 1  # Number of electrons (n)
    alpha_a = 0.5  # Anodic charge transfer coefficient (alpha_a)
    print(f'Tafel Slope (anodic): {tafel_slope(alpha_a, n, T):.2f} V/dec')
    alpha_c = 0.5  # Cathodic charge transfer coefficient (alpha_c)
    print(f'Tafel Slope (cathodic): {tafel_slope(alpha_c, n, T):.2f} V/dec')
    eta_min = -0.25
    eta_max = 0.25  # Overpotential range (eta) [V]

    # ilim = 1.0  # Limiting current density (i_lim) [A/m^2]', 0.01, 10.0, 1.0)
    ilim = np.inf  # Assign a very large number to ilim when the checkbox is not checked

    # Generate potential values
    eta = np.linspace(eta_min, eta_max, 1000)
    i, ia, ic = butler_volmer(eta, i0, alpha_a, alpha_c, n, T, ilim)

    # Plot figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Linear scale
    ax1.plot(i, eta, 'C2-', label='Butler-Volmer')
    ax1.plot(ia, eta, 'C1--', linewidth=2, label='$i_a$ (Anodic)')  # LaTeX-style label and thicker line
    ax1.plot(-ic, eta, 'C0--', linewidth=2, label='$i_c$ (Cathodic)')  # Thicker line
    ax1.set_xlim(-i0*100, i0*100)  # This line limits the x-axis range
    ax1.set_title('Linear Scale')
    ax1.set_xlabel('Current Density (A/m^2)')
    ax1.set_ylabel('Potential (V)')
    ax1.legend()

    # Logarithmic scale
    ax2.plot(np.abs(i), eta, 'C2-',label='Butler-Volmer')
    ax2.plot(np.abs(ia), eta, 'C1--', linewidth=2, label='$i_a$ (Anodic)')  # LaTeX-style label and thicker line
    ax2.plot(np.abs(ic), eta, 'C0--', linewidth=2, label='$i_c$ (Cathodic)')  # Thicker line
    ax2.set_xlim(i0/1000, max(np.abs(i)))  # This line limits the x-axis range
    ax2.set_xscale('log')
    ax2.set_title('Logarithmic Scale')
    ax2.set_xlabel('Current Density (A/m^2)')
    ax2.set_ylabel('Potential (V)')
    ax2.legend()

    plt.tight_layout()
    plt.show()






