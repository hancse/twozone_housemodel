import numpy as np
import matplotlib

# import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D

from housemodel.sourcesink.heatpumps.NTA8800_Q.defrost8800 import frost_factor_8800
from housemodel.sourcesink.heatpumps.NTA8800_Q.HPQ9 import (calc_WP_general,
                                                            plot_plane, plot_lines)

from housemodel.sourcesink.boilers.boilers import SimpleGasBoiler
from housemodel.tools.new_configurator import load_config
from housemodel.basics.flows import Flow
import logging

# matplotlib.use('TkAgg')
matplotlib.use('Qt5Agg')


class HeatpumpNTANew:
    """class for modelling PID-controlled heatpump.

    internally, this class expresses power in kW!
    """

    def __init__(self, name="NTA"):
        self.name = name
        self.T_evap = 20  # evaporator temperature (Toutdoor)
        # self.T_evap_in = None
        # self.T_evap_out = None

        self.T_cond_or = 20  # condenser temperature (outdoor reset)
        self.T_cond_out = 20  # condenser temperature (hot side)
        self.T_cond_in = 20  # condenser temperature (cold side)

        self.cal_T_evap = np.array([7, 7, -7])
        self.cal_T_cond = np.array([35, 55, 35])
        self.cal_COP_val = np.zeros_like(self.cal_T_cond)
        self.cal_Pmax_val = np.zeros_like(self.cal_T_cond)
        self.c_coeff = np.zeros_like(self.cal_T_cond)
        self.p_coeff = np.zeros_like(self.cal_T_cond)
        self.COP_A2W35 = None  # for frost COP reduction
        self.Pmax_A2W35 = None  # for frost P reduction
        self.Pmax_kW = None  # for clipping
        self.P_HP_kW = None
        self.P_HP_W = None
        self.COP = None
        self.Pmin_kW = None
        self.T_max_cond_out = None
        self.frost_power_correction = True

        self.flow = None  # Flow object, initialized as None
        logging.info(f" HeatPumpNew object {self.name} created")

    @classmethod
    def from_dict(cls, d):
        """ classmethod to enable constructing an instance from configuration file.
        """
        return cls(name=d["name"])

    @classmethod
    def from_yaml(cls, config_yaml: str):
        hp = cls()
        d = load_config(config_yaml)
        hpd = d.get("Heatpump")
        hp.name = hpd.get("name")
        logging.info(f" Heatpump '{hp.name}' created \n")
        return hp

    def calculate_heatpump_properties(self):
        pass

    def set_flow(self, new_flow: Flow):
        self.flow = new_flow
        self.calculate_heatpump_properties()

    def set_cal_val(self, cop_val: list, pmax_val: list):
        self.cal_COP_val = np.array(cop_val)
        self.cal_Pmax_val = np.array(pmax_val)
        self.c_coeff = calc_WP_general(self.cal_T_evap, self.cal_T_cond,
                                       self.cal_COP_val, order=1)
        self.p_coeff = calc_WP_general(self.cal_T_evap, self.cal_T_cond,
                                       self.cal_Pmax_val, order=1)

    def set_A2W35(self, cop_val: float, pmax_val: float):
        self.COP_A2W35 = np.array(cop_val)
        self.Pmax_A2W35 = np.array(pmax_val)

    def update(self):
        """Update COP and Power as function of (T_cond, T_evap) according to NTA8800

        Returns:
        """
        self.COP = self.c_coeff[0] + \
                   self.c_coeff[1] * self.T_evap + \
                   self.c_coeff[2] * self.T_cond_out
        self.P_HP_kW = self.p_coeff[0] + \
                       self.p_coeff[1] * self.T_evap + \
                       self.p_coeff[2] * self.T_cond_out  # in kW

        # COP defrost correction
        frost_f = frost_factor_8800([self.T_evap])  # input = list or 1-dim array
        self.COP *= frost_f
        if(self.frost_power_correction == True):
            self.P_HP_kW *= frost_f
        self.P_HP_kW = np.clip(self.P_HP_kW, 0, self.Pmax_kW)
        self.P_HP_W = 1000.0 * self.P_HP_kW.item()

    def adjust(self):
        """adjust T_cond_out if P_HP is not sufficient to reach T_cond_or.
        """
        if self.flow:
            P_req_W = self.flow.heat_rate * (self.T_cond_or - self.T_cond_in)
            P_req_kW = P_req_W/1000

            P_HP_or_kW = self.p_coeff[0] + \
                         self.p_coeff[1] * self.T_evap + \
                         self.p_coeff[2] * self.T_cond_or     # in kW

            if (P_req_kW > self.P_HP_kW) or (P_req_kW > P_HP_or_kW):
                self.T_cond_out = (self.p_coeff[0] +
                                   self.p_coeff[1]*self.T_evap +
                                   self.flow.heat_rate/1000*self.T_cond_in)
                self.T_cond_out /= ((self.flow.heat_rate/1000) - self.p_coeff[2])
            else:
                self.T_cond_out = self.T_cond_or
            self.update()
            # logging.info(f"P_req_W {P_req_W}   P_HP_W {self.P_HP_W}")
            # logging.info(f"T_cond_out {self.T_cond_out}   T_cond_in {self.T_cond_in} \n")


class HybridHPNew:
    """class for modelling a Hybrid Heating system with a heat pump and a gas boiler."""

    def __init__(self, boiler, heat_pump):
        self.boiler = boiler
        self.heat_pump = heat_pump

    def update(self, power_requested, Te, Tc):
        self.heat_pump.cop = self.heat_pump.update(Te, Tc)[0]
        self.heat_pump.power = self.heat_pump.update(Te, Tc)[1]
        self.heat_pump.power = self.heat_pump.power * 1000
        self.boiler.power = self.boiler.update(power_requested - self.heat_pump.power)
        return self.heat_pump.power, self.boiler.power, self.heat_pump.cop


if __name__ == "__main__":
    print("Heat Pump Model (L/W) according to NTA8800:2020, Appendix Q9")
    nta = HeatpumpNTANew()
    nta.set_cal_val([4.0, 3.0, 2.5], [6.0, 2.0, 3.0])

    plot_plane(nta.cal_T_evap, nta.cal_T_cond,
               nta.cal_COP_val, nta.c_coeff, 'COP', 1.0, 5.0)
    plot_plane(nta.cal_T_evap, nta.cal_T_cond,
               nta.cal_COP_val, nta.c_coeff, 'Power', 0.0, 10.0)

    nta.T_evap = 7.0
    nta.T_cond_out = 35.0
    nta.update()
    print(f"COP: {nta.COP}, P_HP_kW {nta.P_HP_kW}")

    nta.T_evap = 7.0
    nta.T_cond_out = 55.0
    nta.update()
    print(f"COP: {nta.COP}, P_HP_kW {nta.P_HP_kW}")

    nta.T_evap = -7.0
    nta.T_cond_out = 35.0
    nta.update()
    print(f"COP: {nta.COP}, P_HP_kW {nta.P_HP_kW}")

    Tin_space = np.linspace(-20, 20, 41, endpoint=True)
    COP_35 = nta.c_coeff[0] + nta.c_coeff[1] * Tin_space + nta.c_coeff[2] * 35.0
    COP_45 = nta.c_coeff[0] + nta.c_coeff[1] * Tin_space + nta.c_coeff[2] * 45.0
    COP_55 = nta.c_coeff[0] + nta.c_coeff[1] * Tin_space + nta.c_coeff[2] * 55.0

    P_35 = nta.p_coeff[0] + nta.p_coeff[1] * Tin_space + nta.p_coeff[2] * 35.0
    P_45 = nta.p_coeff[0] + nta.p_coeff[1] * Tin_space + nta.p_coeff[2] * 45.0
    P_55 = nta.p_coeff[0] + nta.p_coeff[1] * Tin_space + nta.p_coeff[2] * 55.0

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

    """
    print("Hybrid Heat Pump Model with a boiler and a heat pump according to NTA8800:2020, Appendix Q9")

    nta = HeatpumpNTANew()
    nta.set_cal_val([4.0, 3.0, 2.5], [6.0, 2.0, 3.0])

    g = SimpleGasBoiler(P_max=10000, P_min=1500)
    nta.T_evap = 7.0
    nta.T_cond = 35.0
    nta.update()
    myhybridHP = HybridHPNew(g, nta)
    hp_power, boilerpower, cop = myhybridHP.update(7500, 7, 35)
    print(f"hp_power: {hp_power}, boilerpower: {boilerpower}, COP: {cop}")
    """

