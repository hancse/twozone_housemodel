from housemodel.sourcesink.heatpumps.Heatpump_HM import Heatpump_NTA
from housemodel.sourcesink.boilers.boilers_without_PID import GasBoiler
from housemodel.sourcesink.heatpumps.NTA8800_Q.HPQ9 import calc_WP_general


class HybridHP:
    """class for modelling a Hybrid Heating system with a heat pump and a gas boiler.

    """

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
    print("Hybrid Heat Pump Model with a boiler and a heat pump according to NTA8800:2020, Appendix Q9")

    nta = Heatpump_NTA()
    nta.set_cal_val([4.0, 3.0, 2.5], [6.0, 2.0, 3.0])
    nta.c_coeff = calc_WP_general(nta.cal_T_evap, nta.cal_T_cond,
                                  nta.cal_COP_val, order=1)
    nta.p_coeff = calc_WP_general(nta.cal_T_evap, nta.cal_T_cond,
                                  nta.cal_Pmax_val, order=1)

    g = GasBoiler(P_max=10000, P_min=1500)
    c, p = nta.update(7, 35)
    myhybridHP = HybridHP(g, nta)
    hp_power, boilerpower, cop = myhybridHP.update(7500, 7, 35)
    print(f"hp_power: {hp_power}, boilerpower: {boilerpower}, COP: {cop}")
