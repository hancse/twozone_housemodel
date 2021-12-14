from housemodel.sourcesink.boilers_without_PID import GasBoiler
from housemodel.controls.ivPID.PID import PID
from housemodel.sourcesink.heatpumps.Hybrid_HP import HybridHP
from housemodel.sourcesink.heatpumps.Heatpump_HM import Heatpump_NTA, calc_WP_general

class HeatingDevice:
    """Describes the type of heating device e.g. boiler or heat pump"""
    def __init__(self):
        self.maximum_power = None
        self.minimum_power = None

class Controller:
    """Desribes the type of controller e.g. PID or MPC"""
    def __init__(self):
        self.output = None

    def update(self) -> float:
        pass

class HeatingSystem:
    """Abstract class for modelling a Heating system.
    """
    def __init__(self, mysource: HeatingDevice, mycontrol: Controller):
        self.heat_source = mysource
        self.controller = mycontrol

    def compute_boiler_power(self) -> float:
        """Compute power output of the heating system"""
        self.controller.update(10)
        return self.heat_source.update(self.controller.output)

    def compute_heat_pump_power(self, Te, Tc) -> float:
        return self.heat_source.update(Te, Tc)

    def compute_hybrid_power(self, Tindoor, Te, Tc):
        self.controller.update(Tindoor)
        return self.controller.output, self.heat_source.update(self.controller.output, Te, Tc)

if __name__ == "__main__":
    myboiler = GasBoiler(10000, 1500)
    myPID = PID(2000, 0, 0)
    hs = HeatingSystem(myboiler, myPID)
    print(hs.compute_boiler_power())

    nta = Heatpump_NTA()
    nta.set_cal_val([4.0, 3.0, 2.5], [6.0, 2.0, 3.0])

    nta.c_coeff = calc_WP_general(nta.cal_T_evap, nta.cal_T_cond,
                                  nta.cal_COP_val, order=1)

    nta.p_coeff = calc_WP_general(nta.cal_T_evap, nta.cal_T_cond,
                                  nta.cal_Pmax_val, order=1)
    hs2 = HeatingSystem(nta, None)
    test = hs2.compute_heat_pump_power(10, 50)[1]
    print(test)


    myHybridHP = HybridHP(myboiler, nta)
    hs3 = HeatingSystem(myHybridHP, myPID)
    asdf1, [asdf2, asdf3, asdf4] = hs3.compute_hybrid_power(10, 7, 35)

    print(asdf1, asdf2, asdf3, asdf4)


