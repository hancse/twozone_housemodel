import numpy as np

from housemodel.simulation import Solver

from housemodel.controls.ivPID.PID import PID
from housemodel.sourcesink.heatpumps.NTA8800_Q.HPQ9 import calc_WP_general
from housemodel.sourcesink.heatpumps.Heatpump_HM import Heatpump_NTA
from housemodel.controls.heating_curves import hyst, outdoor_reset


class SolverHeatPumpNTA8800PartLoad(Solver):
    def __init__(self, simulation, *args, **kwargs):
        super().__init__(simulation, *args, **kwargs)
        controller_cfg = simulation.house_param['controller']
        control_parameters = np.zeros(3)
        control_parameters[0] = controller_cfg['kp']
        control_parameters[1] = controller_cfg['ki']
        control_parameters[2] = controller_cfg['kd']
        self.control_parameters = control_parameters

    def solve(self, cap_mat_inv, cond_mat, q_vector, SP_T, control_interval, *args, **kwargs):
        kp = self.control_parameters[0]
        ki = self.control_parameters[1]
        kd = self.control_parameters[2]
        t = self.t

        pid = PID(kp, ki, kd, t[0])

        pid.SetPoint = 17.0
        pid.setSampleTime(0)
        pid.setBounds(0, 12000)
        pid.setWindup(12000 / control_interval)

        self.pid = pid

        # Heat pump initialization
        nta = Heatpump_NTA()
        nta.Pmax = 12
        nta.set_cal_val([4.0, 3.0, 2.5], [6.0, 2.0, 3.0])

        nta.c_coeff = calc_WP_general(nta.cal_T_evap, nta.cal_T_cond,
                                      nta.cal_COP_val, order=1)

        nta.p_coeff = calc_WP_general(nta.cal_T_evap, nta.cal_T_cond,
                                      nta.cal_Pmax_val, order=1)
        self.nta = nta

        sim = self.simulation
        self.water_temp = np.zeros_like(sim.T_outdoor_sim)
        self.cop_hp = np.zeros_like(sim.T_outdoor_sim)
        self.p_hp = np.zeros_like(sim.T_outdoor_sim)
        self.cop_hp_corrected = np.zeros_like(sim.T_outdoor_sim)

        # define hysteresis object for heat pump
        self.hp_hyst = hyst(dead_band=0.5, state=True)

        super().solve(cap_mat_inv, cond_mat, q_vector, SP_T, control_interval)

        sim.controlled_power = self.q_vector[2, :]
        sim.cop_hp = self.cop_hp
        sim.cop_hp_corrected = self.cop_hp_corrected
        sim.p_hp = self.p_hp
        sim.water_temp = self.water_temp

        self.data = (sim.time_sim, self.Tair, self.Twall, self.Tradiator, sim.controlled_power, sim.water_temp,
                     sim.cop_hp, sim.cop_hp_corrected)

    def step(self, i):
        sim = self.simulation

        # p_hp = 0
        # determine new setting for COP and heat pump power
        self.water_temp[i] = outdoor_reset(sim.T_outdoor_sim[i], 0.7, 20)
        self.cop_hp[i], self.p_hp[i] = self.nta.update(sim.T_outdoor_sim[i], self.water_temp[i])
        self.p_hp[i] = self.p_hp[i]*1000

        self.pid.SetPoint = self.SP_T[i]
        self.pid.update(self.Tair[i], self.t[i])
        controlled_value = self.pid.output
        controlled_value = np.clip(controlled_value, self.p_hp[i]/2, self.p_hp[i])

        # incorporate hysteresis to control
        controlled_value = self.hp_hyst.update(self.Tair[i], self.SP_T[i], controlled_value)

        #Calculate Part Load Ratio, and Part Load Factor for COP correction SOurce: The impact of the work under partial load on the energy efficiency of an air-to-water heat pump
        PLR = controlled_value/self.p_hp[i]
        Cc = 0.9
        PLF = PLR/(PLR*Cc+(1-Cc))
        self.cop_hp_corrected[i] = PLF*self.cop_hp[i]

        # update q_vector
        self.q_vector[2, i] = controlled_value

        super().step(i)

