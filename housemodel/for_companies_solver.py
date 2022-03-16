from housemodel.simulation import Solver

import numpy as np
from housemodel.controls.ivPID.PID import PID


class SolverForCompanies(Solver):
    def __init__(self, simulation, *args, **kwargs):
        super().__init__(simulation, *args, **kwargs)
        controller_cfg = simulation.house_param['controller']
        control_parameters = np.zeros(3)
        control_parameters[0] = controller_cfg['kp']
        control_parameters[1] = controller_cfg['ki']
        control_parameters[2] = controller_cfg['kd']
        self.control_parameters = control_parameters

    def solve(self, cap_mat_inv, cond_mat, q_vector, SP_T, control_interval, *args, **kwargs):
        # Controller initialization
        # heatingPID = PID(Kp=5000, Ki=0, Kd=0, beta=1, MVrange=(0, 12000), DirectAction=False)
        # heating = 0
        kp = self.control_parameters[0]
        ki = self.control_parameters[1]
        kd = self.control_parameters[2]

        pid = PID(kp, ki, kd, self.t[0])

        pid.SetPoint = 17.0
        pid.setSampleTime(0)
        pid.setBounds(0, 12000)
        pid.setWindup(12000 / control_interval)

        self.pid = pid

        super().solve(cap_mat_inv, cond_mat, q_vector, SP_T, control_interval)

        sim = self.simulation
        sim.Qinst = self.q_vector[2, :] / 1000
        self.data = (sim.time_sim, self.Tair, self.Twall, self.Tradiator, sim.Qinst)

    def step(self, i):
        # here comes the "arduino style" controller
        self.pid.SetPoint = self.SP_T[i]
        self.pid.update(self.Tair[i], self.t[i])
        self.q_vector[2, i] = self.pid.output

        super().step(i)

    def get_model_function(self):
        # not needed if using default model function
        return super().get_model_function()

