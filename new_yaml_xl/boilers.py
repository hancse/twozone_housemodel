
import numpy as np
from simple_pid import PID
import dvg_pid_controller

class gasboiler(PID):
    def __init__(self, kp, ki, kd, T_setpoint,
                 T_node, T_amb, dead_band, P_max):
        super.__init__(kp, ki, kd, T_setpoint)
        self.P_max = P_max
        self.T_amb = T_amb
        self.T_node = T_node
        self.dead_band = dead_band
        self.lower_db = T_node - 0.5 * dead_band
        self.upper_db = T_node + 0.5 * dead_band
        self.output_limits(0, P_max)
        self.Power = 0.0
        self.

    """
    Todo: 
        hysteresis dead band implementation
        bring PID object outside to avoid generation of 8760 PID controllers
        T_setpoint in summer season outside this function
        convert to class; derive PID object from  standard class and add hysteresis
        use yield statement
        https://colab.research.google.com/github/jckantor/CBE30338/blob/master/docs/04.01-Implementing_PID_Control_with_Python_Yield_Statement.ipynb#scrollTo=x_FzBMUc8pUN
 
    """

    def hyst(self, output):
        if self.lower_db < self.T_node < self.upper_db:
            self.set_auto_mode(False)
            output = 0.0
        else:
            self.set_auto_mode(True, last_output=self.Power)

    def update(self, time_step):
        Q = self.__call__(self.T_node - self.T_setpoint, time_step)

        yield Q


if __name__ == "__main__":
    g = gasboiler( 20, 20.1, 0, 10000, kp= 2000)
    # Q = g.
    print ("Q_boiler: ", Q)


    """
    def gasboiler(T_setpoint, T_node, T_amb, P_max, kp, ki=0, kd=0):

    # Q_boiler = pid(T_node)

    # if  T_setpoint - 0.5* dead_band < T_node  < T_setpoint + 0.5* dead_band:

    if (T_setpoint < 10) | (T_amb > 15):
        # implement summer period by a ridiculously low value of T_setpoint
        pid.auto_mode = False
        Q_boiler = 0.0
    else:
        pid.auto_mode = True
        Q_boiler = pid(T_node)
        if  Q_boiler < 0.15*P_max:
            Q_boiler = 0.0

    return Q_boiler
    """