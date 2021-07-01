"""
See Also
    https://pypi.org/project/simple-pid/
    https://simple-pid.readthedocs.io/en/latest/
"""

import numpy as np
# https://simple-pid.readthedocs.io/en/latest/simple_pid.html#module-simple_pid.PID
from simple_pid import PID
# import time
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
# from Temperature_SP import simple_thermostat


class GasBoiler(PID):
    """class for modelling PID-controlled gas boiler.

    """

    def __init__(self, kp, ki, kd, T_setpoint,
                 T_node, T_amb, dead_band, P_max, P_min):
        """

        Args:
            kp (float):          proportional PID gain [W/C]
            ki (float):          integral PID parameter
            kd (float):          derivative PID parameter
            T_setpoint (float):  setpoint (SP) temperature
            T_node (float):      process value (PV) temperature
            T_amb (float):       outdoor temperature used for outdoor
            dead_band (float):   dead band in [C]
            P_max (float):       maximum power of boiler [W]
            P_min (float):       minimum power of boiler [W]
        """
        super().__init__(kp, ki, kd, T_setpoint)
        self.P_max = P_max
        self.P_min = P_min
        """str: Docstring *after* attribute, with type specified."""
        self.T_amb = T_amb
        """str: Docstring *after* attribute, with type specified."""
        self.T_node = T_node
        """str: Docstring *after* attribute, with type specified."""
        self.setpoint = T_setpoint
        # self.dead_band = dead_band

        self.lower_db = self.setpoint - 0.5 * dead_band
        self.upper_db = self.setpoint + 0.5 * dead_band
        # self.output_limits(0, P_max)
        self.Power = 0.0
        self.isEnabled: bool = True
        self.output = 0.0
        self.current_step = 0.0

    """
    Todo: 
        hysteresis dead band implementation
        bring PID object outside to avoid generation of 8760 PID controllers
        T_setpoint in summer season outside this function
        convert to class; derive PID object from  standard class and add hysteresis
        use yield statement
        https://colab.research.google.com/github/jckantor/CBE30338/blob/master/docs/04.01-Implementing_PID_Control_with_Python_Yield_Statement.ipynb#scrollTo=x_FzBMUc8pUN
 
    """

    def update(self):
        self.output = self.__call__(self.T_node, self.current_step)  # should happen always

        # If temperature is above dead band once, disable the PID boiler, zero power
        if self.T_node > self.upper_db:
            if self.isEnabled:
                self.isEnabled = False
            self.output = 0  # overrule P_min

        # If the temperature has decreased from above into the dead band, zero power
        elif (self.lower_db <= self.T_node <= self.upper_db) & (self.isEnabled is False):
            # self.set_auto_mode(False)
            self.output = 0  # overrule P_min

        # If the temperature has increased from below into the dead band, send minimal power
        elif (self.lower_db <= self.T_node <= self.upper_db) & (self.isEnabled is True):
            if self.auto_mode:
                self.set_auto_mode(False)
            self.output = self.P_min

        # If temperature is below dead band once, enable the PID boiler
        elif self.T_node < self.lower_db:
            if not self.isEnabled:
                self.isEnabled = True
            if not self.auto_mode:
                self.set_auto_mode(True)
            if self.output < self.P_min:
                self.output = self.P_min  # without this statement dip around lower dead band limit


if __name__ == "__main__":
    g = GasBoiler(kp=1000, ki=0, kd=0,
                  T_setpoint=20, T_node=15, T_amb=10,
                  dead_band=2, P_max=10000, P_min=1500)
    g.output_limits = (0, g.P_max)  # no offset: P_min = 0
    # g.output_limits = (g.P_min, g.P_max)  # not what we want: P_min is offset for PID output

    # testing Gasboiler class
    g.current_step = 0.1
    tm = np.arange(0, 20, 0.1)
    amplitude = 5 * np.sin(tm) + 18
    hysteresis = []
    state = []
    for i in range(len(tm)):
        g.T_node = amplitude[i]
        g.update()
        hysteresis.append(g.output)
        state.append(g.isEnabled)

    result = np.vstack((amplitude, np.array(hysteresis), state))

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(tm, amplitude, '.b-')
    ax[1].plot(tm, hysteresis, '.b-')
    ax[0].grid(True)
    ax[1].grid(True)
    plt.show()
