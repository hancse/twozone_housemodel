import numpy as np
# https://simple-pid.readthedocs.io/en/latest/simple_pid.html#module-simple_pid.PID
from simple_pid import PID
import time
#from Temperature_SP import simple_thermostat


class gasboiler(PID):
    """class for modelling PID-controlled gasboiler

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

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
            T_amb (float):       outdoor temperature
            dead_band (float):   deadband in [C]
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
        self.dead_band = dead_band

        self.lower_db = T_setpoint - 0.5 * dead_band
        self.upper_db = T_setpoint + 0.5 * dead_band
        #self.output_limits(0, P_max)
        self.Power = 0.0
        self.db_state = True


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
        #If the hysteresis is upward, and withing the control band, send minimal power to the room
        if (self.lower_db < self.T_node < self.upper_db) & (self.db_state == True):
            self.set_auto_mode(False)
            output = self.P_min
        #If the hysteresis is downward, and withing the control band, send no power
        elif (self.lower_db < self.T_node < self.upper_db) & (self.db_state == False):
            self.set_auto_mode(False)
            output = 0
        #If T_node is over the deadband, dont send power and reset hysteresis state
        elif self.T_node > self.upper_db:
            self.db_state = False
            output = 0
        #If T_node is under the deadband, send power with PID control set hysteresis state
        elif self.T_node < self.lower_db:
            self.db_state = True
            self.set_auto_mode(True, last_output=self.Power)
            output = self.__call__(self.T_node)
        return output

if __name__ == "__main__":
    g = gasboiler(kp=100, ki=100, kd=0, T_setpoint=19.9, T_node=15, T_amb=10, dead_band=2, P_max=10000, P_min=4000)
    g.output_limits = (4000, 10000)
    #SP = simple_thermostat(t_on, t_off, T_day, T_night)
    #SP_sim = SP[0:days_sim * 24]
    print(g.update)
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