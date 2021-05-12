
import numpy as np
# https://simple-pid.readthedocs.io/en/latest/simple_pid.html#module-simple_pid.PID
from simple_pid import PID
import dvg_pid_controller


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
    def __init__(self, kp, ki, kd, T_setpoint):
                # T_node, T_amb, dead_band, P_max):
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
        """
        super().__init__(kp, ki, kd, T_setpoint)
        self.P_max = P_max
        """str: Docstring *after* attribute, with type specified."""
        self.T_amb = T_amb
        """str: Docstring *after* attribute, with type specified."""
        self.T_node = T_node
        """str: Docstring *after* attribute, with type specified."""
        self.dead_band = dead_band

        self.lower_db = T_setpoint - 0.5 * dead_band
        self.upper_db = T_setpoint + 0.5 * dead_band
        self.output_limits(0, P_max)
        self.Power = 0.0


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
    g = gasboiler(kp=2000, ki=10, kd=0, T_setpoint=20)
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