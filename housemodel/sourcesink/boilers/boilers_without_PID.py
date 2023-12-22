
class GasBoiler:
    """class for modelling simple gas boiler without internal controller.

    Attributes:
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

    def __init__(self, P_max, P_min):
        self.P_max = P_max
        self.P_min = P_min

    def update(self, requested_power):
        """updates heating power after checking limits.

        implementation can be improved.

        Args:
            requested_power (float): heating power.

        Returns:
            self.P_max (can be changed)
        """
        if requested_power < self.P_min:
            return 0
        if requested_power < self.P_min:
            return self.P_min
        if (self.P_min < requested_power and requested_power < self.P_max):
            return requested_power
        return self.P_max


if __name__ == "__main__":
    g = GasBoiler(P_max=10000, P_min=1500)
    print(g.update(5000))
    print(g.update(1000))
    print(g.update(15000))

