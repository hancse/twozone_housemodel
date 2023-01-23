import numpy as np
from scipy.interpolate import interp1d


class PowerSource:
    """class for storage of contributions to q-vector
    contributions are stored
    - in power values [W]
    - in temperature values [K], to be multiplied by conductivity in [W/K]
    """
    def __init__(self, name=""):
        self.name = name
        self.connected_to: []
        self.powervalues: None  # in [W]
        self.tempvalues: None  # in [W]

    def interpolate_power(self, time_sim, control_interval):
        interp_func = interp1d(time_sim, self.powervalues, fill_value='extrapolate')
        self.powervalues = interp_func(np.arange(0, time_sim[-1] + (6 * 600),
                                                 control_interval))


if __name__ == "__main__":
    s = PowerSource("Test")
    days_sim = 2
    time_sim = np.arange(days_sim*24) * 3600
    s.powervalues = np.arange(days_sim*24) * 0.1
    control_interval = 600
    s.interpolate_power(time_sim, control_interval)

    print()
