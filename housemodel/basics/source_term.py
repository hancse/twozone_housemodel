
import numpy as np
from scipy.interpolate import interp1d


class SourceTerm:
    """class for storage of additive contributions to q-vector "source terms"
    contributions are stored
    - in thermal power values [W]
    - in temperature values [K], to be multiplied by thermal conductivity in [W/K]
    """

    def __init__(self, name=""):
        self.name = name
        self.connected_to = []
        self.values = None  # in [W] or [K]

    # TODO: 6 * 600 geldt alleen voor interval 600 sec vanuit uurdata!!!!!!!
    def interpolate_power(self, time_sim, control_interval):
        interp_func = interp1d(time_sim, self.values, fill_value='extrapolate')
        self.values = interp_func(np.arange(0, time_sim[-1] + (6 * 600),
                                            control_interval))

        # TODO: for glob and cloud the interp1d needs truncation by time_sim length.
        """Example:
        glob = df_nen['globale_zonnestraling'].values
        glob = glob.flatten()
        interp_func_glob = interp1d(time_sim, glob[0:len(time_sim)], fill_value='extrapolate')
        glob_interp = interp_func_glob(np.arange(0, time_sim[-1] + (6 * 600), control_interval))
        """


if __name__ == "__main__":
    s = SourceTerm("Test")
    days_sim = 2
    time_sim = np.arange(days_sim * 24) * 3600
    s.values = np.arange(days_sim * 24) * 0.1
    control_interval = 600
    s.interpolate_power(time_sim, control_interval)

    print()