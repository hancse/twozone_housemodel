import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

def outdoor_reset(T_ambient, slope, intercept=20):
    """Compute outdoor reset temperature.

    Args:
        T_ambient:    (float):  air temperature outside the house in degree C.
        slope:        (float):  slope of the curve the house in degree C
        intercept:    (float):  intercept of the curve

    Returns:
        feed temperature :  (float) containing the outdoor reset temperature

    """
    #https://www.familie-kleinman.nl/energie/elga-ace-stooklijnen/
    feed_temperature = intercept + slope*(intercept - T_ambient)
    return feed_temperature


class hyst():
    """class for general hysteresis behaviour
       with controlled output in and below dead band
    """

    def __init__(self, dead_band: float, state=True):
        """initialize hysteresis object with dead band and in active state

        Args:
            dead_band : dead band
            state:      True/False
        """
        self.dead_band = dead_band
        self.state = state

    def update(self, pv: float, sp: float, con_var: float):
        # If temperature is below dead band, enable heat pump
        if pv < (sp - self.dead_band):
            self.state = True
            return con_var

        # If the temperature has decreased from above into the dead band, zero power
        elif ((sp - self.dead_band) <= pv <= (sp + self.dead_band)):
            if self.state is False:
                return 0.0
            elif self.state is True:
                return con_var

        # If temperature has increased above dead band, disable heat pump
        elif pv > (sp + self.dead_band):
            self.state = False
            return 0.0

        """
        # hysteresis_state = True
        # hysteresis_band = 0.5
        ...
        ...
        # If temperature is below dead band, enable heat pump
        if Tair[i] < (SP_T[i] - hysteresis_band):
            water_temp[i] = outdoor_reset(T_outdoor_sim[i], 0.7, 20)
            cop_hp[i], p_hp = nta.update(T_outdoor_sim[i], water_temp[i])
            hysteresis_state = True

        # If the temperature has decreased from above into the dead band, zero power
        elif ((SP_T[i] - hysteresis_band) <= Tair[i] <= (SP_T[i] + hysteresis_band)):
            if hysteresis_state is False:
                p_hp = 0
            elif hysteresis_state is True:
                water_temp[i] = outdoor_reset(T_outdoor_sim[i], 0.7, 20)
                cop_hp[i], p_hp = nta.update(T_outdoor_sim[i], water_temp[i])

        #If temperature has increase above deadband, disable heat pump
        elif Tair[i] > (SP_T[i] + hysteresis_band):
            p_hp = 0
            hysteresis_state = False
        """

if __name__ == "__main__":
    T = np.linspace(-20, 20, 41, endpoint=True)
    water_temp = np.zeros(41)
    for i in range(41):
         water_temp[i] = outdoor_reset(T[i], 0.8, 20)

    plt.figure(figsize=(15, 5))  # key-value pair: no spaces
    plt.plot(T, water_temp, label='Feed temperature')
    plt.legend(loc='best')
    plt.title("Feed temperature")
    plt.show()