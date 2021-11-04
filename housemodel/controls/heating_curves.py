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