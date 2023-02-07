# See: https://buildingenergygeeks.org/a-simple-rc-model-python.html

import pandas as pd
import numpy as np

import scipy.optimize as so

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")


# Function that will calculate the indoor temperature for a given value of R and C
def simpleRC(inputs, Ri, Ro, Ci, Ce, Ai, Ae):
    time = inputs[:, 0]
    to = inputs[:, 1]
    phi_h = inputs[:, 2]
    phi_s = inputs[:, 3]

    ti = np.zeros(len(time))
    te = np.zeros(len(time))

    # Initial temperatures
    ti[0] = output[0]
    te[0] = (Ri * to[0] + Ro * ti[0]) / (Ri + Ro)

    # Loop for calculating all temperatures
    for t in range(1, len(output)):
        dt = time[t] - time[t - 1]
        ti[t] = ti[t - 1] + dt / Ci * ((te[t - 1] - ti[t - 1]) / Ri + phi_h[t - 1] + Ai * phi_s[t - 1])
        te[t] = te[t - 1] + dt / Ce * ((ti[t - 1] - te[t - 1]) / Ri + (to[t - 1] - te[t - 1]) / Ro + Ae * phi_s[t - 1])

    return ti


df = pd.read_csv('data/statespace.csv')

fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex='all')
ax[0].plot(df['Time'].values/3600, df['P_hea'].values, label="Heating power ($W$)")
ax[0].plot(df['Time'].values/3600, df['I_sol'].values, label="Solar irr. ($W/m^2$)")
ax[1].plot(df['Time'].values/3600, df['T_int'].values, label="Indoor temp. ($C$)")
ax[1].plot(df['Time'].values/3600, df['T_ext'].values, label="Outdoor temp. ($C$)")

[a.legend() for a in ax]
ax[1].set_xlabel("time (hour)")
plt.show()

inputs = df[['Time', 'T_ext', 'P_hea', 'I_sol']].values
output = df['T_int'].values

p_opt, p_cov = so.curve_fit(f=simpleRC, xdata=inputs, ydata=output,
                            p0=(0.01, 0.01, 1e6, 1e7, 5, 5))

# Saving results into a dataframe and displaying it
res1 = pd.DataFrame(index=['Ri', 'Ro', 'Ci', 'Ce', 'Ai', 'Ae'])
res1['avg'] = p_opt
res1['std'] = np.diag(p_cov)**0.5
print(res1)

curve = simpleRC(inputs, p_opt[0], p_opt[1], p_opt[2], p_opt[3], p_opt[4], p_opt[5])
fig2, ax = plt.subplots(1, 1, figsize=(8, 6), sharex='all')
ax.plot(df['Time'].values/3600, df['T_int'].values, label="Indoor temp. ($C$)")
ax.plot(df['Time'].values / 3600, curve, label="Outdoor temp. ($C$)")

ax.legend()
ax.set_xlabel("time (hour)")
plt.show()

