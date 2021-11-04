import numpy as np

# Q.2.1. Energy fraction
# Q_hp_MJ per year, Q_other_MJ per month

F = Q_hp_MJ/ / np.sum(Q_other_MJ)

# in practice, start with F = 1.0

# Q.2.2
# [kWh] --> [MJ]: multiply with 3.6 (3600/1000)

eta = Q_hp_MJ / Q_el_MJ

# values per year: is this the same as SCOP or SPF?

# Q.2.3

Q_hp_MJ = 3.6 * np.sum(DeltaT * P_hp)

# Q_hp_MJ per year
# DeltaT: time interval: e.g. one hour
# P_hp dependent on outside temperature, sum over outside temperatures

# Q.2.4

Q_hp_el_MJ = 3.6 * np.sum(DeltaT * (P_h / COP)) + W_aux_el

# Q.2.5

W_aux_el = np.sum(DeltaT * P_aux_el) / 1.0e6
# sum over P_aux [W], conversion [J] --> [MJ]

# Q.2.6 alleen modulerend

# Q.2.7

# Q.2.8

# this section proposes a first-order, bilinear dependence of
# heat pump COP upon T_evap (outside and T_cond (inside)
# the norm uses theta for temperature

# correctiefactor



