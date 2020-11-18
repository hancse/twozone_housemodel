# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:05:19 2020

@author: TrungNguyen
"""
# Defining main function
from house import house  # exposed function "house" in house module
# function "model" in module house is private
# import house as hs

import parameters as par

# import internal_heat_gain as hg
from internal_heat_gain import internal_heat_gain

# import Temperature_SP as sp
from Temperature_SP import temp_sp

# import Total_Irrad as irrad  # unused module
"""
The modules above are scripts. On import, these scripts are executed.
Although this serial program structure works, it does not offer the benefits of using functions.
"""
import matplotlib.pyplot as plt


def main():
    # Define Simulation time
    days_Sim = 20  # number of simulation days
    time_sim = par.time[0:days_Sim * 24]
    Qsolar_Sim = par.Qsolar[0:days_Sim * 24]
    # Qsolar_Sim = Qsolar[0:days_Sim*24]*0

    # Qinternal_Sim = hg.Qinternal[0:days_Sim * 24]
    Qint = internal_heat_gain(400, 150, 8, 23)
    Qinternal_Sim = Qint[0:days_Sim*24]

    # Qinst_Sim = Qinst_Sim[0:days_Sim*24][:,0]
    T_outdoor_Sim = par.Toutdoor[0:days_Sim * 24]

    # Set point
    # SP_Sim = sp.SP[0:days_Sim * 24]
    SP = temp_sp(8, 23, 17, 20, 7, 16, 8, 15, 18)
    SP_Sim = SP[0:days_Sim * 24]

    CF = par.CF
    Rair_outdoor = par.Rair_outdoor
    Rair_wall = par.Rair_wall
    Cair = par.Cair
    Cwall = par.Cwall

    data = house(T_outdoor_Sim, Qinternal_Sim, Qsolar_Sim, SP_Sim, time_sim, CF, Rair_outdoor, Rair_wall, Cair, Cwall)

    # ______plot the results________________
    plt.figure(figsize=(20, 5))  # key-value pair: no spaces
    plt.plot(data[0], label='Tair')
    plt.plot(data[1], label='Twall')
    plt.plot(SP_Sim, label='SP_Temperature')
    # plt.plot(T_outdoor_Sim,label='Toutdoor')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    main()  # temporary solution, recommended syntax
