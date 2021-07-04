# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:57:55 2020

@author: TrungNguyen
"""
import datetime
import holidays

import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


def temp_sp(t1, t2, Night_T_SP, Day_T_SP, Wu_time, duty_wu,
            Work_time, duty_w, back_home):
    """ Define Temperature SP for 1 days (24 hours).

    Args:
        t1:           Presence from [hour]
        t2:           Presence until [hour]
        Night_T_SP:   Set temperature of thermostat at night from time t2
        Day_T_SP:     Set wishes temperature of thermostat
        Wu_time:      Define wake up time in the morning, temperature set to 20
        duty_wu:
        Work_time:    Define time that people go to work.
        duty_w:
        back_home:    Define time that people back from work 18:00

    Returns:
        SP:           Temperature SP profile
    """

    # t1= 8                                #Presence from [hour]
    # t2= 23                               #Presence until [hour]

    # Night_T_SP=17                        # Set temperature of thermostat at night from time t2
    # Day_T_SP=20							 # Set wishes temperature of thermostat

    # Define Wake up time
    # Wu_time =7           				 # Define wake up time in the morning, temperature set to 20
    # duty_wu = 23-7

    # Go to work time/ leave the house
    # Work_time = 8           			 # Define time that people go to work.
    # duty_w   = 23-8

    # Back to home
    # back_home = 18                       #Define time that people back from work 18:00

    # Creating profile

    days_hours = 24  # number_of_hour_in_oneday + start hour at 0
    days = 365  # number of simulation days
    periods = 24 * 3600 * days  # in seconds (day_periods*365 = years)
    pulse_width = (t2 - t1) / 24  # % of the periods
    phase_delay = t1  # in seconds

    # temperature different between day and night.
    delta_T = Day_T_SP - Night_T_SP
    duty_b = 23 - 18

    # create simulation time
    time_t = np.linspace(0, periods, (days_hours * days) + 1)

    t = np.linspace(0, 1, (days_hours * days) + 1, endpoint=False)  # +1 start from 0 days=1
    temp1 = signal.square(2 * np.pi * days * t, duty=duty_wu / 24)
    temp1 = np.clip(temp1, 0, 1)

    # add delay to array
    temp1 = np.roll(temp1, Wu_time)

    t = np.linspace(0, 1, (days_hours * days) + 1, endpoint=False)  # +1 start from 0 days=1
    temp2 = signal.square(2 * np.pi * days * t, duty=duty_w / 24)
    temp2 = np.clip(temp2, 0, 1)

    # add delay to array
    temp2 = np.roll(temp2, Work_time)

    t = np.linspace(0, 1, (days_hours * days) + 1, endpoint=False)  # +1 start from 0 days=1
    temp3 = signal.square(2 * np.pi * days * t, duty=duty_b / 24)
    temp3 = np.clip(temp3, 0, 1)

    # add delay to array
    temp3 = np.roll(temp3, back_home)

    # Calculate SP
    temp4 = temp1 - temp2 + temp3
    SP = (temp4 * delta_T) + Night_T_SP

    SP = SP[np.newaxis]
    SP = SP.T

    # Plot 48 hours
    # plt.plot(time_t[0:48], SP[0:48])
    # plt.plot(SP[0:48])
    # plt.plot(time_t[0:48])

    # plt.ylabel('Temperature_SP (degC)')
    # plt.xlabel('time (sec)')
    # plt.legend(loc=2)
    # print(Qinternal)
    SP = np.delete(SP, -1, 0)

    return SP


def thermostat_sp(t1, t2, Night_T_SP, Day_T_SP, Flex_T_SP, Wu_time, not_at_home, back_home):
    """ Define Temperature SP for 1 days (24 hours).

    Args:
        t1:             Presence from [hour]
        t2:             Presence until [hour]
        Night_T_SP :    Set temperature of thermostat at night from time t2
        Day_T_SP   :    Set wishes temperature of thermostat
        Flex_T_SP     :    Flexible temperature setting for the time that people are not at home
                        or go to work
                        
        Wu_time    :    Define wake up time in the morning, temperature set to 20
        not_at_home:    Define time that people go to work or shopping.
        back_home:      Define time that people back from work 18:00

    Returns:
        SP:           Temperature SP profile
    """
    # t1= 8                                #Presence from [hour]
    # t2= 23                               #Presence until [hour]

    # Night_T_SP=17                        # Set temperature of thermostat at night from time t2
    # Day_T_SP=20							 # Set wishes temperature of thermostat

    # Define Wake up time
    # Wu_time =7           				 # Define wake up time in the morning, temperature set to 20
    duty_wu = t2 - Wu_time

    # Go to work time/ leave the house
    # Work_time = 8           			 # Define time that people go to work.
    duty_w = t2 - not_at_home

    # Back to home
    # back_home = 18                       #Define time that people back from work 18:00
    duty_b = t2 - back_home

    # Creating profile

    days_hours = 24  # number_of_hour_in_oneday + start hour at 0
    days = 365  # number of simulation days
    periods = 24 * 3600 * days  # in seconds (day_periods*365 = years)
    pulse_width = (t2 - t1) / 24  # % of the periods
    phase_delay = t1  # in seconds

    # temperature different between day and night.
    delta_T = Day_T_SP - Night_T_SP

    # temperature different between day and night.
    # if (Day_T_SP - Flex_T_SP != 0):
    delta_T_flex = Day_T_SP - Flex_T_SP
    # else:
    #    delta_T_flex=delta_T

    # -----------------------
    t = np.linspace(0, 1, (days_hours * days) + 1, endpoint=False)  # +1 start from 0 days=1
    temp1 = signal.square(2 * np.pi * days * t, duty=duty_wu / 24)
    temp1 = np.clip(temp1, 0, 1)
    # add delay to array
    temp1 = np.roll(temp1, Wu_time)

    # ----------------
    t = np.linspace(0, 1, (days_hours * days) + 1, endpoint=False)  # +1 start from 0 days=1
    temp2 = signal.square(2 * np.pi * days * t, duty=duty_w / 24)
    temp2 = np.clip(temp2, 0, 1)
    # add delay to array
    temp2 = np.roll(temp2, not_at_home)

    # ___________
    t = np.linspace(0, 1, (days_hours * days) + 1, endpoint=False)  # +1 start from 0 days=1
    temp3 = signal.square(2 * np.pi * days * t, duty=duty_b / 24)
    temp3 = np.clip(temp3, 0, 1)
    # add delay to array
    temp3 = np.roll(temp3, back_home)

    # ----------------
    t = np.linspace(0, 1, (days_hours * days) + 1, endpoint=False)  # +1 start from 0 days=1
    temp_flex = signal.square(2 * np.pi * days * t, duty=duty_b / 24)
    temp_flex = np.clip(temp_flex, 0, 1)
    # add delay to array
    temp_flex = np.roll(temp_flex, back_home)

    # Calculate SP
    temp1 = (temp1 * delta_T) + Night_T_SP
    temp2 = (temp2 * delta_T_flex) + Night_T_SP
    temp3 = (temp3 * delta_T) + Night_T_SP
    temp_flex = temp_flex * (delta_T - delta_T_flex)
    # plt.plot(temp1[0:24])
    # plt.plot(temp2[0:24])
    # plt.plot(temp_flex[0:24])

    temp4 = temp1 - temp2 - temp_flex + temp3
    # plt.plot(temp4[0:24])
    SP = temp4
    # SP_weekday=(temp4*delta_T)+Night_T_SP
    # plt.plot(SP_weekday[0:24])
    SP = SP[np.newaxis]
    SP = SP.T
    SP = np.delete(SP, -1, 0)

    # SP_weekday=SP_weekday.flatten()

    return SP


def SP_profile(SP_weekday, SP_dayoff):
    """ Define Temperature SP for 1 days (24 hours).

    Args:
        SP_weekday:          week day temperature profile
        SP_dayoff:           holiday or weekend temperature profile

    Returns:
        SP:                 Temperature SP profile for 8760 hours(year)
    """
    base = datetime.datetime(2020, 1, 1)
    date = np.array([base + datetime.timedelta(days=i) for i in range(365)])
    nl_holidays = holidays.Netherlands(years=2020)

    temp2 = np.zeros((0, 1))

    # A years profile loop
    for i in range(len(date)):

        temp = SP_weekday[0:24]

        if date[i].strftime("%A") == 'Saturday' or date[i].strftime("%A") == 'Sunday':
            # print(i)
            temp = SP_dayoff[0:24]
            # print('------------------weekend------------------')

        if nl_holidays.get(date[i]) != None:
            # print(i)
            temp = SP_dayoff[0:24]

        # if i == 359:
        # plt.plot(SP_weekday[0:23])
        #   plt.plot(temp[7:24])

        temp2 = np.concatenate((temp2, temp))

    SP = temp2[0:8760].flatten()

    return SP


def simple_thermostat(t_on=7, t_off=22, SP_day=20.0, SP_night=17.0,
                      begin_summer=3500, end_summer=6500,
                      to_work=8, from_work=18, SP_absent=17.0):
    """simple home thermostat setting routine with day/night setting and summer period

       optional: setpoint reduction during absence for work

    Args:
        t_on:            start time of high (day) thermostat setting
        t_off:           end time of high (day) thermostat setting
        SP_day:          set point thermostat day
        SP_night:        set point thermostat night
        begin_summer:    begin seasonal no-heating (summer) period
        end_summer:      end seasonal no-heating (summer) period
        to_work:         optional start time of daily absence
        from_work:       optional start time of daily return home
        SP_absent:       set point thermostat during daily absence

    Returns:
        sp: array with setpoint temperatures
    """
    hours = np.linspace(0, 8760, 8760, endpoint=False)
    sp = np.where( ((hours % 24) >= t_off) | ((hours % 24) < t_on), SP_night, SP_day )
    # sp = np.where(((hours % 24) >= to_work) & ((hours % 24) < from_work), SP_absent, sp)
    sp = np.where((hours >= begin_summer) & (hours < end_summer), 5, sp)
    return sp

if __name__ == "__main__":
    SP = simple_thermostat(t_on=6, t_off=22,
                      SP_day=20.0, SP_night=17.0,
                      begin_summer=3500, end_summer=6500)


    # Plot 120 hours
    plt.figure(figsize=(5, 5))
    plt.plot(SP[0:240], label='setpoint')
    plt.ylabel('Temperature_SP ($ \degree $ C)')
    plt.xlabel('time (h)')
    plt.legend(loc='best')
    plt.title("Simple thermostat")
    plt.tight_layout()
    plt.show()

    setpoint = temp_sp(8, 23, 17, 20, 7, 16, 8, 15, 18)

    # Plot 48 hours
    plt.figure(figsize=(5, 5))
    plt.plot(setpoint[0:48], label='setpoint')
    plt.ylabel('Temperature_SP (degC)')
    plt.xlabel('time (h)')
    plt.legend(loc='best')
    plt.show()

    week_day_setpoint = thermostat_sp(8, 23, 17, 20, 17, 7, 8, 18)
    day_off_setpoint = thermostat_sp(8, 23, 17, 20, 18, 10, 13, 15)
    # plt.plot(day_off_setpoint[0:24], label='setpoint')
    # print(week_data.iloc[0]['Datum'] == week_data.iloc[0]['holidays'])
    SP = SP_profile(week_day_setpoint, day_off_setpoint)

    # Plot 48 hours
    plt.figure(figsize=(5, 5))
    # plt.plot(week_day_setpoint[0:48], label='setpoint')
    # plt.plot(day_off_setpoint[0:48], label='setpoint')
    # plt.plot(SP[0:120], label='setpoint')
    plt.plot(SP[24 * 0:24 * 5], label='setpoint')
    # plt.ylabel('Temperature_SP (degC)')
    # plt.xlabel('time (h)')
    # plt.legend(loc='best')
    # plt.show()
