import scipy.signal as sp
import numpy as np # linear algebra
import matplotlib.pyplot as plt
#from scipy import signal

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

    days_hours   = 24                    #number_of_hour_in_oneday + start hour at 0
    days         = 365                   #number of simulation days
    periods      = 24*3600*days          #in seconds (day_periods*365 = years)
    pulse_width  = (t2-t1)/24            # % of the periods
    phase_delay  = t1                    #in seconds

    # temperature different between day and night.
    delta_T= Day_T_SP - Night_T_SP
    duty_b   = 23-18

    # create simulation time
    time_t = np.linspace(0,periods,(days_hours*days)+1)

    t = np.linspace(0,1,(days_hours*days)+1,endpoint=False)          #+1 start from 0 days=1
    temp1 = sp.square(2 * np.pi* days * t,duty=duty_wu/24)
    temp1 = np.clip(temp1, 0, 1)

    # add delay to array
    temp1 = np.roll(temp1,Wu_time)

    t = np.linspace(0,1,(days_hours*days)+1,endpoint=False)          #+1 start from 0 days=1
    temp2 = sp.square(2 * np.pi* days * t,duty=duty_w/24)
    temp2 = np.clip(temp2, 0, 1)

    # add delay to array
    temp2 = np.roll(temp2,Work_time)

    t = np.linspace(0,1,(days_hours*days)+1,endpoint=False)          #+1 start from 0 days=1
    temp3 = sp.square(2 * np.pi* days * t,duty=duty_b/24)
    temp3 = np.clip(temp3, 0, 1)

    # add delay to array
    temp3 = np.roll(temp3,back_home)

    # Calculate SP
    temp4 = temp1-temp2+temp3
    SP = (temp4*delta_T)+Night_T_SP

    SP = SP[np.newaxis]
    SP = SP.T

    # Plot 48 hours
    # plt.plot(time_t[0:48], SP[0:48])
    plt.plot(SP[0:48])
    # plt.plot(time_t[0:48])

    # plt.ylabel('Temperature_SP (degC)')
    # plt.xlabel('time (sec)')
    # plt.legend(loc=2)
    # print(Qinternal)
    SP = np.delete(SP, -1, 0)

    return SP


if __name__ == "__main__":
    setpoint = temp_sp(8, 23, 17, 20, 7, 16, 8, 15, 18)

    # Plot 48 hours
    plt.figure(figsize=(5, 5))
    plt.plot(setpoint[0:48], label='setpoint')
    plt.ylabel('Temperature_SP (degC)')
    plt.xlabel('time (h)')
    plt.legend(loc='best')
    plt.show()