from scipy import signal
import numpy as np # linear algebra
import matplotlib.pyplot as plt

def internal_heat_gain(Qday, DeltaQ, t1, t2):
    """

    Args:
        Qday:      Daytime internal heat generation [W]
        DeltaQ:    Internal heat generation difference between day and night [W]
        t1:        Presence from [hour]
        t2:        Presence until [hour]

    Returns:
        Q_internal: (array): internal heat generation profile for one year

    """

    # DeltaQ     = 150                     #Internal heat gain difference between day and night
    # day_DeltaQ = DeltaQ                 #Day Delta Q internal [W]
    # Qday       = 400                     #Day internal heat gain W
    nightQ     = Qday - DeltaQ           #Night internal heat gain

    # t1= 8                                #Presence from [hour]
    # t2= 23                               #Presence until [hour]

    days_hours   = 24                    #number_of_hour_in_oneday + start hour at 0
    days         = 365                   #number of simulation days
    periods      = 24*3600*days          #in seconds (day_periods*365 = years)
    pulse_width  = (t2-t1)/24            # % of the periods
    phase_delay  = t1                    #in seconds


    #t = np.linspace(0, 24*3600, 24)
    t= np.linspace(0,1,(days_hours*days)+1,endpoint=False)          #+1 start from 0
    pulseday = signal.square(2 * np.pi* days * t,duty=pulse_width)
    pulseday = np.clip(pulseday, 0, 1)
    # add delay to array
    pulseday=np.roll(pulseday,phase_delay)

    # pulse week generator

    week = days/7
    pulse_w  = 0.99

    pulse_week = signal.square(2*np.pi*week*t,duty=pulse_w)
    pulse_week = np.clip(pulse_week, 0, 1)

    #create simulation time
    time_t = np.linspace(0,periods,(days_hours*days)+1)

    # Internal heat gain

    Qinternal = nightQ + pulseday*DeltaQ*pulse_week
    Qinternal = Qinternal[np.newaxis]
    Qinternal = Qinternal.T

    # Plot 48 hours
    # plt.plot(time_t[0:48], Qinternal[0:48])
    # plt.ylabel('Internal heat gain (W)')
    # plt.xlabel('time (sec)')
    # plt.legend(loc=2)

    # print(Qinternal)
    Qinternal = np.delete(Qinternal, -1, 0)

    return Qinternal

if __name__ == "__main__":
    Q = internal_heat_gain(400, 150, 8, 23)

    # Plot 48 hours
    plt.figure(figsize=(5, 5))
    plt.plot(Q[0:480], label='internal heat')
    plt.ylabel('Internal heat gain (W)')
    plt.xlabel('time (h)')
    plt.legend(loc='best')
    plt.show()