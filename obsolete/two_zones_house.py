"""
house model base on 7R4C network
"""

from scipy.integrate import odeint       # ODE solver
import numpy as np                       # linear algebra


# Define model
def house_m_zone(x,t,T_outdoor,Q_internal,Q_solar,SP_T,
                 Qinst,Qinst_1,CF,Rair_outdoor,Rair_wall,
                 Cair,Cwall,Rair_z12,Rair_z21,Rair_cc,Cwall_cc,Tc):
    
    """model function for scipy.integrate.odeint.

    :param x: variable array dependent on time
    :param t:
    :param T_outdoor:
    :param Q_internal:
    :param Q_solar:
    :param SP_T:
    :param Qinst:
    :param CF:
    :param Rair_outdoor:
    :param Rair_wall:
    :param Cair:
    :param Cwall:
    :param Rair_z12:
    :param Rair_z21:
    :param Rair_cc:
    :param Cwall_cc:

    :return:

    x,t: ode input function func : callable(x, t, ...) or callable(t, x, ...)
    Computes the derivative of y at t.
    If the signature is ``callable(t, y, ...)``, then the argument tfirst` must be set ``True``.
    """

    # States :

    Tair_z1            = x[0]   # air temperature zone 1
    Twall              = x[1]   # wall temperature 
    Tair_z2            = x[2]   # air temperature zone 2
    Twall_cc           = x[3]   # temperature of concrete wall between zone_1 and zone_2
    
    # 
    Rair_outdoor_z1 = Rair_outdoor
    Rair_outdoor_z2 = Rair_outdoor
    Rair_wall_z2    = Rair_wall
    Rair_wall_z1    = Rair_wall
    
    #
    
    Cair_z1   = Cair
    Cair_z2   = Cair
    Cwall_z1  = Cwall
    Cwall_z2  = Cwall
    
    #
    
    Q_solar_z1 = Q_solar
    Q_solar_z2 = Q_solar

    #     err      = SP_T-Tair
    #     integaldt= err
    #     integald= np.clip(integaldt, 0, 5)
    #     Qinst    = kP*(err) + ki*integal # kd*Tair.dt() # PID form
    #     Qinst=np.clip(Qinst, 0, 7000)
    #     #m.Equation(Integl.dt()==err )

    # ____Air temperature zone 1 equation_______
    
    Tair_z1dt   = ((T_outdoor-Tair_z1)/Rair_outdoor_z1 + (Twall-Tair_z1)/Rair_wall_z1
                   + (Tair_z2-Tair_z1)/Rair_z21 + Qinst + Q_internal + CF*Q_solar_z1)/Cair_z1
    
    #_____Wall temperature_______
        
    Twalldt  = ((Tair_z1-Twall)/Rair_wall_z1 + (Tair_z2-Twall)/Rair_wall_z2 + (1-CF)*Q_solar_z1)/Cwall_z1
    
    #________Air temperature zone 2 equation
    
    Tair_z2dt   = ((Twall-Tair_z2)/Rair_wall_z2 + ((T_outdoor-Tair_z2)/Rair_outdoor_z2 + (Tair_z1-Tair_z2)/Rair_z12 - Tc) 
                   + (Twall_cc-Tair_z2)/Rair_cc + 0  + Qinst_1 + Q_internal + CF*Q_solar_z2)/Cair_z2
    
    #________temperature of the concrete between Zone 1 and Zone 2___
    
    Twall_ccdt   = ((Tair_z2-Twall_cc)/Rair_cc + (1-CF)*Q_solar_z2)/Cwall_cc
   
    return [Tair_z1dt,Twalldt,Tair_z2dt,Twall_ccdt]    


# Initial Conditions for the States
def house(T_outdoor,Q_internal,Q_solar,SP_T,time_sim,
          CF,Rair_outdoor,Rair_wall,Cair,
          Cwall,Rair_z12,Rair_z21,Rair_cc,Cwall_cc,controller_value,z1_on_off,z2_on_off):
    
    """Compute air and wall tempearature inside the house.

    :param T_outdoor:    (array):  Outdoor temperature in degree C
    :param Q_internal:   (array):  Internal heat gain in w.
    :param Q_solar:      (array):  Solar irradiation on window [W]
    :param SP_T:         (array):  Setpoint tempearature from thermostat.
    :param time_sim:     (array)  :  simulation time

    :param CF:
    :param Rair_outdoor:
    :param Rair_wall:
    :param Cair:
    :param Cwall:
    :param Rair_z12:
    :param Rair_z21:
    :param Rair_cc:
    :param Cwall_cc:
    :param Tc:                      Correction values for the heat flow from top to bottom 
                                    in degree C
    :return:             tuple :  Tuple containing (Tair, Twall, conrumption):

                - Tair (float):   air temperature inside the house in degree C.
                - Twall (float):  wall temperature inside the house in degree C
                - consumption :   power consumption in kwh 

    Qinst ?	  (array):  instant heat from heat source such as HP or boiler [W].

    """

    #Solar Collector
    Tair_z10    = 20   
    Twall_z10   = 20
    Tair_z20    = 20
    Twall_cc0   = 20
    #Twall_z20   = 15
    
    y0 = [Tair_z10,Twall_z10,Tair_z20,Twall_cc0]
    
    # Time Interval (sec)
    #t = np.linspace(0,60*60,days_Sim*24)  # Define Simulation time with resolution
    #time_sim      = time[0:days_Sim*24]
    t = time_sim           # Define Simulation time with sampling time
    
    Tair_z1   = np.ones(len(t))*Tair_z10
    Twall_z1  = np.ones(len(t))*Twall_z10
    Tair_z2   = np.ones(len(t))*Tair_z20
    Twall_cc   = np.ones(len(t))*Twall_cc0
    
    #Twall_z2  = np.ones(len(t))*Twall_z20
    #SP_T= np.ones(len(t))*20
    Consumption_z1 = np.ones(len(t))
    Consumption_z2 = np.ones(len(t))

    kp = controller_value
    for i in range(len(t)-1):
        
        err=SP_T[i+1] - Tair_z1[i]
        err_1=SP_T[i+1] - Tair_z2[i]
        Qinst=err*kp*z1_on_off
        Qinst=np.clip(Qinst,0,4000)
        Qinst_1=err_1*kp*z2_on_off
        Qinst_1=np.clip(Qinst_1,0,4000)
        
        # add correction values for heat flow from top to bottom
        if (Tair_z1[i] - Tair_z2[i])>=0:
            Tc=0
        else:
            Tc= (Tair_z1[i] - Tair_z2[i])/Rair_z12
        
        if (T_outdoor[i]>= 15):
            Qinst=0
            Qinst_1=0
        else:
            Qinst=Qinst
            Qinst_1=Qinst_1
        
        inputs = (T_outdoor[i],Q_internal[i],Q_solar[i],SP_T[i],
                  Qinst,Qinst_1,CF,Rair_outdoor,Rair_wall,Cair,
                  Cwall,Rair_z12,Rair_z21,Rair_cc,Cwall_cc,Tc)
    
        ts = [t[i],t[i+1]]
        y = odeint(house_m_zone,y0,ts,args=inputs)
        # Store results
    
    #delatat is a delay base on distance between buffer tank and Solar Collector    
        #deltat    = rp**(2)*pi*lp/(FP[i+1])
        Tair_z1 [i+1]         = y[-1][0]
        Twall_z1[i+1]         = y[-1][1]
        Tair_z2 [i+1]         = y[-1][2]
        Twall_cc[i+1]         = y[-1][3]
       
        # Adjust initial condition for next loop
        y0 = y[-1] # current value of Tair and Twall
        Consumption_z1[i] = Qinst
        Consumption_z2[i] = Qinst_1
        
    return Tair_z1, Twall_z1,Tair_z2, Twall_cc, Consumption_z1, Consumption_z2
