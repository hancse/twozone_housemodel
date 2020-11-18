"""
Define input parameters and Initialization of Dwelling parameters

    10 July 2018
    Arie Taal, Baldiri Salcedo HHS
    Missing Parameters: The incident solar heat is divided between Cwall and Cair by the  
    convection factor (CF=0.8)
    Qinst : Q (instant  by heating or cooling needed) at this moments
    Last modify by Trung Nguyen
"""

import Total_Irrad as Irr 
"""
Module parameters.py first imports and executes the script in the module Total_Irrad...
"""
# read outside temperature
Toutdoor = Irr.Toutdoor
"""
...therefore, suddenly a variable Irr.Toutdoor pops up. 
"""
print("ID Toutdoor: ", id(Toutdoor), ", ID Irr.Toutdoor: ", id(Irr.Toutdoor))
"""
 the "new" variable Toutdoor is at the same memory address as 
 the "old"variable Irr.Toutdoor!
see: https://www.w3schools.com/python/ref_func_id.asp
the instruction has NOT COPIED the variable, but now it has TWO NAMES
this may cause TROUBLE! 
"""
# define window surface in m2
# Windows surface [E,SE,S,SW,W,NW,N,NE] [m2]
# -90 (E), -45 (SE), 0 (S), 45 (SW), 90 (W), 135 (NW), 180 (N), 225 (NE)

glass = [0,0, 9.5, 0 ,0, 0, 9.5, 0]

# Window solar transmitance, g-value
g_value = 0.7

# Time base on 1 hour sampling from NEN

time = Irr.qsunS[0]
# time = first row of Irr.qsunSouth (time axis) in seconds [0, 3600, 7200, ...]
print("ID time: ", id(time), ", ID Irr.qsunS[0]: ", id(Irr.qsunS[0]))
""" 
the "new" variable time is NOT at the same memory address as 
the "old" variable Irr.qsunS[0]!
because the value of the first element of an array is assigned to a scalar (float)
the instruction now has COPIED the variable
this asks for extreme programmer awareness!
"""
# Calculate Qsolar on window

Qsolar = (Irr.qsunE[1]*glass[0] + Irr.qsunSE[1]*glass[1] + 
                      Irr.qsunS[1]*glass[2] + Irr.qsunSW[1]*glass[3] + 
                      Irr.qsunW[1]*glass[4] + Irr.qsunNW[1]*glass[5] + 
                      Irr.qsunN[1]*glass[6] + Irr.qsunNE[1]*glass[7]) * g_value

# with input NEN5060, glass and g_value, qsun can give a single result Qsolar
			

#Envelope surface (facade + roof + ground) [m2]
A_facade = 160.2

#Envelope thermal resitance, R-value [m2/KW]
Rc_facade = 1.3

#Window thermal transmittance, U-value [W/m2K]
Uglass = 2.9

CF=0.8
#Ventilation, air changes per hour [#/h]
n=0.55

#Internal volume [m3]
V_dwelling = 275.6

#Facade construction
N_facade  = 2                  # Middle_weight =2 Light_weight =1 / Heavy_weight

#Floor and internal walls surface [m2]
A_internal_mass = 106+64

#Floor and internal walls construction
N_internal_mass = 2             # Middle_weight =2 / Light_weight=1 / Heavy_weight

#N_internal_mass = 2   # 1: Light weight construction / 2: Middle weight construction / 3: Heavy weight construction
#N_facade = 2          # 1: Light weight construction / 2: Middle weight construction / 3: Heavy weight construction
"""
Nine input variables are added to the "workspace"
They are hard-coded and can not be changed easily by a user interface
the variable names are hidden in the parameters.py module
"""
#_______________________________________________________

#Initial parameters file for House model

##Predefined variables
 
rho_air = 1.20             # density air in [kg/m3]
c_air  = 1005              # specific heat capacity air [J/kgK]
alpha_i_facade = 8
alpha_e_facade = 23
alpha_internal_mass = 8
"""
The predefined variables should be defined in a configuration file
They are probably equal for all house types.
"""
#Variables from Simulink model, dwelling mask (dwelling mask???????)
#Floor and internal walls construction.
#It is possible to choose between light, middle or heavy weight construction

if N_internal_mass==1:          #Light weight construction 
    
    c_internal_mass = 840         #Specific heat capacity construction [J/kgK]
    th_internal_mass = 0.1        #Construction thickness [m]
    rho_internal_mass = 500       #Density construction in [kg/m3]
      
elif N_internal_mass == 2:       #Middle weight construction
    
    c_internal_mass = 840        # Specific heat capacity construction [J/kgK]
    th_internal_mass = 0.1       # Construction thickness [m]
    rho_internal_mass = 1000     # Density construction in [kg/m3]
     
else:                          #Heavy weight construction
         
    c_internal_mass = 840        #Specific heat capacity construction [J/kgK]
    th_internal_mass = 0.2       #Construction thickness [m]
    rho_internal_mass = 2500     #Density construction in [kg/m3]

# Facade construction
# It is possible to choose between light, middle or heavy weight constructionIt is assumed that furniture and the surface part of the walls have the same temperature as the air and the wall mass is divided between the air and wall mass. Thus, the capacity of the air node consists of the air capacity, furniture capacity and capacity of a part of the walls. Appendix I presents the coefficients in the dwelling model. In the resistance Rair_outdoor the influence of heat transmission through the outdoor walls and natural ventilation is considered. It is assumed that furniture and the surface part of the walls have the same temperature as the air and the wall mass is divided between the air and wall mass. Thus, the capacity of the air node consists of the air capacity, furniture capacity and capacity of a part of the walls. Appendix I presents the coefficients in the dwelling model. In the resistance Rair_outdoor the influence of heat transmission through the outdoor walls and natural ventilation is considered. 

if N_facade == 1:              # Light weight construction
    
    c_facade = 840             # Specific heat capacity construction [J/kgK]
    rho_facade = 500           # Density construction in [kg/m3]
    th_facade = 0.1            # Construction thickness [m]

elif  N_facade == 2:           # Middle weight construction
    
    c_facade = 840             # Specific heat capacity construction [J/kgK]
    rho_facade = 1000          # Density construction in [kg/m3]
    th_facade = 0.1            # Construction thickness [m]

else:                        # Heavy weight construction
    
    c_facade = 840             # Specific heat capacity construction [J/kgK]
    rho_facade = 2500          # Density construction in [kg/m3]
    th_facade = 0.2            # Construction thickness [m]
"""
Eight extra input parameters! Hardcoded!
A configuration (*.ini) file is also a better place for these input variables
"""

# calculation part
Aglass = sum(glass)          # Sum of all glass surfaces [m2]

#Volume floor and internal walls construction [m3]

V_internal_mass = A_internal_mass * th_internal_mass

# A_internal_mass:  Floor and internal walls surface [m2]

qV = (n*V_dwelling)/3600            # Ventilation, volume air flow [m3/s],

# n: ventilation air change per hour;  V_dwelling : internal volume m3
qm = qV*rho_air                    # Ventilation, mass air flow [kg/s]

#Dwelling temperatures calculation
#Calculation of the resistances

Rair_wall = 1.0 / (A_internal_mass * alpha_internal_mass)  # Resistance indoor air-wall

U = 1.0 / (1.0 / alpha_i_facade + Rc_facade +1 / alpha_e_facade)  # U-value indoor air-facade

Rair_outdoor = 1.0 / (A_facade * U + Aglass * Uglass + qm * c_air) # Resistance indoor air-outdoor air

# Calculation of the capacities
Cair = rho_internal_mass * c_internal_mass * V_internal_mass /2.0 + rho_air * c_air * V_dwelling # Capacity indoor air + walls

Cwall = rho_internal_mass * c_internal_mass * V_internal_mass / 2.0                           # Capacity walls

"""
Thirty-three variables have been introduced in this module.
This happens out-of-sight of the user, who must edit this script in order to change parameters.
From Simulation.py, you have to look back into parameters.py to know the meaning of a parameter.
How many colleagues and research partners are willing to do this?
"""