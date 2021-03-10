"""
A certain Python style gives the modules (*.py files) names of a profession:
in this style, the module that encapsulates the parameter configuration can be called
configurator.py
the module performs the following tasks:
1. read the input parameters for the model simulation from a configuration file
   "Pythonic" configuration file types are *.ini, *.yml, *.toml and *.json
   The *.yml can cope with array parameters. This makes it more useful than the *.ini format
   The *.json format can also represent arrays. It is used when the input data comes from a database.
2. convert the input parameters to a dict
3. optionally, convert the dict to a dataclass object
4. get additional parameters from NEN5060
5. perform calculations to prepare for ODE integration
"""

import yaml

"""
The predefined variables are now defined in a configuration file
All parameters read from configuration (*.yml) file 
"""

def load_config(config_name: str):
    with open(config_name) as config_file:
        hp = yaml.safe_load(config_file)  # hp = house_parameters
        return hp


def save_config(hp):
    with open("../config2R2C.yml", "w") as config_outfile:
        yaml.dump(hp, config_outfile, indent=4)


# Variables from Simulink model, dwelling mask (dwelling mask???????)
# Floor and internal walls construction.
# It is possible to choose between light, middle or heavy weight construction

"""
# Facade construction
# It is possible to choose between light, middle or heavy weight construction

the parameters c_internal_mass, th_internal_mass and rho_internal_mass
               c_facade, th_facade and rho_facade are now lists
               the indices to these lists are N_internal_mass an N_facade
"""

# It is assumed that furniture and the surface part of the walls have the same temperature
# as the air and the wall mass is divided between the air and wall mass.
# Thus, the Heat capacity of the air node consists of the air heat capacity,
# furniture Heat capacity and Heat capacity of a part of the walls.
# Appendix I presents the coefficients in the dwelling model.
# In the resistance Rair_outdoor the influence of heat transmission through the outdoor walls
# and natural ventilation is considered.


def calculateRC(hp: dict):
    """

    Args:
        hp:

    Returns:
        Rair_wall :
        Cwall :
        Rair_outdoor :
        Cair :
    """
    # assignment to local variables from hp: dict 
	
    # Dwelling temperatures calculation
	# file://../house.py 	
	
	#_______________________Dwelling temperatures calculation________________________

	#   Tairdt = ((T_outdoor - Tair) / Rair_outdoor + (Twall - Tair) / Rair_wall 
	#             + Qdotinst + Qdot_internal + CF * Qdot_solar) / Cair

	#   Twalldt = ((Tair - Twall) / Rair_wall + (1 - CF) * Qdot_solar) / Cwall

	#Calculation of the resistances.

	#Resistance indoor air-wall:            Rair_wall = 1.0 / (A_internal_mass * alpha_internal_mass) 

	#Resistance indoor air-outdoor air:     Rair_outdoor = 1.0 / (A_facade * U + A_glass * U_glass + qm * c_air)
	#      
	#       Sum of all glass surfaces:      A_glass  = g_value.sum(glass)
	#          
	#       U-value indoor air-facade:      U = 1.0 / (1.0 / alpha_i_facade + Rc_facade + 1 / alpha_e_facade)  
	#      
	#       Ventilation, mass air flow:     qm = qV * rho_air
	#           
	#       Ventilation, volume air flow:   qV = (n * V_dwelling) / 3600
    
	# Initial parameters file for House model
    # Predefined variables
   
   
	#CF(solar radiation):  the convection factor is the part 
	#of the solar radiation that entersthe room and is released directly convectively into the room.
    CF = hp['ventilation']['CF'] 
	# density air in [kg/m3]
	rho_air = hp['initial']['rho_air']
	# specific heat capacity air [J/kgK]
    c_air = hp['initial']['c_air']  
	# Internal volume [m3]
    V_dwelling = hp['dimensions']['V_dwelling']
	
	

	#__________Resistance indoor air-wall____________
	
	# Floor and internal walls surface [m2]
    A_internal_mass = hp['dimensions']['A_internal_mass']
	# Internal wall thermal resistance
	alpha_internal_mass = hp['initial']['alpha_internal_mass']
	#Resistance indoor air-wall:
    Rair_wall = 1.0 / (A_internal_mass * alpha_internal_mass)  # Resistance indoor air-wall
	
	#_________Resistance indoor air-outdoor air__________

	# Envelope thermal resistance, R-value [m2/KW]
    Rc_facade = hp['thermal']['Rc_facade']
	# Heat transfer coefficient [W/m2K]. 
    # Interior surface thermal resistance Ri= 1/ alpha_i_facade
	alpha_i_facade = hp['initial']['alpha_i_facade']
	# Heat transfer coefficient [W/m2K]. 
    # Exterior surface thermal resistance Rse = 1/ alpha_e_facade
    alpha_e_facade = hp['initial']['alpha_e_facade']
    
	
	 #___U-value indoor air-facade___
    U = 1.0 / (1.0 / alpha_i_facade + Rc_facade + 1 / alpha_e_facade)    
	
	#Envelope surface (facade + roof + ground) [m2]
    A_facade = hp['dimensions']['A_facade']
	#______Sum of all glass surfaces [m2]______
	A_glass = sum(hp['glass'].values())  
    A_glass -= hp['glass']['g_value']
    #print(A_glass)
    #Window thermal transmittance, U-value [W/m2K]
    Uglass = hp['thermal']['U_glass']
	# Ventilation, air changes per hour [#/h]
    n = hp['ventilation']['n']
	#Ventilation, volume air flow [m3/s]
	qV = (n * V_dwelling) / 3600  
	#_____Ventilation, mass air flow [kg/s]______
    qm = qV * rho_air  
	# Resistance indoor air-outdoor air
	Rair_outdoor = 1.0 / (A_facade * U + A_glass * Uglass + qm * c_air)  

	#Calculation of the thermal capacitances (heat capacities).

	#Heat capacity indoor air + walls:	Cair = rho_internal_mass * c_internal_mass * V_internal_mass / 2.0 + rho_air * c_air * V_dwelling    

	#Heat capacity walls(opt1): 			Cwall = rho_internal_mass * c_internal_mass * V_internal_mass / 2.0 
	#Heat capacity walls(opt2):   		Cwall = rho_facade * c_facade * V_facade / 2.0
	#	
	#		Volume facade walls construction [m3]:  V_facade = A_facade*th_facade  
	#												th_facade: Construction thickness [m]
	
	#__________Heat capacity indoor air + walls____________
	
    #Floor and internal walls construction
	#Light_weight = 0 / Middle_weight = 1  / Heavy_weight = 2
    N_internal_mass = hp['construction']['N_internal_mass']
	#Specific heat capacity construction [J/kgK]
    c_internal_mass = hp['thermal']['c_internal_mass'][N_internal_mass]  
	#Construction thickness [m]
    th_internal_mass = hp['construction']['th_internal_mass'][N_internal_mass]
	#Density construction in [kg/m3]	
    rho_internal_mass = hp['construction']['rho_internal_mass'][N_internal_mass]  
	#Volume floor and internal walls construction [m3]
    V_internal_mass = A_internal_mass * th_internal_mass

	#Heat capacity walls (Envelop model has both walls where both surfaces are used for energy storage -> the construction thickness/2)
    Cair = rho_internal_mass * c_internal_mass * V_internal_mass / 2.0 + rho_air * c_air * V_dwelling

	#___________Heat capacity walls____________________
	
	#Facade construction
    #Light_weight = 0 / Middle_weight = 1  / Heavy_weight = 2
    N_facade = hp['construction']['N_facade']
	#Specific heat capacity construction [J/kgK]
	c_facade = hp['thermal']['c_facade'][N_facade]
	#Construction thickness [m]	
    th_facade = hp['construction']['th_facade'][N_facade]  
	#Density construction in [kg/m3]
    rho_facade = hp['construction']['rho_facade'][N_facade]
	#Volume facade walls construction [m3]:
	V_facade = A_facade*th_facade	
	#Heat capacity walls (Envelop model has both walls where both surfaces are used for energy storage -> the construction thickness/2)
    Cwall = rho_internal_mass * c_internal_mass * V_internal_mass / 2.0  
	#Cwall = rho_facade * c_facade * V_facade / 2.0  # Capacity walls


    return Rair_wall, Cwall, Rair_outdoor, Cair

# Time base on 1 hour sampling from NEN
"""
time = Irr.qsunS[0]
# time = first row of Irr.qsunSouth (time axis) in seconds [0, 3600, 7200, ...]
print("ID time: ", id(time), ", ID Irr.qsunS[0]: ", id(Irr.qsunS[0]))

the "new" variable time is NOT at the same memory address as 
the "old" variable Irr.qsunS[0]!
because the value of the first element of an array is assigned to a scalar (float)
the instruction now has COPIED the variable
this asks for extreme programmer awareness!

# define window surface in m2
# Windows surface [E,SE,S,SW,W,NW,N,NE] [m2]
# -90 (E), -45 (SE), 0 (S), 45 (SW), 90 (W), 135 (NW), 180 (N), 225 (NE)
# Window solar transmittance, g-value
# Calculate Qsolar on window

Qsolar = (Irr.qsunE[1] * hp['glass']['E'] + Irr.qsunSE[1] * hp['glass']['SE'] +
          Irr.qsunS[1] * hp['glass']['S'] + Irr.qsunSW[1] * hp['glass']['SW'] +
          Irr.qsunW[1] * hp['glass']['W'] + Irr.qsunNW[1] * hp['glass']['NW'] +
          Irr.qsunN[1] * hp['glass']['N'] + Irr.qsunNE[1] * hp['glass']['NE']) * hp['g_value']

# with input NEN5060, glass and g_value, qsun can give a single result Qsolar

"""