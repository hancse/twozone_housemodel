# Function to make an initial huess for the r and C values of a house
#   prepared by Maarten van den Berg

#   INPUTS:
# A_facade, Surface area of the facade
# Rc_facade, Thermal resistance of the facade
# Aglass, Surface area of the glass
# Uglass, Thermal transmittance of the glass
# n, Number of ventilations of the house per hour
# V_dwelling Volume of the house
# A_internal_mass, Surface aera of the internal structure of the house
# N_internal_mass, Type of strucure of the internal mass
# N_facade, Type of strucure of the facade

def RC_calculator_2R2C(A_facade, Rc_facade, Aglass, Uglass, n, V_dwelling, A_internal_mass, N_internal_mass, N_facade):
    # Density of air:
    rho_air = 1.20
    # Specific heat of air:
    c_air=1005

    alpha_i_facade=8
    alpha_e_facade=23
    alpha_internal_mass=8

    if N_internal_mass == 1:  # Light weight construction
        c_internal_mass = 840  # Specific heat capacity construction[J / kgK]
        th_internal_mass = 0.1  # Construction thickness[m]
        rho_internal_mass = 500  # Density construction in [kg / m3]

    elif N_internal_mass == 2:  # Middle weight construction
        c_internal_mass = 840  # Specific heat capacity construction[J / kgK]
        th_internal_mass = 0.1  # Construction thickness[m]
        rho_internal_mass = 1000  # Density construction in [kg / m3]

    else: # Heavy weight construction
        c_internal_mass = 840  # Specific heat capacity construction[J / kgK]
        th_internal_mass = 0.2  # Construction thickness[m]
        rho_internal_mass = 2500  # Density construction in [kg / m3]

    V_internal_mass=A_internal_mass*th_internal_mass   # Volume floor and internal walls construction [m3]
    qV=(n*V_dwelling)/3600            # Ventilation, volume air flow [m3/s]
    qm=qV*rho_air                     # Ventilation, mass air flow [kg/s]

    # Calculation of the resistances
    Rair_wall = 1/(A_internal_mass*alpha_internal_mass)  # Resistance indoor air-wall
    U = 1/(1/alpha_i_facade+Rc_facade+1/alpha_e_facade)  # U-value indoor air-facade
    Rair_outdoor = 1/(A_facade*U+Aglass*Uglass+qm*c_air)  # Resitance indoor air-outdoor air

    # Calculation of the capacities
    C_indoor = rho_internal_mass*c_internal_mass*V_internal_mass/2 + rho_air*c_air*V_dwelling  # Capacity indoor air + walls
    C_wall = rho_internal_mass*c_internal_mass*V_internal_mass/2  # Capacity walls
    return Rair_outdoor, Rair_wall, C_indoor, C_wall

if __name__ == "__main__":
    r0, r01, c0, c1 = RC_calculator_2R2C(100, 1.3, 18, 2.9, 0.55, 275.6, 170, 2, 2)
