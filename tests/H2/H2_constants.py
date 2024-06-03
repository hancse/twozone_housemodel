
import numpy as np


RGAS = 8.3144
EPS_WATER = 78.54               # relative permittivity water
EPS_ZERO = 8.8541878128e-12     # permittivity vacuum
EPS_ZW = EPS_ZERO * EPS_WATER   # often together
E_CHARGE = 1.602e-19            # elementary charge
BOLTZ_T = 1.38e-23 * 298        # Boltzmann constant x temperature
N_AVO = 6.02e23                 # Avogadro's number
FARADAY = N_AVO * E_CHARGE
ETA_VISC = 0.89e-3              # viscosity water
TWOPI = 2.0 * np.pi
RHO_W = 1000                    # density water [kg/m^3]
U_IDEAL = 1.229                 # ideal cell voltage in [V]

