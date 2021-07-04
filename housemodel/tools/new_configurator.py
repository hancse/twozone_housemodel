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
import numpy as np
import yaml
from scipy.sparse import diags, spdiags

"""
The predefined variables are now defined in a configuration file
All parameters read from configuration (*.yml) file 
"""


def load_config(config_name: str):
    """

    Args:
        config_name:

    Returns:

    """
    with open(config_name) as config_file:
        hp = yaml.safe_load(config_file)  # hp = house_parameters
        return hp


def save_config(hp):
    """

    Args:
        hp:

    Returns:

    """
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
    Rair_outdoor = 1.0 / hp['chains'][0]['links'][0]['Conductance']
    Cair = hp['chains'][0]['links'][0]['Capacity']
    Rair_wall = 1.0 / hp['chains'][0]['links'][1]['Conductance']
    Cwall = hp['chains'][0]['links'][1]['Capacity']

    return Rair_wall, Cwall, Rair_outdoor, Cair

def make_c_matrix(capacity_list: list):
    """make C matrix.

    Args:
        capacity_list: list of node thermal capacities
    Returns:
        (ndarray): diagonal 2d array with thermal capacities
    """
    cap_array = np.array(capacity_list)
    return np.diag(cap_array, k=0)


def make_k_matrix(conductance_list):
    """make K matrix.

    Args:
        conductance_list: list of connecting thermal conductances

    Returns:
       (ndarray): diagonal 2d array with thermal conductances
    """
    cond_array = np.array(conductance_list)
    up_low = cond_array[1:]
    up_low_padded = np.pad(up_low, (0, 1))
    # adding [0] for now, more elegant solution? numpy.pad?
    main_diag = np.add(cond_array, up_low_padded)
    diagonals = [main_diag, -1.0 * up_low, -1.0 * up_low]
    return diags(diagonals, [0, 1, -1]).toarray()
