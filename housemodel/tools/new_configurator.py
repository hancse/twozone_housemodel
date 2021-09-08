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
from scipy.sparse import diags  # , spdiags

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
    """make the diagonal C matrix.

    Args:
        capacity_list: list of node thermal capacities
    Returns:
        (array): diagonal 2d array with thermal capacities
    """
    cap_array = np.array(capacity_list)
    return np.diag(cap_array, k=0)


def make_c_inv_matrix(capacity_list: list):
    """make the reciprocal (inverse) of the diagonal C matrix.

    Args:
        capacity_list: list of node thermal capacities
    Returns:
        (array): diagonal 2d array with thermal capacities
    """

    cap_array = np.array(capacity_list)
    cap_array_reciprocal = np.reciprocal(cap_array)
    return np.diag(cap_array_reciprocal, k=0)


def make_k_matrix(conductance_list):
    """make the K matrix.

    Args:
        conductance_list: list of connecting thermal conductances

    Returns:
       (array): diagonal 2d array with thermal conductances

    Hint: use numpy.negative or unary minus operator (-)
    """
    cond_array = np.array(conductance_list)
    up_low = cond_array[1:]
    up_low_padded = np.pad(up_low, (0, 1))
    # adding [0] for now, more elegant solution? numpy.pad?
    main_diag = np.add(cond_array, up_low_padded)
    diagonals = [main_diag, up_low, -up_low]
    return diags(diagonals, [0, 1, -1]).toarray()


def make_k_minus_matrix(conductance_list):
    """make the negative of the K matrix.

    Args:
        conductance_list: list of connecting thermal conductances

    Returns:
       (array): diagonal 2d array with thermal conductances

    Hint: use numpy.negative or unary minus operator (-)
    """
    cond_array = np.array(conductance_list)
    up_low = cond_array[1:]
    up_low_padded = np.pad(up_low, (0, 1))
    # adding [0] for now, more elegant solution? numpy.pad?
    main_diag = np.add(cond_array, up_low_padded)
    diagonals = [-1.0 * main_diag, up_low, -up_low]
    return diags(diagonals, [0, 1, -1]).toarray()


def add_chain(C_mat, new_c_element,
              K_mat, new_k_element, anchor,
              q_vect, new_q_element):
    """

    Args:
        C_mat:
        new_c_element: 
        K_mat:
        new_k_element:
        anchor:
        q_vect:
        new_q_element:

    Returns:

    """
    new_c_mat = np.block([[C_mat, np.zeros((2, 1))],
                          [np.zeros((1, 2)), new_c_element]])
    new_k_mat = np.block([[K_mat, np.zeros((2, 1))],
                          [np.zeros((1, 2)), 0]])
    idx = new_k_mat.shape[0] - 1
    new_k_mat[anchor, anchor] += new_k_element
    new_k_mat[idx, idx] += new_k_element
    new_k_mat[anchor, idx] = -new_k_element
    new_k_mat[idx, anchor] = -new_k_element
    new_q_vect = np.vstack((q_vect, new_q_element))
    return new_c_mat, new_k_mat, new_q_vect


if __name__ == "__main__":
    C = np.array([[1.0, 0.0],
                  [0.0, 2.0]])
    K = np.array([[0.1, 0.0],
                  [0.0, 0.2]])
    q = np.array([[0.5],
                  [0.6]])
    new_C, new_K, new_q = add_chain(C, 3.0,
                                    K, 0.3, 0,
                                    q, 0.7)

    for row in C:
        print('  '.join(map(str, row)))
    print()
    for row in new_C:
        print('  '.join(map(str, row)))
    print()

    for row in K:
        print('  '.join(map(str, row)))
    print()
    for row in new_K:
        print('  '.join(map(str, row)))
    print()

    for row in q:
        print('  '.join(map(str, row)))
    print()
    for row in new_q:
        print('  '.join(map(str, row)))
