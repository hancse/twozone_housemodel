import numpy as np
# from scipy.sparse import diags  # , spdiags
from scipy.linalg import block_diag
import networkx as nx
from housemodel.tools.new_configurator import load_config


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


def add_c_inv_block(first, second):
    return block_diag(first, second)


def add_k_block(first, second):
    return block_diag(first, second)


def stack_q(first, second):
    return np.vstack((first, second))
    # return np.concatenate(first, second)


def make_edges(edge_list):
    """

    Args:
        edge_list: list with edge info [source_node (int), target_node (int), weight (float)]

    Returns:
        K_matrix (ndarray):  2D matrix with conductances in network
    """
    G = nx.Graph()
    for e in edge_list:
        t = tuple(e)
        G.add_weighted_edges_from([t])
    A = nx.adjacency_matrix(G)
    B = A.toarray()
    row_sums = np.sum(B, axis=1).tolist()
    K_matrix = -B + np.diag(np.array(row_sums), k=0)
    return K_matrix


if __name__ == "__main__":
    lines = load_config(str("excel_for_companies_nodes_and_edges.yaml"))
    Kmat = make_edges(lines['edges'])
    print(Kmat)

    C = np.array([[1.0, 0.0],
                  [0.0, 2.0]])
    K = np.array([[0.1, 0.0],
                  [0.0, 0.2]])
    q = np.array([[0.5],
                  [0.6]])

    # https: // www.delftstack.com / howto / python / print - matrix - python /
    for row in C:
        print('  '.join(map(str, row)))
    print()

    for row in K:
        print('  '.join(map(str, row)))
    print()

    for row in q:
        print('  '.join(map(str, row)))
    print()

    q1 = np.array([[1], [2]])
    B = np.array([10, 11, 12]).reshape(-1, 1)
    q2 = np.array([[3], [4]])

    qt = np.add(q1, q2)
    print(qt)
    for row in qt:
        print('  '.join(map(str, row)))
    print()

