# https://www.dangtrinh.com/2015/08/excel-to-list-of-dictionaries-in-python.html

import numpy as np
import pandas as pd
import openpyxl
import string
# from ruamel import yaml
import yaml
# from scipy.sparse import diags  # , spdiags
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")


def C_from_elements(elements: dict):
    """

    Args:
        elements: list with edge info [source_node (int), target_node (int), weight (float)]

    Returns:
        C_matrix (ndarray):  2D matrix with conductances in network
    """
    G = nx.Graph()
    t = []
    for e in elements:
        # t.append([e['NodeA'], e['NodeB'], 1.0])   # e['Capacity']])
        t.append([e['NodeA'], e['NodeB'], 0.5 * e['Capacity']])
    G.add_weighted_edges_from(t)
    print(G.nodes)

    # H = nx.Graph()
    # H.add_nodes_from(sorted(G.nodes(data=True)))
    # H.add_edges_from(G.edges(data=True))
    # print(H.nodes())

    # nx.draw(G, with_labels=True)
    # plt.show()

    A = nx.adjacency_matrix(G, nodelist=list(range(G.order())))
    B = A.toarray()
    print(B, "\n")

    row_sums = np.sum(B, axis=1).tolist()
    C_matrix = np.diag(np.array(row_sums), k=0)
    return C_matrix


def K_from_elements(elements: dict):
    """

    Args:
        e: list with edge info [source_node (int), target_node (int), weight (float)]

    Returns:
        K_matrix (ndarray):  2D matrix with conductances in network
    """
    G = nx.Graph()
    t = []
    for e in elements:
        # t.append([e['NodeA'], e['NodeB'], 1.0])
        t.append([e['NodeA'], e['NodeB'], e['Conductivity']])
    G.add_weighted_edges_from(t)
    print(G.nodes)

    # H = nx.Graph()
    # H.add_nodes_from(sorted(G.nodes(data=True)))
    # H.add_edges_from(G.edges(data=True))
    # print(H.nodes())

    # nx.draw(H, with_labels=True)
    # plt.show()

    A = nx.adjacency_matrix(G, nodelist=list(range(G.order())))
    B = A.toarray()
    print(B, "\n")

    row_sums = np.sum(B, axis=1).tolist()
    K_matrix = B - np.diag(np.array(row_sums), k=0)
    return K_matrix


def flowlist_to_edges(fl: list):
    # fl = list(range(10))
    # el = [[fl[i], fl[i+1]] for i in range(len(fl)-1)]
    # el2 = [[i, j] for i, j in zip(fl[:-1], fl[1:])]
    el3 = [[i, j] for i, j in zip(fl, fl[1:])]
    # print(el3)
    return el3


def flow_to_F_matrix(flowlist: list, rank: int):
    # flatten
    flattened = [val for f in flowlist for val in f]

    print("The original list : " + str(flattened))
    # Finding missing elements in List
    # res = [ele for ele in range(max(flattened) + 1) if ele not in flattened]
    res = list(set(range(rank)) - set(flattened))
    print("The list of missing elements : " + str(res))

    G = nx.DiGraph()
    G.add_nodes_from(res)
    for f in flowlist:
        G.add_edges_from(flowlist_to_edges(f))

    print(f"nodes: {G.nodes}")
    A = nx.adjacency_matrix(G, nodelist=list(range(G.order())))
    B = A.toarray()
    # sign changes ?
    C = B - B.T
    return C


if __name__ == "__main__":
    df = pd.read_excel('xl_for_yaml.xlsx')
    dd = df.T.to_dict()
    print(dd)
    dlist = [value for value in dd.values()]
    print(dlist)

    with open("xl_for_yaml.yml", "w") as config_outfile:
        yaml.dump(dlist, config_outfile, indent=2, sort_keys=False)

    Cmatrix = C_from_elements(dlist)
    print(Cmatrix, "\n")

    Kmatrix = K_from_elements(dlist)
    print(Kmatrix)

    df = pd.read_excel('xl_for_yaml.xlsx', sheet_name='flows')

    flows = []
    for row in range(len(df.index)):
        flows.append(df.iloc[row].values.tolist())  ## This will give you 7th row

    F = []
    for flow in flows:
        nodelist = [x for x in flow[3:] if np.isnan(x) == False]
        F.append(nodelist)

    Fmatrix = []
    for f in F:
         Fmatrix.append(flow_to_F_matrix([f], 7))  # first argument is a LIST, not a SCALAR
    print(Fmatrix)

    for n in range(len(flows)):
        factor = flows[n][2]
        Fmatrix[n] *= factor
    print(Fmatrix)

    Fall = np.zeros_like(Fmatrix[0])
    for n in range(len(Fmatrix)):
        Fall += Fmatrix[n]
    print(Fall)

    Fall = np.where( Fall <= 0, Fall, 0)
    print(Fall)

    row_sums = np.sum(Fall, axis=1).tolist()
    Fall = Fall - np.diag(np.array(row_sums), k=0)
    print(Fall)










