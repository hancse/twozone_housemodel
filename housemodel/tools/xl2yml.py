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

def plot_graph(g):
    # g = nx.Graph()
    g.add_nodes_from(sorted(G.nodes(data=True)))
    g.add_edges_from(G.edges(data=True))
    print(g.nodes())
    nx.draw(g, with_labels=True)
    plt.show()


def C_from_elements(elements: list):
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
    # plot_graph(G)

    A = nx.adjacency_matrix(G, nodelist=list(range(G.order())))
    B = A.toarray()
    print(B, "\n")

    row_sums = np.sum(B, axis=1).tolist()
    C_matrix = np.diag(np.array(row_sums), k=0)
    return C_matrix


def C_from_elements2(df: pd.DataFrame):
    """assemble C-matrix from Dataframe

    Args:
        df: Dataframe from Excel spreadsheet

    Returns:
        lumped mass matrix with heat capacity of nodes
    """
    # convert Dataframe into list of spreadsheet rows, called "rows"
    # rows becomes a list of lists
    rows = []
    for row in range(len(df.index)):
        rows.append(df.iloc[row].values.tolist())

    # extract element 4: and element 2 of each row into "nodelists"
    nodelists = []
    for row in rows:
        nodelist = [x for x in row[4:] if np.isnan(x) == False]
        nodelist.append(0.5 * row[2])
        nodelists.append(nodelist)

    # nodelists is a list [node, node, weight], suitable for networkx
    G = nx.Graph()
    G.add_weighted_edges_from(nodelists)
    print(G.nodes)
    # plot_graph(G)

    A = nx.adjacency_matrix(G, nodelist=list(range(G.order())))
    B = A.toarray()
    print(B, "\n")

    row_sums = np.sum(B, axis=1).tolist()
    C_matrix = np.diag(np.array(row_sums), k=0)
    return C_matrix


def K_from_elements(elements: list):
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
    # plot_graph(G)

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


def F_from_flows(flows: dict):
    # convert Dataframe into list of spreadsheet rows, called "rows"
    rows = []
    for row in range(len(flows.index)):
        rows.append(flows.iloc[row].values.tolist())  ## This will give you 7th row

    # extract element 3: of each row into "nodelists"
    nodelists = []
    for row in rows:
        nodelist = [x for x in row[3:] if np.isnan(x) == False]
        nodelists.append(nodelist)

    # convert each nodelist into an adjacency matrix resulting in a list of matrices called "Fmatrices"
    Fmatrices = []
    for nl in nodelists:
        Fmatrices.append(flow_to_F_matrix([nl], 7))  # first argument is a LIST, not a SCALAR
    print(Fmatrices)

    # multiply each adjacency matrix with a factor found in element 2 of each row
    for n in range(len(rows)):
        factor = rows[n][2]
        Fmatrices[n] *= factor
    print(Fmatrices)

    # combine Fmatrices into matrix Fall
    Fall = np.zeros_like(Fmatrices[0])
    for n in range(len(Fmatrices)):
        Fall += Fmatrices[n]
    print(Fall)

    # remove matrix elements < 0 from Fall
    Fall = np.where(Fall <= 0, Fall, 0)
    print(Fall)

    # create diagonal elements in Fall, so that som over each row is zero
    row_sums = np.sum(Fall, axis=1).tolist()
    Fall = Fall - np.diag(np.array(row_sums), k=0)
    print(Fall)

    return Fall


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
    Fmatrix = F_from_flows(df)
    print(Fmatrix)

    df = pd.read_excel('xl_for_yaml.xlsx', sheet_name='elements_new')
    Cmatrix2 = C_from_elements2(df)
    np.testing.assert_array_almost_equal(Cmatrix, Cmatrix2)











