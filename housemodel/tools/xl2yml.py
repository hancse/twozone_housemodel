
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


def index_to_col(index):
    # return string.uppercase[index]
    return openpyxl.utils.cell.get_column_letter(index+1)


def excel_to_dict(excel_path, headers=[]):
    wb = openpyxl.load_workbook(excel_path)
    sheet = wb['Sheet1']
    result_dict = []
    for row in range(2, sheet.max_row+1):
        line = dict()
        for header in headers:
            cell_value = sheet[index_to_col(headers.index(header)) + str(row)].value
            if type(cell_value) == 'unicode':
                cell_value = cell_value.encode('utf-8').decode('ascii', 'ignore')
                cell_value = cell_value.strip()
            elif type(cell_value) is int:
                cell_value = str(cell_value)
            elif cell_value is None:
                cell_value = ''
            line[header] = cell_value
        result_dict.append(line)
    return result_dict

def C_from_elements(elements:dict):
    """

    Args:
        e: list with edge info [source_node (int), target_node (int), weight (float)]

    Returns:
        C_matrix (ndarray):  2D matrix with conductances in network
    """
    G = nx.Graph()
    t = []
    for e in elements:
        # t.append([e['NodeA'], e['NodeB'], 1.0])   # e['Capacity']])
        t.append([e['NodeA'], e['NodeB'], 0.5*e['Capacity']])
    G.add_weighted_edges_from(t)
    print(G.nodes)

    # H = nx.Graph()
    # H.add_nodes_from(sorted(G.nodes(data=True)))
    # H.add_edges_from(G.edges(data=True))
    # print(H.nodes())

    nx.draw(G, with_labels=True)
    plt.show()

    A = nx.adjacency_matrix(G, nodelist=list(range(G.order())))
    B = A.toarray()
    print(B, "\n")

    row_sums = np.sum(B, axis=1).tolist()
    C_matrix = np.diag(np.array(row_sums), k=0)
    return C_matrix

def K_from_elements(elements:dict):
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


if __name__ == "__main__":
    # from excel_utils import excel_to_dict
    data = excel_to_dict('xl_for_yaml.xlsx', ['Element', 'Name', 'Capacity', 'Conductivity', 'Flow', 'NodeA', 'NodeB'])
    for d in data:
        print(d)
    with open("xl_for_yaml.yml", "w") as config_outfile:
        yaml.dump(data, config_outfile, indent=4, sort_keys=False)

    df = pd.read_excel('xl_for_yaml.xlsx')
    dd = df.T.to_dict()
    print(dd)
    dlist = [value for value in dd.values()]
    print(dlist)

    C = C_from_elements(dlist)
    print(C , "\n")

    K = K_from_elements(dlist)
    print(K)

