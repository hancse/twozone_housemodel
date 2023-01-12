# import numpy as np
import networkx as nx
import numpy as np

from housemodel.tools.new_configurator import load_config


class Flow:
    def __init__(self):
        self.label = None
        self.rate = None
        self.density = None
        self.cp = None
        self.node_list = []  # empty list
        self.edges = []
        self.f_mat = None

    def flow_from_dict(self, d: dict):
        """initializes "nodes" attribute with data from yaml file

        Args:
            d: dict read from yaml file

        Returns:
            None
        """
        self.label = d["label"]
        self.rate = d["flow_rate"]
        self.density = d["density"]
        self.cp = d["cp"]
        self.node_list = d["pass_by"]

    def nodelist_to_edges(self):
        """converts self.nodelist to edge pairs for networkx.

        """
        # fl = list(range(10))
        # el = [[fl[i], fl[i+1]] for i in range(len(fl)-1)]
        # el2 = [[i, j] for i, j in zip(fl[:-1], fl[1:])]
        self.edges = [[i, j] for i, j in zip(self.node_list, self.node_list[1:])]
        # print(self.edges)

    def make_Fmatrix(self, rank: int):
        """converts nodelist into edges and F-matrix.

        Args:
            rank (int): order od total system matrix
        Returns:
            None
        """
        # check if attribute node_list exists
        if not self.node_list:
            return()
        # make list with edges from node_list
        self.edges = [[i, j] for i, j in zip(self.node_list, self.node_list[1:])]
        # find missing elements in node_list
        res = list(set(range(rank)) - set(self.node_list))
        print(f"List of missing elements : {res}")

        G = nx.DiGraph()
        G.add_nodes_from(res)
        G.add_edges_from(self.edges)

        # print(f"nodes: {G.nodes}")
        A = nx.adjacency_matrix(G, nodelist=list(range(G.order())))
        B = A.toarray()
        # sign changes ?
        C = B - B.T
        self.f_mat = C.astype('float64')
        self.f_mat = np.multiply(self.f_mat,
                                 (self.cp * self.density * self.rate))


if __name__ == "__main__":
    from pathlib import Path
    CONFIGDIR = Path(__file__).parent.parent.absolute() / "buildings"
    param = load_config(str(CONFIGDIR / "xl_for_2R2Chouse_buffer.yml"))

    flows = []
    flows.append(Flow())
    flows[0].flow_from_dict(param['flows'][0])
    flows.append(Flow())
    flows[1].flow_from_dict(param['flows'][1])

    # flows[0].nodelist_to_edges()

    flows[0].make_Fmatrix(7)
    print(flows[0].f_mat)
    print()

    flows[1].make_Fmatrix(7)
    print(flows[1].f_mat)
    print()