# import numpy as np
import networkx as nx
import numpy as np

from housemodel.tools.new_configurator import load_config


class Flow:
    def __init__(self, label="", flow_rate=50.0e-6, density=1000,
                       cp=4190, node_list=None):
        self.label = label
        self.flow_rate = flow_rate
        self.density = density
        self.cp = cp
        if node_list is None:
            self.node_list = []         # empty list
        else:
            self.node_list = node_list
        self.heat_rate = None
        # self.edges = []
        self.df_mat = None

        self.update_heat_rate()

    @classmethod
    def from_dict(cls, d):
        """ classmethod to enable constructing an instance from configuration file.
        """
        return cls(label=d["label"],flow_rate=d["flow_rate"], density=d["density"],
                   cp=d["cp"], node_list=d["pass_by"])

    def update_heat_rate(self):
        self.heat_rate = self.flow_rate * self.density * self.cp  # [J/ Ks) = [W/K]

    def set_flow_rate(self, new_flow_rate):
        self.flow_rate = new_flow_rate
        self.update_heat_rate()

    def make_df_matrix(self, rank: int):
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
        edges = [[i, j] for i, j in zip(self.node_list, self.node_list[1:])]
        # find missing elements in node_list
        res = list(set(range(rank)) - set(self.node_list))
        print(f"List of missing elements : {res}")

        G = nx.DiGraph()
        G.add_nodes_from(res)
        G.add_edges_from(edges)

        # print(f"nodes: {G.nodes}")
        A = nx.adjacency_matrix(G, nodelist=list(range(G.order())))
        B = A.toarray()
        # sign changes ?
        C = B - B.T
        self.df_mat = C.astype('float64')
        # self.df_mat = np.multiply(self.df_mat, (self.cp * self.density * self.flow_rate))

    """
    def flow_from_dict(self, d: dict):
        # initializes "nodes" attribute with data from yaml file

        Args:
            d: dict read from yaml file

        self.label = d["label"]
        self.flow_rate = d["flow_rate"]
        self.density = d["density"]
        self.cp = d["cp"]
        self.node_list = d["pass_by"]
    """

    """
    def nodelist_to_edges(self):
        # converts self.nodelist to edge pairs for networkx.

        # fl = list(range(10))
        # el = [[fl[i], fl[i+1]] for i in range(len(fl)-1)]
        # el2 = [[i, j] for i, j in zip(fl[:-1], fl[1:])]
        self.edges = [[i, j] for i, j in zip(self.node_list, self.node_list[1:])]
        # print(self.edges)
    """


if __name__ == "__main__":
    from pathlib import Path
    CONFIGDIR = Path(__file__).parent.parent.parent.absolute()
    param = load_config(str(CONFIGDIR / "for_2R2Chouse_buffer.yaml"))

    flows = []
    flows.append(Flow.from_dict(param['flows'][0]))
    # flows[0].flow_from_dict(param['flows'][0])
    flows.append(Flow.from_dict(param['flows'][1]))
    # flows[1].flow_from_dict(param['flows'][1])

    # flows[0].nodelist_to_edges()

    flows[0].make_df_matrix(7)
    print(flows[0].df_mat)
    print()

    flows[1].make_df_matrix(7)
    print(flows[1].df_mat)
    print()
