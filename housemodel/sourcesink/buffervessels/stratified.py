import numpy as np
from dataclasses import dataclass

from housemodel.tools.ckf_tools import (make_c_inv_matrix,
                                        add_c_inv_block,
                                        make_edges)


@dataclass
class BufferNode:
    label: str
    tag: int
    cap: float
    temp: float


@dataclass
class FixedNode:
    label: str
    connected_to: []
    temp: float     # [K]
# if methods are defined this turns into a normal class object


@dataclass
class BufferEdge:
    label: str
    conn_nodes: []  # empty list (tuple, ndarray)
    cond: float  # [W/K]
    # src: int
    # sink: int


class StratifiedBuffer:
    def __init__(self):
        self.num_layers = 0
        self.num_edges = 0
        self.nodes = []           # np.zeros(self.num_nodes, dtype=object)
        self.edges = []           # np.zeros(self.num_nodes - 1)
        self.boundaries = []
        self.ambient = None

        self.c_inv_mat = None  # np.array(self.num_layers, self.num_layers)
        self.k_mat = None  # np.array(self.num_layers, self.num_layers)
        self.f_mat = None  # np.array(self.num_layers, self.num_layers)
        self.q_vec = None  # np.zeros(self.num_nodes, 1)

        self.tag_list = []
        self.cap_list = []
        self.cond_list = []

    def nodes_from_dict(self, lod: list):
        """initializes "nodes" attribute with data from yaml file

        Args:
            lod: list of dicts read from yaml file

        Returns:
            None
        """
        self.num_layers = len(lod)
        for n in range(self.num_layers):
            node = BufferNode(label=lod[n]["label"],
                              tag=lod[n]["tag"],
                              cap=lod[n]["capacity"],
                              temp=lod[n]["T_ini"])
            # append by reference, therefore new node object in each iteration
            self.nodes.append(node)
        self.tag_list = [n.tag for n in self.nodes]

    def fill_c_inv(self):
        self.cap_list = [n.cap for n in self.nodes]
        if len(self.cap_list) > 0:
            self.c_inv_mat = make_c_inv_matrix(self.cap_list)
        else:
            print(f"Error: cap_list empty")

    def edges_from_dict(self, lol):
        self.num_edges = len(lol)
        for n in range(self.num_edges):
            edge = BufferEdge(label="",
                              conn_nodes=[lol[n][0], lol[n][1]],
                              cond=lol[n][2])
            self.edges.append(edge)

    def fill_k(self, lol):
        """select local edges belonging to object and make k-matrix.

        Args:
            lol: list of edge lists [from, to, weight]

        Returns:

        """
        el = [e for e in lol if e[0] in self.tag_list and e[1] in self.tag_list]
        self.k_mat = make_edges(el)

    def boundaries_from_dict(self, lod):
        for n in range(len(lod)):
            node = FixedNode(label=lod[n]["label"],
                             temp=lod[n]["T_ini"],
                             connected_to=lod[n]["connected_to"])
            # append by reference, therefore new node object in each iteration
            self.boundaries.append(node)
        self.ambient = [fn for fn in self.boundaries if fn.label == "indoor"][0]

    def add_fixed_to_k(self):
        """

        Returns:

        """
        for c in self.ambient.connected_to:
            index = c[0]
            cond = c[1]
            self.k_mat[index, index] -= cond

    def make_q_vec(self):
        self.q_vec = np.zeros((self.num_layers, 1))

    def add_fixed_to_q(self):
        for c in self.ambient.connected_to:
            index = c[0]
            cond = c[1]
            self.q_vec[index] += cond


if __name__ == "__main__":
    pass
