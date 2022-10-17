import numpy as np
from dataclasses import dataclass, field

from housemodel.tools.ckf_tools import (make_c_inv_matrix,
                                        add_c_inv_block,
                                        make_edges)
from housemodel.tools.new_configurator import load_config


@dataclass
class HouseNode:
    # first
    label: str = field(default="label")
    tag: int = field(default=-1)
    cap: float = field(default=0.0)     # [J/K]
    temp: float = field(default=20.0)  # [K]


@dataclass
class FixedNode:
    label: str
    connected_to: []
    temp: float     # [K]
# if methods are defined this turns into a normal class object


@dataclass
class HouseEdge:
    label: str
    conn_nodes: []  # empty list (tuple, ndarray)
    cond: float     # [W/K]
    # src: int
    # sink: int


@dataclass
class Flow:
    label: str
    flow_rate: float  # [m^3/s]
    density: float    # [kg/m^3]
    cp: float         # J/(kg K)]
    heat_rate: float  # [J/s = W]


class House:
    def __init__(self):
        self.num_nodes = 0
        self.num_edges = 0
        self.nodes = []           # np.zeros(self.num_nodes, dtype=object)
        self.edges = []           # np.zeros(self.num_nodes - 1)
        self.boundaries = []
        self.ambient = None

        self.c_inv_mat = None     # np.zeros((self.num_nodes, self.num_nodes))
        self.k_mat = None         # np.zeros_like(self.c_inv_mat)
        self.q_vec = None         # np.zeros(self.num_nodes, 1)

        self.q_solar = None
        self.q_int = None

        self.tag_list = []
        self.cap_list = []
        self.cond_list = []

    def nodes_from_dict(self, lod: list):
        """initializes "nodes" attribute with data from yaml file
           makes a list from tags belonging to the House object

        Args:
            lod: list of dicts read from yaml file

        Returns:
            None
        """
        self.num_nodes = len(lod)
        # node = HouseNode()
        for n in range(self.num_nodes):
            node = HouseNode(label=lod[n]["label"],
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
            edge = HouseEdge(label="",
                             conn_nodes=[lol[n][0], lol[n][1]],
                             cond=lol[n][2])
            self.edges.append(edge)

    def fill_k(self, lol):
        """ select local edges belonging to object and make k-matrix.

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
        self.ambient = [fn for fn in self.boundaries if fn.label == "outdoor"][0]

    def add_fixed_to_k(self):
        """

        Returns:

        """
        # fnl = [fn for fn in self.boundaries for index in fn.connected_to if index[0] in self.tag_list]
        # res = []
        # [res.append(x) for x in fnl if x not in res]
        for c in self.ambient.connected_to:
            index = c[0]
            cond = c[1]
            self.k_mat[index, index] -= cond

    def make_q_vec(self):
        self.q_vec = np.zeros((self.num_nodes, 1))

    def add_fixed_to_q(self):
        for c in self.ambient.connected_to:
            index = c[0]
            cond = c[1]
            self.q_vec[index] += cond


if __name__ == "__main__":
    h = House()
    c_list = [1.0, 2.0]
    c1 = make_c_inv_matrix(c_list)
    print(c1, "\n")
    cb = add_c_inv_block(c1, c1)
    print(cb, '\n')
    k_list =[[0, 1, 1.0]]
    k1 = make_edges(k_list)
    print(k1, '\n')

    from pathlib import Path
    CONFIGDIR = Path(__file__).parent.absolute()
    house_param = load_config(str(CONFIGDIR / "xl_for_2R2Chouse_buffer.yml"))
    print()
