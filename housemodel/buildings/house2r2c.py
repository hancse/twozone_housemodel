import numpy as np
from dataclasses import dataclass

from housemodel.tools.ckf_tools import (make_c_inv_matrix,
                                        add_c_inv_block,
                                        make_edges)


@dataclass
class HouseNode:
    label: str
    cap: float   # [J/K]
    temp: float  # [K]


@dataclass
class FixedNode:
    label: str
    temp: float     # [K]


@dataclass
class HouseEdge:
    label: str
    cond: float   # [W/K]
    conn_nodes: []   # empty list (tuple, ndarray)
    src: int
    sink: int


class House2R2C:
    def __init__(self):
        self.num_nodes = 2
        self.nodes = np.ndarray(self.num_nodes, dtype=HouseNode)
        self.edges = np.ndarray(self.num_nodes - 1)
        self.ambient = FixedNode

        self.c_mat = np.zeros((self.num_nodes, self.num_nodes))
        self.k_mat = np.zeros_like(self.c_mat)
        self.q_solar = None
        self.q_int = None


if __name__ == "__main__":
    h = House2R2C()
    c_list = [1.0, 2.0]
    c1 = make_c_inv_matrix(c_list)
    print(c1, "\n")
    cb = add_c_inv_block(c1, c1)
    print(cb)
