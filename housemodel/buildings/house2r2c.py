import numpy as np
from dataclasses import dataclass


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
    conn_nodes: list   # tuple, ndarray
    src: int
    snk: int


class House2R2C:
    def __init__(self):
        self.num_nodes = 2
        self.c_mat = np.zeros(self.num_nodes, self.num_nodes)
        self.k_mat = np.zeros_like(self.c_mat)
        self.f_mat = np.zeros_like(self.c_mat)
        self.nodes = np.array(self.num_nodes, type=HouseNode)

        self.edges = np.array(self.num_nodes - 1)


if __name__ == "__main__":
    pass
