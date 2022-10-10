import numpy as np
from dataclasses import dataclass


@dataclass
class BufferNode:
    label: str
    cap: float
    temp: float


@dataclass
class BufferEdge:
    label: str
    src: int
    snk: int


class StratifiedBuffer:
    def __init__(self):
        self.num_layers = 3
        self.c_mat = np.array(self.num_layers, self.num_layers)
        self.k_mat = np.array(self.num_layers, self.num_layers)
        self.f_mat = np.array(self.num_layers, self.num_layers)
        self.nodes = np.array(self.num_layers, type=BufferNode)

        self.edges = np.array(self.num_layers - 1)


if __name__ == "__main__":
    pass
