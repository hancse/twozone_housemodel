import numpy as np


class Flow:
    def __init__(self):
        self.rate = None
        self.label = None
        self.density = None
        self.cp = None
        self.node_list = []  # empty list
        self.f_mat = None

    def nodes_from_dict(self, lod: list):
        """initializes "nodes" attribute with data from yaml file

        Args:
            lod: list of dicts read from yaml file

        Returns:
            None
        """
        self.num_layers = len(lod)
        for n in range(self.num_layers):
            node = CapacityNode(label=lod[n]["label"],
                              tag=lod[n]["tag"],
                              cap=lod[n]["capacity"],
                              temp=lod[n]["T_ini"])
            # append by reference, therefore new node object in each iteration
            self.nodes.append(node)
        self.tag_list = [n.tag for n in self.nodes]

    def calc_fmatrix(self):
        pass


if __name__ == "__main__":
    pass
