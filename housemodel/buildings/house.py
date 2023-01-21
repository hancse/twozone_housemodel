import numpy as np

from housemodel.tools.ckf_tools import (make_c_inv_matrix,
                                        add_c_inv_block,
                                        make_edges)
from housemodel.tools.new_configurator import load_config
from housemodel.buildings.components import (CapacityNode,
                                             FixedNode,
                                             CondEdge)
import logging

logging.basicConfig(level="DEBUG")
# logging.basicConfig(level="INFO")
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)


class House:
    def __init__(self, name=""):
        self.name = name
        self.num_nodes = 0
        self.num_edges = 0
        self.nodes = []            # np.zeros(self.num_nodes, dtype=object)
        self.edges = []            # np.zeros(self.num_nodes - 1)
        self.boundaries = []
        self.ambient = None

        self.c_inv_mat = None  # np.zeros((self.num_nodes, self.num_nodes))
        self.k_mat = None      # np.zeros_like(self.c_inv_mat)
        self.q_vec = None      # np.zeros(self.num_nodes, 1)

        self.q_solar = None
        self.q_int = None

        self.tag_list = []
        self.cap_list = []
        self.cond_list = []

        logging.info(f" House object {self.name} created")

    def nodes_from_dict(self, lod: list):
        """initializes "nodes" attribute with data from yaml file
           makes a list from tags belonging to the House object

        Args:
            lod: list of dicts read from yaml file

        Returns:
            None
        """
        self.num_nodes = len(lod)
        for n in range(self.num_nodes):
            node = CapacityNode(label=lod[n]["label"],
                                tag=lod[n]["tag"],
                                cap=lod[n]["capacity"],
                                temp=lod[n]["T_ini"])
            # append by reference, therefore new node object in each iteration
            self.nodes.append(node)
            logging.debug(f" node '{node.label}' with tag {node.tag} appended to {self.name}")
        self.tag_list = [n.tag for n in self.nodes]
        logging.debug(f" tag_list {self.tag_list}")

    def fill_c_inv(self):
        self.cap_list = [n.cap for n in self.nodes]
        if len(self.cap_list) > 0:
            self.c_inv_mat = make_c_inv_matrix(self.cap_list)
            logging.debug(f" c_inv_matrix: \n {self.c_inv_mat}")
        else:
            logging.error(f" Error: cap_list empty")

    def edges_from_dict(self, lol):
        self.num_edges = len(lol)
        for n in range(self.num_edges):
            edge = CondEdge(label="",
                            conn_nodes=[lol[n][0], lol[n][1]],
                            cond=lol[n][2])
            self.edges.append(edge)
            logging.debug(f" edge from {edge.conn_nodes[0]} to {edge.conn_nodes[1] } appended to {self.name}")

    def fill_k(self, lol):
        """select local edges belonging to object and make k-matrix.

        Args:
            lol: list of edge lists [from, to, weight] read from config file
        """
        el = [e for e in lol if e[0] in self.tag_list and e[1] in self.tag_list]
        self.k_mat = make_edges(el)
        logging.debug(f" k_matrix: \n {self.k_mat}")

    def boundaries_from_dict(self, lod):
        for n in range(len(lod)):
            node = FixedNode(label=lod[n]["label"],
                             temp=lod[n]["T_ini"],
                             connected_to=lod[n]["connected_to"])
            # append by reference, therefore new node object in each iteration
            self.boundaries.append(node)
            logging.debug(f" boundary '{node.label}' appended to {self.name}")

        self.ambient = [fn for fn in self.boundaries if fn.label == "outdoor"][0]
        logging.debug(f" ambient is '{self.ambient.label}' for {self.name}")

    """
    def add_fixed_to_k(self):
        # add conductivities to boundary "ambient" to diagonal elements of k-matrix.

        # 
        # fnl = [fn for fn in self.boundaries for index in fn.connected_to if index[0] in self.tag_list]
        # res = []
        # [res.append(x) for x in fnl if x not in res]
        for c in self.ambient.connected_to:
            index = c[0]
            cond = c[1]
            self.k_mat[index, index] -= cond
            logging.debug(f" ambient connected to node '{self.nodes[index].label}'")
    """

    def add_ambient_to_k(self):
        """selectively add conductivity to boundary condition "ambient" to diagonal elements of k-matrix.

        """
        for c in self.ambient.connected_to:
            idx = self.tag_list.index(c[0])
            cond = c[1]
            self.k_mat[idx, idx] -= cond
            logging.debug(f" ambient connected to node '{self.nodes[idx].label}'")
        logging.debug(f" k_matrix: \n {self.k_mat}")

    def make_empty_q_vec(self):
        self.q_vec = np.zeros((self.num_nodes, 1))
        logging.debug(f" empty q-vector created of rank {self.num_nodes}")

    """
    def add_fixed_to_q(self):
        # add terms from ALL boundary conditions (external nodes) like T_outdoor and T_indoor.

           # - loops over ALL FixedNode object in self.boundaries
           # - for each FixedNode adds T/Rth to the corresponding element of self.q_vec
           # - the right element is found via the index of the tag in self.taglist
        
        for b in self.boundaries:
            for c in b.connected_to:
                idx = self.tag_list.index(c[0])
                cond = c[1]
                self.q_vec[idx] += cond * b.temp
                logging.debug(f" ambient added to q-vector element {idx}")
        logging.debug(f" q_vector: \n {self.q_vec}")
    """

    def add_ambient_to_q(self):
        """selectively add terms from boundary condition "ambient" to elements of q-vector.
        """
        for c in self.ambient.connected_to:
            idx = self.tag_list.index(c[0])
            cond = c[1]
            self.q_vec[idx] += cond * self.ambient.temp
            logging.debug(f" ambient added to q-vector element {idx} ({self.nodes[idx].label})")
        logging.debug(f" q_vector: \n {self.q_vec}")


if __name__ == "__main__":
    h = House()
    c_list = [1.0, 2.0]
    c1 = make_c_inv_matrix(c_list)
    print(c1, "\n")
    cb = add_c_inv_block(c1, c1)
    print(cb, '\n')
    k_list = [[0, 1, 1.0]]
    k1 = make_edges(k_list)
    print(k1, '\n')

    from pathlib import Path

    CONFIGDIR = Path(__file__).parent.absolute()
    house_param = load_config(str(CONFIGDIR / "xl_for_2R2Chouse_buffer.yml"))
    print()
