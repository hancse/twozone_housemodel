
import numpy as np

from housemodel.basics.ckf_tools import make_c_inv_matrix
from housemodel.tools.new_configurator import load_config
from housemodel.basics.components import (CapacityNode, CondEdge, FixedNode)
import logging

# logging.basicConfig(level="DEBUG")
logging.basicConfig(level="INFO")
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)


class Building:
    """ class for abstract representation of building topology with nodes and edges.

    Attributes:
        name (str): name of building.
        num_nodes (int): number of CapacityNode objects.
        num_edges (int): number of CondEdge objects.
        nodes (list): list of CapacityNode objects.
        edges (list): list of CondEdge objects.
        boundaries (list): list of FixedNode objects.
        ambient (FixedNode): FixedNode object with label "outdoor".
        c_inv_mat (array): inverse of capacity matrix C^{-1} [K/J]
        k_ext_mat (array): conductance of nodes to ambient [W/K].
        q_vec (array): vector with external heat sources/sinks [W].
        tag_list (list): list of tags of nodes.
        edge_list (list): list of connected node pairs
    """
    def __init__(self, name=""):
        self.name = name
        self.num_nodes = 0
        self.num_edges = 0
        self.nodes = []            # np.zeros(self.num_nodes, dtype=object)
        self.edges = []
        self.boundaries = []
        self.ambient = None

        self.c_inv_mat = None  # np.zeros((self.num_nodes, self.num_nodes))
        self.k_ext_mat = None  # np.zeros_like(self.c_inv_mat)
        self.q_vec = None      # np.zeros(self.num_nodes, 1)

        self.tag_list = []
        self.edge_list = []
        # self.cap_list = []

    @classmethod
    def from_dict(cls, d):
        """ classmethod to enable constructing an instance from configuration file.
        """
        return cls(name=d["name"])

    @classmethod
    def from_yaml(cls, config_yaml: str):
        """ classmethod to enable constructing an instance from configuration file.
        """
        b = cls()
        d = load_config(config_yaml)
        bd = d.get("Building")
        b.name = bd.get("name")
        logging.info(f" Building '{b.name}' created \n")
        # nds = bd.get("nodes")
        b.nodes_from_dict(bd.get("nodes"))
        # edgs = bd.get("edges")
        b.edges_from_dict(bd.get("edges"))
        # bndrs = d.get("boundaries")
        b.boundaries_from_dict(d.get("boundaries"))
        return b

    def nodes_from_dict(self, lod: list):
        """initializes "nodes" attribute with data from yaml file.

           makes a list from tags belonging to the Building object.

        Args:
            lod: list of dictionaries read from yaml file.
        """
        self.num_nodes = len(lod)
        for n in range(self.num_nodes):
            node = CapacityNode(label=lod[n]["label"],
                                tag=lod[n]["tag"],
                                cap=lod[n]["capacity"],
                                temp=lod[n]["T_ini"])
            # append by reference, therefore new node object in each iteration
            self.nodes.append(node)
            logging.info(f" node '{node.label}' with tag {node.tag} appended to {self.name}")
        self.tag_list = [n.tag for n in self.nodes]
        logging.info(f" tag_list {self.tag_list} \n")

    def fill_c_inv(self):
        """generate cap_list and fill c_inv_matrix.
        """
        self.cap_list = [n.cap for n in self.nodes]
        if len(self.cap_list) > 0:
            self.c_inv_mat = make_c_inv_matrix(self.cap_list)
            logging.info(f" c_inv_matrix: \n {self.c_inv_mat}")
        else:
            logging.error(f" Error: cap_list empty")

    def boundaries_from_dict(self, lod):
        """generate Fixed Node objects from configuration file
        choose "outdoor" node as ambient for Building class.

        Args:
            lod: list-of dicts read from "boundaries" section in .yaml configuration file.
        """
        for n in range(len(lod)):
            node = FixedNode(label=lod[n]["label"],
                             temp=lod[n]["T_ini"],
                             connected_to=lod[n]["connected_to"])
            # append by reference, therefore new node object in each iteration
            self.boundaries.append(node)
            logging.info(f" boundary '{node.label}' appended to {self.name}")

        self.ambient = [fn for fn in self.boundaries if fn.label == "outdoor"][0]
        logging.info(f" ambient is '{self.ambient.label}' for {self.name} \n")

    def make_k_ext_and_add_ambient(self):
        """make external "k_ext" matrix and selectively add conductivity
        to boundary condition "ambient" to diagonal elements.
        """
        if self.num_nodes > 0:                                            # c-1 matrix and rank has to be defined
            self.k_ext_mat = np.zeros((self.num_nodes, self.num_nodes))   # initialize with zeros
            for c in self.ambient.connected_to:
                idx = self.tag_list.index(c[0])
                cond = c[1]
                self.k_ext_mat[idx, idx] += cond
                logging.info(f" ambient connected to node '{self.nodes[idx].label}'")
            logging.info(f" k_ext matrix: \n {self.k_ext_mat}")

    # obsolete functions which are only relevant for TotalSystem class
    # and are not used in subsystem classes

    def edges_from_dict(self, lol):
        """read attributes "edges" and "edge_list" from list of edges (dictionary).

        Args:
            lol (list of lists):
        """
        if self.edge_list or self.edges:
            self.edges = []
            self.edge_list = []
        if not lol or len(lol) <= 0:
            return

        self.num_edges = len(lol)
        for n in range(self.num_edges):
            edge = CondEdge(label="",
                            conn_nodes=[lol[n][0], lol[n][1]],
                            cond=lol[n][2])
            self.edges.append(edge)
            self.edge_list.append(lol[n])
            logging.info(f" edge from {edge.conn_nodes[0]} to {edge.conn_nodes[1] } appended to {self.name}")
        logging.info(f" internal edges ready\n")

    """
    def fill_k(self, lol):
        # select local edges belonging to object and make k-matrix.

        Args:
            lol: list of edge lists [from, to, weight] read from config file
        
        el = [e for e in lol if e[0] in self.tag_list and e[1] in self.tag_list]
        self.k_mat = make_edges(el)
        logging.info(f" k_matrix: \n {self.k_mat}")
    """

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
            logging.info(f" ambient connected to node '{self.nodes[index].label}'")
    """

    """
    def make_empty_q_vec(self):
        self.q_vec = np.zeros((self.num_nodes, 1))
        logging.info(f" empty q-vector created of rank {self.num_nodes}")
    """

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
                logging.info(f" ambient added to q-vector element {idx}")
        logging.info(f" q_vector: \n {self.q_vec}")
    """

    """
    def add_ambient_to_q(self):
        # selectively add terms from boundary condition "ambient" to elements of q-vector.
        
        for c in self.ambient.connected_to:
            idx = self.tag_list.index(c[0])
            cond = c[1]
            self.q_vec[idx] += cond * self.ambient.temp
            logging.info(f" ambient added to q-vector element {idx} ({self.nodes[idx].label})")
        logging.info(f" q_vector: \n {self.q_vec}")
    """


if __name__ == "__main__":
    from pathlib import Path
    CONFIGDIR = Path(__file__).parent.parent.parent.absolute() / "tests"
    param = load_config(str(CONFIGDIR / "for_2R2Chouse_buffer.yaml"))

    h = Building("TestBuilding")
    section = param["Building"]
    # read nodes attribute from dictionary and create capacity matrix
    h.nodes_from_dict(section["nodes"])
    h.fill_c_inv()
    # read FixedNode objects (external nodes);
    h.boundaries_from_dict(param["boundaries"])  # function selects "outdoor" as ambient
    h.make_k_ext_and_add_ambient()  # initialize k_ext_mat and add diagonal elements
    logging.info(f" \n\n C^-1: \n {h.c_inv_mat} \n K_ext: \n {h.k_ext_mat}, \n q_vec: \n {h.q_vec} \n")
    print()

    # create Building object and read nodes, internal edges and boundaries
    h2 = Building.from_yaml(str(CONFIGDIR / "for_2R2Chouse_buffer.yaml"))
    # make C_inv matrix and K_ext matrix, print result
    h2.fill_c_inv()
    h2.make_k_ext_and_add_ambient()
    logging.info(f" \n\n C^-1: \n {h.c_inv_mat} \n\n "
                f"K_ext: \n {h.k_ext_mat}, \n\n "
                f"q_vec: \n {h.q_vec} \n")
    print()

