import numpy as np

from housemodel.basics.ckf_tools import make_c_inv_matrix
from housemodel.tools.new_configurator import load_config
from housemodel.basics.components import (CapacityNode, FixedNode)
import logging

# logging.basicConfig(level="DEBUG")
logging.basicConfig(level="INFO")


# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)


class StratifiedBuffer:
    def __init__(self, name=""):
        self.name = name
        self.num_nodes = 0  # = number of layers in the vessel
        self.num_edges = 0
        self.nodes = []  # np.zeros(self.num_nodes, dtype=object)
        self.edges = []  # np.zeros(self.num_nodes - 1)
        self.boundaries = []
        self.ambient = None

        self.c_inv_mat = None  # np.zeros((self.num_nodes, self.num_nodes))
        self.k_mat = None  # np.zeros_like(self.c_inv_mat)
        self.k_ext_mat = None  # np.zeros_like(self.c_inv_mat)
        self.q_vec = None  # np.zeros(self.num_nodes, 1)
        self.f_mat = None  # np.zeros(self.num_nodes, self.num_nodes)

        self.tag_list = []
        self.cap_list = []
        self.cond_list = []

        logging.info(f" StratifiedBuffer object {self.name} created")

    def nodes_from_dict(self, lod: list):
        """initializes "nodes" attribute with data from yaml file
           makes a list from tags belonging to the StratifiedBuffer object

        Args:
            lod: list of dicts read from yaml file
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
        logging.info(f" tag_list {self.tag_list}")

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
        """generate Fixed Node objects from configuration file.

        choose "indoor" node as ambient for StratifiedBuffer class

        Args:
            lod: list-of dicts read from "boundaries" section in *.yaml configuration file

        """
        for n in range(len(lod)):
            node = FixedNode(label=lod[n]["label"],
                             temp=lod[n]["T_ini"],
                             connected_to=lod[n]["connected_to"])
            # append by reference, therefore new node object in each iteration
            self.boundaries.append(node)
            logging.info(f" boundary '{node.label}' appended to {self.name}")

        self.ambient = [fn for fn in self.boundaries if fn.label == "indoor"][0]
        logging.info(f" ambient is '{self.ambient.label}' for {self.name}")

    def make_k_ext_and_add_ambient(self):
        """make external "k_ext" matrix and selectively add conductivity to boundary condition "ambient"
        to diagonal elements

        """
        if self.num_nodes > 0:  # c-1 matrix and rank has to be defined
            self.k_ext_mat = np.zeros((self.num_nodes, self.num_nodes))  # initialize with zeros
            for c in self.ambient.connected_to:
                idx = self.tag_list.index(c[0])
                cond = c[1]
                self.k_ext_mat[idx, idx] += cond
                logging.info(f" ambient connected to node '{self.nodes[idx].label}'")
            logging.info(f" k_ext matrix: \n {self.k_ext_mat}")

    # obsolete functions which are only relevant for TotalSystem class
    # and are not used in subsystem classes

    """
    def edges_from_dict(self, lol):
        self.num_edges = len(lol)
        for n in range(self.num_edges):
            edge = CondEdge(label="",
                            conn_nodes=[lol[n][0], lol[n][1]],
                            cond=lol[n][2])
            self.edges.append(edge)
            logging.info(f" edge from {edge.conn_nodes[0]} to {edge.conn_nodes[1] } appended to {self.name}")
    """

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


class StratifiedBufferNew:
    """parent class for cylindrical stratified buffer vessel

    """

    def __init__(self, name="DefaultBuffer", begin_tag=0, num_layers=5,
                 volume=0.1, height=1.0, U_wall=0.12, T_ini=20):
        self.name = name
        self.begin_tag = begin_tag   # anchor point of buffer vessel in house model
        self.num_nodes = num_layers
        self.volume = volume
        self.height = height
        self.U_wall = U_wall         # conductivity vessel wall to ambient in [W/K m^2]
        self.T_ini = T_ini

        self.end_node = self.begin_tag + num_layers - 1  # anchor point of cold water return from house model
        self.nodes = []
        self.edges = []
        self.boundaries = []
        self.ambient = None

        self.tag_list = []
        self.cap_list = []
        self.edge_list = []

        self.c_inv_mat = None
        self.k_int_mat = None
        self.k_ext_mat = None

        self.rho = 1000  # [kg/m^3]
        self.cp = 4190  # [J/(K kg)]
        self.conductivity = 0.644  # [W/m K]  # specific thermal conductivity between layers (lambda or k)

        self.A_base = None  # contact area between layers (= area of vessel bottom and top face)
        self.radius = None
        self.layer_height = None
        self.A_wall_layer = None
        self.cap_layer = None

        self.temperatures = None
        self.calculate_buffer_properties()

    @classmethod
    def from_dict(cls, d):
        """ classmethod to enable constructing an instance from configuration file.
        """
        return cls(name=d["name"],begin_tag=d["begin_tag"], num_layers=d["num_layers"],
                            volume=d["volume"], height=d["height"],
                            U_wall=d["U_wall"], T_ini=d["T_ini"])

    def calculate_buffer_properties(self):
        self.A_base = self.volume / self.height
        self.radius = np.sqrt(self.A_base / np.pi)
        self.layer_height = self.height / self.num_nodes
        self.A_wall_layer = 2 * np.pi * self.radius * self.layer_height
        self.cap_layer = (self.volume / self.num_nodes) * self.rho * self.cp

    def set_volume(self, vol):
        self.volume = vol
        self.calculate_buffer_properties()

    def set_height(self, h):
        self.height = h
        self.calculate_buffer_properties()

    def set_U_wall(self, u):
        self.U_wall = u

    def set_T_ini(self, t):
        self.T_ini = t

    def set_rho(self, r):
        self.rho = r
        self.calculate_buffer_properties()

    def set_cp(self, cp):
        self.cp = cp
        self.calculate_buffer_properties()

    def set_cond_medium(self, c):
        self.conductivity = c

    def generate_nodes(self):
        """initializes "nodes" attribute with data from yaml file
           makes a list from tags belonging to the StratifiedBuffer object

        """
        for n in range(self.num_nodes):
            node = CapacityNode(label=f"{self.name}{n}",
                                tag=self.begin_tag + n,
                                cap=self.cap_layer,
                                temp=self.T_ini)
            # append by reference, therefore new node object in each iteration
            self.nodes.append(node)
            logging.info(f" node '{node.label}' with tag {node.tag} appended to {self.name}")
        self.tag_list = [n.tag for n in self.nodes]
        logging.info(f" tag_list {self.tag_list}")

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
        """generate Fixed Node objects from configuration file.

        choose "indoor" node as ambient for StratifiedBuffer class

        Args:
            lod: list-of dicts read from "boundaries" section in *.yaml configuration file

        """
        for n in range(len(lod)):
            node = FixedNode(label=lod[n]["label"],
                             temp=lod[n]["T_ini"],
                             connected_to=lod[n]["connected_to"])
            # append by reference, therefore new node object in each iteration
            self.boundaries.append(node)
            logging.info(f" boundary '{node.label}' appended to {self.name}")

        self.ambient = [fn for fn in self.boundaries if fn.label == "indoor"][0]
        logging.info(f" ambient is '{self.ambient.label}' for {self.name}")

    def make_k_ext_and_add_ambient(self):
        """make external "k_ext" matrix and selectively add conductivity to boundary condition "ambient"
        to diagonal elements

        """
        if self.num_nodes > 0:  # c-1 matrix and rank has to be defined
            self.k_ext_mat = np.zeros((self.num_nodes, self.num_nodes))  # initialize with zeros
            for c in self.ambient.connected_to:
                idx = self.tag_list.index(c[0])
                cond = c[1]
                self.k_ext_mat[idx, idx] += cond
                logging.info(f" ambient connected to node '{self.nodes[idx].label}'")
            logging.info(f" k_ext matrix: \n {self.k_ext_mat}")

    def generate_edges(self):
        c = self.conductivity * self.A_base / self.layer_height
        self.edge_list = [[a, b, c] for a, b in zip(self.tag_list, self.tag_list[1:])]

    def generate_ambient(self):
        c_mid = self.U_wall * self.A_wall_layer
        c_end = self.U_wall * (self.A_wall_layer + self.A_base)
        self.ambient = FixedNode(label='indoor',
                                 temp=20.0,
                                 connected_to=[[a, c_mid] for a in self.tag_list])
        for c in self.ambient.connected_to:
            if c[0] == self.tag_list[0] or c[0] == self.tag_list[-1]:
                c[1] = c_end
        # self.ambient.connected_to = [[i, j if (self.tag_list[0] < i < self.tag_list[-1]) else c_end] for i, j in b2.ambient.connected_to]


if __name__ == "__main__":
    from pathlib import Path
    CONFIGDIR = Path(__file__).parent.parent.parent.parent.absolute()
    param = load_config(str(CONFIGDIR / "for_heat_pump_NTA8800_with_buffervessel_nodes_edges.yaml"))
    b0 = StratifiedBufferNew()
    b1 = StratifiedBufferNew.from_dict(param["Buffer"])

    b2 = StratifiedBufferNew(name="MyBuffer", num_layers=8)
    b2.generate_nodes()
    b2.fill_c_inv()
    b2.generate_edges()
    b2.generate_ambient()
    b2.make_k_ext_and_add_ambient()

    print()

    """

    from pathlib import Path
    CONFIGDIR = Path(__file__).parent.parent.parent.parent.absolute()
    param = load_config(str(CONFIGDIR / "for_2R2Chouse_buffer.yaml"))

    b2.boundaries_from_dict(param['boundaries'])
    b2.make_k_ext_and_add_ambient()

    b = StratifiedBuffer("MyBuffer")
    c_list = [1.0, 2.0]
    c1 = make_c_inv_matrix(c_list)
    print(c1, "\n")
    k_list = [[0, 1, 1.0]]
    """
