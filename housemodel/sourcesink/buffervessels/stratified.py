import numpy as np
from scipy.integrate import solve_ivp  # ODE solver

from housemodel.basics.ckf_tools import make_c_inv_matrix
from housemodel.tools.new_configurator import load_config
from housemodel.basics.components import (CapacityNode, FixedNode)
from housemodel.basics.totalsystem import TotalSystem
from housemodel.basics.flows import Flow

import matplotlib
import matplotlib.pyplot as plt
import logging

matplotlib.use('Qt5Agg')

# logging.basicConfig(level="DEBUG")
logging.basicConfig(level="INFO")

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)


class StratifiedBufferNew:
    """parent class for cylindrical stratified buffer vessel

    """

    def __init__(self, name="DefaultBuffer", begin_tag=0, num_layers=5,
                 volume=0.1, height=1.0, U_wall=0.12, T_amb=18.0, T_ini=20):
        self.name = name
        self.begin_tag = begin_tag   # anchor point of buffer vessel in house model
        self.num_nodes = num_layers
        self.volume = volume
        self.height = height
        self.U_wall = U_wall         # conductivity vessel wall to ambient in [W/K m^2]
        self.T_amb = T_amb
        self.T_ini = T_ini

        self.end_node = self.begin_tag + num_layers - 1  # anchor point of cold water return from house model
        self.nodes = []
        self.edges = []
        self.num_edges = 0
        self.boundaries = []
        self.ambient = None

        self.tag_list = []
        self.cap_list = []
        self.edge_list = []
        self.cond_list = []

        self.c_inv_mat = None
        self.k_int_mat = None
        self.k_ext_mat = None

        self.q_vec = None  # np.zeros(self.num_nodes, 1)
        self.f_mat = None  # np.zeros(self.num_nodes, self.num_nodes)

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
        logging.info(f" StratifiedBufferNew object {self.name} created")

    @classmethod
    def from_dict(cls, d):
        """ classmethod to enable constructing an instance from configuration file.
        """
        return cls(name=d["name"], begin_tag=d["begin_tag"], num_layers=d["num_layers"],
                   volume=d["volume"], height=d["height"],
                   U_wall=d["U_wall"], T_amb=d["T_amb"], T_ini=d["T_ini"])

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
                                 temp=self.T_amb,
                                 connected_to=[[a, c_mid] for a in self.tag_list])
        for c in self.ambient.connected_to:
            if c[0] == self.tag_list[0] or c[0] == self.tag_list[-1]:
                c[1] = c_end
        # self.ambient.connected_to = [[i, j if (self.tag_list[0] < i < self.tag_list[-1]) else c_end] for i, j in b2.ambient.connected_to]


def model(t, x, tot_sys):
    """model function for scipy.integrate.solve_ivp.

    Args:
        t:                (array):   time
        x:                (float):   row vector with temperature nodes
        tot_sys:          (object):  model system

    Returns:
        (list): vector elements of dx/dt
    """

    # Conversion of 1D array to a 2D array
    # https://stackoverflow.com/questions/5954603/transposing-a-1d-numpy-array
    x = np.array(x)[np.newaxis]

    dTdt = (-tot_sys.k_mat @ x.T) - (tot_sys.f_mat @ x.T) + tot_sys.q_vector
    dTdt = np.dot(tot_sys.c_inv_mat, dTdt)

    return dTdt.flatten().tolist()


if __name__ == "__main__":
    from pathlib import Path
    CONFIGDIR = Path(__file__).parent.parent.parent.parent.absolute()
    param = load_config(str(CONFIGDIR / "for_heat_pump_NTA8800_with_buffervessel_nodes_edges.yaml"))
    # param = load_config(str(CONFIGDIR / "for_2R2Chouse_buffer.yaml"))

    b0 = StratifiedBufferNew()
    b1 = StratifiedBufferNew.from_dict(param["Buffer"])

    b2 = StratifiedBufferNew(name="MyBuffer", volume=5,
                             height=2.5, num_layers=8,
                             T_amb=20, T_ini=80)
    Tsupply = 62.5
    Treturn = 50
    supply_flow = 0      # m^3/s
    demand_flow = 0.001  # m^3/s = 1 kg/s

    leak_to_amb_top = b2.U_wall*(b2.A_base+b2.A_wall_layer)
    leak_to_amb_mid = b2.U_wall*b2.A_wall_layer
    layer_to_layer = b2.A_base*b2.conductivity/b2.layer_height
    print(f"{leak_to_amb_top}, {leak_to_amb_mid}, {layer_to_layer}, {b2.cap_layer}")

    b2.generate_nodes()
    b2.fill_c_inv()
    b2.generate_edges()
    b2.generate_ambient()
    b2.make_k_ext_and_add_ambient()

    total = TotalSystem("Buffer", [b2])
    total.sort_parts()
    # compose c-1-matrix from parts and merge tag_lists
    total.merge_c_inv()
    total.merge_tag_lists()

    # compose k-matrix from edges
    total.edges_between_from_dict(param["edges"])
    total.merge_edge_lists_from_parts_and_between()

    total.fill_k(total.edge_list)
    total.merge_k_ext()
    total.k_mat += total.k_ext_mat
    total.merge_ambients()  # assignment by reference, no copy!
    total.make_empty_q_vec()
    # logger.info(f" \n\n {total.c_inv_mat} \n\n {total.k_mat}, \n\n {total.q_vec} \n")

    # calculate flow matrices and combine into f_mat_all
    if total.flows:
        total.flows = []
    total.flows.append(Flow("Supply", flow_rate=0.001,
                            node_list=[0, 1, 2, 3, 4, 5, 6, 7, 0]))
    total.flows.append(Flow("Demand", flow_rate=0.001,
                            node_list=[7, 6, 5, 4, 3, 2, 1, 0, 7]))
    for f in total.flows:
        f.make_df_matrix(rank=total.k_mat.shape[0])
        print(f"{f.flow_rate*f.density*f.cp}")

    initial_condition = np.ones(b2.num_nodes) * b2.T_ini
    inputs = (total,)
    result = solve_ivp(model, [0, 3600 * 2], initial_condition, args=inputs)

    plt.figure(figsize=(10, 5))
    for i in range(len(result.y)):
        plt.plot(result.t, result.y[i, :], label=f'$T_{i + 1}$')
    plt.legend(loc='best')
    plt.title("Stratified Buffervessel Simulation")
    plt.show()

    print()
