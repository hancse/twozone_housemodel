import numpy as np
from scipy.linalg import block_diag
import itertools

from housemodel.tools.new_configurator import load_config

from housemodel.basics.ckf_tools import (make_c_inv_matrix,
                                         add_c_inv_block,
                                         make_edges)

from housemodel.basics.components import (FixedNode,
                                          CondEdge)

from housemodel.buildings.building import Building
from housemodel.basics.flows import Flow
from housemodel.sourcesink.radiators.linear_radiator import LinearRadiator

import logging

# logging.basicConfig(level="DEBUG")
logging.basicConfig(level="INFO")
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

# input parameter parts contains the subsystems of TotalSystem
# like Building, StratifiedBuffer or LinearRadiator objects
# if parts is a tuple, the variable is immutable, and can have a default value parts=()
# if parts is a list, it is mutable (sortable)
# then it must have a default value of parts=None and a default statement
# if parts is None: parts = []
# see: https://stackoverflow.com/questions/61260600/avoiding-default-argument-value-is-mutable-warning-pycharm


class TotalSystem:
    def __init__(self, name="", parts=None):   # immutable "list" = tuple
        self.name = name
        if parts is None:
            self.parts = []
        else:
            self.parts = parts
        self.num_nodes = 0
        self.num_edges = 0
        self.nodes = []            # np.zeros(self.num_nodes, dtype=object)
        self.edges = []            # np.zeros(self.num_nodes - 1)
        # self.boundaries = []
        self.ambients = None
        self.flows = []

        self.c_inv_mat = None  # np.zeros((self.num_nodes, self.num_nodes))
        self.k_mat = None      # np.zeros_like(self.c_inv_mat)
        self.k_ext_mat = None  # np.zeros_like(self.c_inv_mat)
        self.q_vec = None      # np.zeros(self.num_nodes, 1)
        self.f_mat = None      # np.zeros(self.num_nodes, self.num_nodes)

        self.q_solar = None
        self.q_int = None

        self.tag_list = []
        # self.cap_list = []
        # self.cond_list =
        self.edge_list = []
        self.edge_list_between_parts = []

        logging.info(f" TotalSystem object {self.name} created")

    def edges_from_dict(self, lol):
        """reads ALL edges from parameter dict

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

    def edges_between_from_dict(self, lol):
        """reads ONLY edges BETWEEN parts from parameter dict.
        """
        if self.edge_list_between_parts or self.edges:
            self.edges = []
            self.edge_list_between_parts = []
        if not lol or len(lol) <= 0:
            logging.info(f" no edges found between parts")
            return

        self.num_edges = len(lol)
        for n in range(self.num_edges):
            edge = CondEdge(label="",
                            conn_nodes=[lol[n][0], lol[n][1]],
                            cond=lol[n][2])
            self.edges.append(edge)
            self.edge_list_between_parts.append(lol[n])
            logging.info(f" edge from {edge.conn_nodes[0]} to {edge.conn_nodes[1] } appended to {self.name}")

    def edges_between_from_yaml(self, config_yaml: str):
        """reads ONLY edges BETWEEN parts from config yaml file.
        """
        lol = load_config(config_yaml).get("edges")
        if self.edge_list_between_parts or self.edges:
            self.edges = []
            self.edge_list_between_parts = []
        if not lol or len(lol) <= 0:
            logging.info(f" no edges found between parts")
            return

        self.num_edges = len(lol)
        for n in range(self.num_edges):
            edge = CondEdge(label="",
                            conn_nodes=[lol[n][0], lol[n][1]],
                            cond=lol[n][2])
            self.edges.append(edge)
            self.edge_list_between_parts.append(lol[n])
            logging.info(f" edge from {edge.conn_nodes[0]} to {edge.conn_nodes[1] } appended to {self.name}")

    def flows_from_yaml(self, config_yaml: str):
        """ method to enable constructing an instance from yaml file.
        """
        if self.flows:
            self.flows = []
        d = load_config(config_yaml)
        fd = d.get("flows")
        for n in range(len(fd)):
            self.flows.append(Flow.from_dict(fd[n]))
            self.flows[n].make_df_matrix(rank=self.k_mat.shape[0])

    def combine_flows(self):
        # combine F-matrices into matrix total.f_mat
        self.f_mat = np.zeros_like(self.flows[0].df_mat)
        for n in range(len(self.flows)):
            self.f_mat += np.multiply(self.flows[n].df_mat,
                                      self.flows[n].heat_rate)

        # remove matrix elements > 0 from Fall
        self.f_mat = np.where(self.f_mat <= 0, self.f_mat, 0)

        # create diagonal elements in Fall, so that som over each row is zero
        row_sums = np.sum(self.f_mat, axis=1).tolist()
        self.f_mat = self.f_mat - np.diag(np.array(row_sums), k=0)

    def fill_k(self, lol):
        """select global edges belonging to total system and make k-matrix.

        Args:
            lol: list of edge lists [from, to, weight]
        """
        # selection is always necessary!
        el = [e for e in lol if e[0] in self.tag_list and e[1] in self.tag_list]
        self.k_mat = make_edges(el)
        logging.info(f" k_matrix: \n {self.k_mat}")

    def complete_k(self, lol):
        """add global edges BETWEEN subsystems to complete k-matrix.

        Args:
            lol: list of edge lists [from, to, weight] read from config file
        """
        # selection should not be necessary
        el = [e for e in lol if e[0] in self.tag_list and e[1] in self.tag_list]
        self.k_mat = make_edges(el)
        logging.info(f" k_matrix: \n {self.k_mat}")

    def boundaries_from_dict(self, lod):
        for n in range(len(lod)):
            node = FixedNode(label=lod[n]["label"],
                             temp=lod[n]["T_ini"],
                             connected_to=lod[n]["connected_to"])
            # append by reference, therefore new node object in each iteration
            self.boundaries.append(node)
            logging.info(f" boundary '{node.label}' appended to {self.name}")
        # for total system, "ambient" is "outdoor"
        self.ambient = [fn for fn in self.boundaries if fn.label == "outdoor"][0]
        logging.info(f" ambient is '{self.ambient.label}' for {self.name}")

    def add_fixed_to_k(self):
        """add conductivities to boundary "ambient" to diagonal elements of k-matrix.

        """
        # fnl = [fn for fn in self.boundaries for index in fn.connected_to if index[0] in self.tag_list]
        # res = []
        # [res.append(x) for x in fnl if x not in res]
        for c in self.ambient.connected_to:
            index = c[0]
            cond = c[1]
            self.k_mat[index, index] -= cond
            logging.info(f" ambient connected to node '{self.nodes[index].label}'")

    def make_empty_q_vec(self):
        self.q_vec = np.zeros((self.num_nodes, 1))
        # logging.info(f" empty q-vector created of rank {self.num_nodes}")

    # def make_empty_f_mat(self):
    #    self.f_mat = np.zeros((self.num_nodes, 1))
    #    # logging.info(f" empty q-vector created of rank {self.num_nodes}")

    def add_fixed_to_q(self):
        """add terms from ALL boundary conditions (external nodes) like T_outdoor and T_indoor.

            - loops over ALL FixedNode object in self.boundaries
            - for each FixedNode adds T/Rth to the corresponding element of self.q_vec
            - the right element is found via the index of the tag in self.taglist
        """
        for b in self.boundaries:
            for c in b.connected_to:
                idx = self.tag_list.index(c[0])
                cond = c[1]
                self.q_vec[idx] += cond * b.temp
                logging.info(f" ambient added to q-vector element {idx}")
        logging.info(f" q_vector: \n {self.q_vec}")

    def add_ambient_to_q(self):
        """selectively add terms "ambient temperature*conductivity_to_ambient" to q-vector.

        TotalSystem needs this function to update the q_vector in the solver

        Note: using list comprehension:
            q = [c for conn in [p.ambient.connected_to for p in tot_sys.parts if p.ambient is not None] for c in conn]
        """
        for p in self.parts:
            if p.ambient is not None:
                for c in p.ambient.connected_to:
                    idx = self.tag_list.index(c[0])
                    cond = c[1]
                    new_power = cond * p.ambient.temp
                    self.q_vec[idx] += new_power
                    # logging.info(f" {p.ambient.label} ({new_power}) added to q-vector element {idx}")
        # logging.info(f" q_vector: \n {self.q_vec}")

    def myFunc(self, p):
        return p.tag_list[0]

    def sort_parts(self):
        """sorting parts according to the first element in their attribute 'tag_list'.
        """
        if type(self.parts) is list:
            self.parts.sort(key=self.myFunc)
        elif type(self.parts) is tuple:
            part_list = list(self.parts)
            part_list.sort(key=self.myFunc)
            self.parts = tuple(part_list)

    def merge_c_inv(self):
        """merge inverse capacity matrices of parts by block diagonal addition
        """
        my_tup = (p.c_inv_mat for p in self.parts)
        self.c_inv_mat = block_diag(*my_tup)
        self.num_nodes = np.shape(self.c_inv_mat)[0]
        assert self.num_nodes == sum([p.num_nodes for p in self.parts]), "total # nodes incorrect"

    def merge_tag_lists(self):
        """merge tag_lists from parts. This should always yield a sorted tag_list
        since the parts are already sorted on parts.tag_list[0]
        See: https://www.geeksforgeeks.org/python-ways-to-concatenate-two-lists
        See: https://stackoverflow.com/questions/18114415/how-do-i-concatenate-3-lists-using-a-list-comprehension
        """
        self.tag_list = [t for tag in [p.tag_list for p in self.parts] for t in tag]

    def merge_edge_lists(self, lol):
        """merge internal edge lists from parts and edge list read from config .yaml file.
        since the parts are already sorted on parts.tag_list[0]
        See: https://www.geeksforgeeks.org/python-ways-to-concatenate-two-lists
        See: https://stackoverflow.com/questions/18114415/how-do-i-concatenate-3-lists-using-a-list-comprehension
        See Also: https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
        See Also: https://stackoverflow.com/questions/1077015/how-can-i-get-a-flat-result-from-a-list-comprehension-instead-of-a-nested-list
        """
        self.edge_list = [item for sublist in lol for item in sublist]

    def merge_edge_lists1(self, lol):
        flat_list = []
        for sublist in lol:
            for item in sublist:
                flat_list.append(item)
        self.edge_list = flat_list

    def merge_edge_lists2(self, lol):
        lol = [[1, 2, 3], [4, 5, 6], [7], [8, 9]]
        merged = list(itertools.chain.from_iterable(lol))
        self.edge_list = merged

    def merge_edge_lists_from_parts_and_between(self):
        merged = [p.edge_list for p in self.parts]
        merged.append(self.edge_list_between_parts)
        self.merge_edge_lists(merged)

    def merge_k_ext(self):
        """merge external conductivity matrices of parts by block diagonal addition
        """
        my_tup = (p.k_ext_mat for p in self.parts)
        self.k_ext_mat = block_diag(*my_tup)

    def merge_ambients(self):
        self.ambients = [p.ambient for p in self.parts if p.ambient is not None]

    def add_source_to_q(self, source, src_index: int):
        for c in source.connected_to:
            idx = self.tag_list.index(c[0])
            fraction = c[1]
            new_power = fraction * source.values[src_index]
            self.q_vec[idx] += new_power
            # logging.info(f" source {source.name}[{src_index}] ({new_power}) added to q-vector element {idx}")

    def find_node_label_from_tag(self, tag):
        for p in self.parts:
            for n in p.nodes:
                if n.tag == tag:
                    return n.label

    def find_tag_from_node_label(self, label):
        for p in self.parts:
            for n in p.nodes:
                if n.label == label:
                    return n.tag


if __name__ == "__main__":
    house = Building()
    house.tag_list = [0, 1]
    radiator = LinearRadiator()
    radiator.tag_list = [2]
    t = TotalSystem("Test", [radiator, house])
    logging.info(f"parts (unsorted): {t.parts}")
    t.sort_parts()
    logging.info(f"parts (sorted): {t.parts}")
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
