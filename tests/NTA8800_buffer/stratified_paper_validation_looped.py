import numpy as np
from scipy.integrate import solve_ivp  # ODE solver

from housemodel.basics.ckf_tools import make_c_inv_matrix
from housemodel.tools.new_configurator import load_config
from housemodel.basics.components import (CapacityNode, FixedNode)
from housemodel.basics.totalsystem import TotalSystem
from housemodel.basics.flows import Flow

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level="INFO")

from housemodel.sourcesink.buffervessels.stratified import model, StratifiedBufferNew

from pathlib import Path

CONFIGDIR = Path(__file__).parent.absolute()
param = load_config(str(CONFIGDIR / "for_heat_pump_NTA8800_with_buffervessel_nodes_edges.yaml"))

b = StratifiedBufferNew(name="MyBuffer", volume=200,
                         height=4, num_layers=10,
                         T_amb=10, T_ini=80)
Tsupply = 80
Treturn = 40
supply_flow = 0  # m^3/s
demand_flow = 0.010  # m^3/s

b.calculate_buffer_properties()
print(f"{b.radius}")

leak_to_amb_top = b.U_wall*(b.A_base+b.A_wall_layer)
leak_to_amb_mid = b.U_wall*b.A_wall_layer
layer_to_layer = b.A_base*b.conductivity/b.layer_height
print(f"{leak_to_amb_top}, {leak_to_amb_mid}, {layer_to_layer}, {b.cap_layer}")

b.generate_nodes()
b.fill_c_inv()
b.generate_edges()
b.generate_ambient()
b.make_k_ext_and_add_ambient()

total = TotalSystem("Buffer", [b])
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
total.add_ambient_to_q()

# calculate flow matrices and combine into f_mat_all
if total.flows:
    total.flows = []
total.flows.append(Flow("Supply", flow_rate=supply_flow,
                        node_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
total.flows.append(Flow("Demand", flow_rate=demand_flow,
                        node_list=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]))

for f in total.flows:
    f.make_df_matrix(rank=total.k_mat.shape[0])
    print(f"{f.flow_rate * f.density * f.cp}")

# combine F-matrices into matrix total.f_mat
total.f_mat = np.zeros_like(total.flows[0].df_mat)
for n in range(len(total.flows)):
    total.f_mat += np.multiply(total.flows[n].df_mat, total.flows[n].heat_rate)

# remove matrix elements > 0 from total.f_mat
total.f_mat = np.where(total.f_mat <= 0, total.f_mat, 0)

# create diagonal elements in total.f_mat, so that som over each row is zero
row_sums = np.sum(total.f_mat, axis=1).tolist()
total.f_mat = total.f_mat - np.diag(np.array(row_sums), k=0)

total.q_vec[9] += total.flows[1].heat_rate * Treturn
total.f_mat[9, 9] += total.flows[1].heat_rate

initial_condition = []
#inputs = (total,)
#result = solve_ivp(model, [0, 3600 * 12], initial_condition, args=inputs)

t = np.arange(0, 3600*25, 600)
output = np.zeros((b.num_nodes, len(t)))
output[:,0] = np.ones(b.num_nodes) * b.T_ini
for i in range(len(t) - 1):
    if t[i]<3600*11:
        total.make_empty_q_vec()
        total.add_ambient_to_q()
        # total.make_empty_f_mat()
        # calculate flow matrices and combine into f_mat_all
        if total.flows:
            total.flows = []
        total.flows.append(Flow("Supply", flow_rate=supply_flow,
                                node_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        total.flows.append(Flow("Demand", flow_rate=demand_flow,
                                node_list=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]))

        for f in total.flows:
            f.make_df_matrix(rank=total.k_mat.shape[0])
            print(f"{f.flow_rate * f.density * f.cp}")

        # combine F-matrices into matrix total.f_mat
        total.f_mat = np.zeros_like(total.flows[0].df_mat)
        for n in range(len(total.flows)):
            total.f_mat += np.multiply(total.flows[n].df_mat, total.flows[n].heat_rate)

        # remove matrix elements > 0 from total.f_mat
        total.f_mat = np.where(total.f_mat <= 0, total.f_mat, 0)

        # create diagonal elements in total.f_mat, so that som over each row is zero
        row_sums = np.sum(total.f_mat, axis=1).tolist()
        total.f_mat = total.f_mat - np.diag(np.array(row_sums), k=0)

        total.q_vec[9] += total.flows[1].heat_rate * Treturn
        total.f_mat[9, 9] += total.flows[1].heat_rate
    elif t[i]<3600*15:
        total.make_empty_q_vec()
        total.add_ambient_to_q()
        # total.make_empty_f_mat()
        supply_flow = 2.4e3/((80 - output[9, i])*4190)
        demand_flow = 0

        # calculate flow matrices and combine into f_mat_all
        if total.flows:
            total.flows = []
        total.flows.append(Flow("Supply", flow_rate=supply_flow,
                                node_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        total.flows.append(Flow("Demand", flow_rate=demand_flow,
                                node_list=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]))

        for f in total.flows:
            f.make_df_matrix(rank=total.k_mat.shape[0])
            print(f"{f.flow_rate * f.density * f.cp}")

        # combine F-matrices into matrix total.f_mat
        total.f_mat = np.zeros_like(total.flows[0].df_mat)
        for n in range(len(total.flows)):
            total.f_mat += np.multiply(total.flows[n].df_mat, total.flows[n].heat_rate)

        # remove matrix elements > 0 from total.f_mat
        total.f_mat = np.where(total.f_mat <= 0, total.f_mat, 0)

        # create diagonal elements in total.f_mat, so that som over each row is zero
        row_sums = np.sum(total.f_mat, axis=1).tolist()
        total.f_mat = total.f_mat - np.diag(np.array(row_sums), k=0)

        total.q_vec[0] += total.flows[0].heat_rate * Tsupply
        total.f_mat[0, 0] += total.flows[0].heat_rate

    elif t[i] < 3600 * 17.5:
        total.make_empty_q_vec()
        total.add_ambient_to_q()
        # total.make_empty_f_mat()
        supply_flow = 0
        demand_flow = 0.01

        # calculate flow matrices and combine into f_mat_all
        if total.flows:
            total.flows = []
        total.flows.append(Flow("Supply", flow_rate=supply_flow,
                                node_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        total.flows.append(Flow("Demand", flow_rate=demand_flow,
                                node_list=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]))

        for f in total.flows:
            f.make_df_matrix(rank=total.k_mat.shape[0])
            print(f"{f.flow_rate * f.density * f.cp}")

        # combine F-matrices into matrix total.f_mat
        total.f_mat = np.zeros_like(total.flows[0].df_mat)
        for n in range(len(total.flows)):
            total.f_mat += np.multiply(total.flows[n].df_mat, total.flows[n].heat_rate)

        # remove matrix elements > 0 from total.f_mat
        total.f_mat = np.where(total.f_mat <= 0, total.f_mat, 0)

        # create diagonal elements in total.f_mat, so that som over each row is zero
        row_sums = np.sum(total.f_mat, axis=1).tolist()
        total.f_mat = total.f_mat - np.diag(np.array(row_sums), k=0)

        total.q_vec[0] += total.flows[0].heat_rate * Tsupply
        total.f_mat[0, 0] += total.flows[0].heat_rate

        total.q_vec[9] += total.flows[1].heat_rate * Treturn
        total.f_mat[9, 9] += total.flows[1].heat_rate

    elif t[i] < (3600 * 25):
        total.make_empty_q_vec()
        total.add_ambient_to_q()
        # total.make_empty_f_mat()
        supply_flow = 2.4e3/((80 - output[9, i])*4190)
        demand_flow = 0.01

        # calculate flow matrices and combine into f_mat_all
        if total.flows:
            total.flows = []
        total.flows.append(Flow("Supply", flow_rate=supply_flow,
                                node_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        total.flows.append(Flow("Demand", flow_rate=demand_flow,
                                node_list=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]))

        for f in total.flows:
            f.make_df_matrix(rank=total.k_mat.shape[0])
            print(f"{f.flow_rate * f.density * f.cp}")

        # combine F-matrices into matrix total.f_mat
        total.f_mat = np.zeros_like(total.flows[0].df_mat)
        for n in range(len(total.flows)):
            total.f_mat += np.multiply(total.flows[n].df_mat, total.flows[n].heat_rate)

        # remove matrix elements > 0 from total.f_mat
        total.f_mat = np.where(total.f_mat <= 0, total.f_mat, 0)

        # create diagonal elements in total.f_mat, so that som over each row is zero
        row_sums = np.sum(total.f_mat, axis=1).tolist()
        total.f_mat = total.f_mat - np.diag(np.array(row_sums), k=0)

        total.q_vec[0] += total.flows[0].heat_rate * Tsupply
        total.f_mat[0, 0] += total.flows[0].heat_rate

        total.q_vec[9] += total.flows[1].heat_rate * Treturn
        total.f_mat[9, 9] += total.flows[1].heat_rate

    if np.any(initial_condition):
        initial_condition = output[:, i]
    else:
        initial_condition = np.ones(b.num_nodes) * b.T_ini

    if np.any(initial_condition):
        initial_condition = output[:, i]
    else:
        initial_condition = np.ones(b.num_nodes) * b.T_ini

    inputs = (total,)
    ts = [t[i], t[i + 1]]
    result = solve_ivp(model, ts, initial_condition, args=inputs)

    #result = solve_ivp(model_radiator_m, ts, y0,
    #                  method='RK45', args=inputs,
    #                   first_step=control_interval)

    #Tair[i + 1] = result.y[0, -1]
    #Twall[i + 1] = result.y[1, -1]
    #Tradiator[i + 1] = result.y[2, -1]

    #b.T_ini = result.y[:, -1]
    output[:, i + 1] = np.transpose(result.y[:, -1])

plt.figure(figsize=(10, 5))
for i in range(len(output)):
    plt.plot(t/3600, output[i, :], label=f'$T_{i}$')
plt.legend(loc='best')
plt.xlim(0, 24)
plt.ylim(40, 100)
plt.title("Stratified Buffervessel Validation")
plt.show()
