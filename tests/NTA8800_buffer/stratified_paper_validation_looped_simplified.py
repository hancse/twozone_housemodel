import numpy as np
from scipy.integrate import solve_ivp  # ODE solver

from housemodel.basics.totalsystem import TotalSystem
from housemodel.basics.flows import Flow
from housemodel.sourcesink.radiators.radiators import Radiator
from housemodel.sourcesink.buffervessels.stratified import model, StratifiedBufferNew

import matplotlib
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level="INFO")
matplotlib.use('Qt5Agg')
# from pathlib import Path
# CONFIGDIR = Path(__file__).parent.absolute()

b = StratifiedBufferNew(name="MyBuffer", num_layers=10,
                        volume=200, height=4,
                        T_amb=10, T_ini=80)
print(f"{b.radius}")
leak_to_amb_top = b.U_wall * (b.A_base + b.A_wall_layer)
leak_to_amb_mid = b.U_wall * b.A_wall_layer
layer_to_layer = b.A_base * b.conductivity / b.layer_height
print(f"{leak_to_amb_top}, {leak_to_amb_mid}, {layer_to_layer}, {b.cap_layer}")

b.generate_nodes()
b.generate_edges()
b.generate_ambient()
b.fill_c_inv()
b.make_k_ext_and_add_ambient()

total = TotalSystem("Buffer", [b])
total.sort_parts()
# compose c-1-matrix from parts and merge tag_lists
total.merge_c_inv()
total.merge_tag_lists()

# compose k-matrix from edges
total.merge_edge_lists_from_parts_and_between()

total.fill_k(total.edge_list)
total.merge_k_ext()
total.k_mat += total.k_ext_mat
total.merge_ambients()  # assignment by reference, no copy!
total.make_empty_q_vec()
total.add_ambient_to_q()
logging.info(f" \n\n {total.c_inv_mat} \n\n {total.k_mat}, \n\n {total.q_vec} \n")

Tsupply = 80
Treturn = 40
supply_flow_rate = 0  # m^3/s
demand_flow_rate = 1.0e-3  # m^3/s

# calculate flow matrices and combine into f_mat_all
if total.flows:
    total.flows = []
total.flows.append(Flow("Supply", flow_rate=supply_flow_rate,
                        node_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
total.flows.append(Flow("Demand", flow_rate=demand_flow_rate,
                        node_list=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]))
for f in total.flows:
    f.make_df_matrix(rank=total.k_mat.shape[0])
    # print(f"{f.flow_rate * f.density * f.cp}")

initial_condition = []

control_interval = 600
t = np.arange(0, 3600 * 25, control_interval)
output = np.zeros((b.num_nodes, len(t)))
output[:, 0] = np.ones(b.num_nodes) * b.T_ini  # approximation

# Radiator object
deg = u"\u00b0"  # degree sign
r = Radiator(name="Rad", exp_rad=1.3)
r.q_zero = 100000  # W
r.T_supply = output[0, 0]
r.T_amb = 20
r.T_return = (r.T_supply + r.T_amb) / 2.0
r.set_flow(total.flows[1])  # demand flow
print(f"Heat rate: {r.flow.heat_rate} [W/K] \n")

r.update(r.func_rad_lmtd)
T_ret = np.ones(len(t)) * r.T_return

print(f"Q_dot: {r.q_dot} [W], T_return: {r.T_return} {deg}C")
print(f"radiator to room: {r.flow.heat_rate * (output[0, 0] - r.T_return)} [W]")
print(f"radiator to room: {r.Km * np.power(r.get_lmtd(), r.exp_rad)} [W] with Delta T_LMTD = {r.get_lmtd()}")
print(f"top-bottom: {r.flow.heat_rate * (output[0, 0] - output[9, 0])} [W]")
print(f"back-to-bottom: {r.flow.heat_rate * (r.T_return - output[9, 0])} [W]")

inputs = (total,)

for i in range(len(t) - 1):
    if t[i] < 3600 * 11:
        total.make_empty_q_vec()
        total.add_ambient_to_q()
        total.combine_flows()

        # MvdB
        # total.q_vec[9] += total.flows[1].heat_rate * Treturn
        # total.f_mat[9, 9] += total.flows[1].heat_rate

        # new
        r.T_supply = output[0, i]
        r.update(r.func_rad_lmtd)
        total.q_vec[9] += r.flow.heat_rate * r.T_return
        total.f_mat[9, 9] += r.flow.heat_rate

    elif t[i] < 3600 * 15:
        # MvdB
        # electric heater 2.4 MW; heat transfer controlled by flow
        # flow becomes VERY large if bottom layer approaches 80 C!
        # unstable system
        # supply_flow_rate = 2.4e3/((80 - output[9, i])*4190)
        # demand_flow_rate = 0

        # new
        supply_flow_rate = 1.0e-3  # m^3/s
        demand_flow_rate = 0

        total.flows[0].set_flow_rate(supply_flow_rate)
        total.flows[1].set_flow_rate(demand_flow_rate)

        total.make_empty_q_vec()
        total.add_ambient_to_q()
        total.combine_flows()

        # MvdB
        # total.q_vec[0] += total.flows[0].heat_rate * Tsupply
        # total.f_mat[0, 0] += total.flows[0].heat_rate

        # new
        r.T_supply = output[0, i]
        r.update(r.func_rad_lmtd)
        total.q_vec[0] += total.flows[0].heat_rate * Tsupply
        total.f_mat[0, 0] += total.flows[0].heat_rate
        # total.q_vec[9] += r.flow.heat_rate * r.T_return
        # total.f_mat[9, 9] += r.flow.heat_rate

    elif t[i] < 3600 * 17.5:
        # MvdB
        supply_flow_rate = 0
        demand_flow_rate = 1.0e-3  # m^3/s
        total.flows[0].set_flow_rate(supply_flow_rate)
        total.flows[1].set_flow_rate(demand_flow_rate)

        total.make_empty_q_vec()
        total.add_ambient_to_q()
        total.combine_flows()

        # MvdB
        # total.q_vec[0] += total.flows[0].heat_rate * Tsupply
        # total.f_mat[0, 0] += total.flows[0].heat_rate
        # total.q_vec[9] += total.flows[1].heat_rate * Treturn
        # total.f_mat[9, 9] += total.flows[1].heat_rate

        # new
        r.T_supply = output[0, i]
        r.update(r.func_rad_lmtd)
        total.q_vec[0] += total.flows[0].heat_rate * Tsupply
        total.f_mat[0, 0] += total.flows[0].heat_rate
        total.q_vec[9] += r.flow.heat_rate * r.T_return
        total.f_mat[9, 9] += r.flow.heat_rate

    elif t[i] < (3600 * 25):
        # MvdB
        # supply_flow = 2.4e3/((80 - output[9, i])*4190)
        # demand_flow = 0.01

        # new
        supply_flow_rate = 0.50e-3  # m^3/s
        demand_flow_rate = 1.0e-3
        total.flows[0].set_flow_rate(supply_flow_rate)
        total.flows[1].set_flow_rate(demand_flow_rate)

        total.make_empty_q_vec()
        total.add_ambient_to_q()
        total.combine_flows()

        # MvdB
        # total.q_vec[0] += total.flows[0].heat_rate * Tsupply
        # total.f_mat[0, 0] += total.flows[0].heat_rate
        # total.q_vec[9] += total.flows[1].heat_rate * Treturn
        # total.f_mat[9, 9] += total.flows[1].heat_rate

        # new
        r.T_supply = output[0, i]
        r.update(r.func_rad_lmtd)
        total.q_vec[0] += total.flows[0].heat_rate * Tsupply
        total.f_mat[0, 0] += total.flows[0].heat_rate
        total.q_vec[9] += r.flow.heat_rate * r.T_return
        total.f_mat[9, 9] += r.flow.heat_rate

    if np.any(initial_condition):
        initial_condition = output[:, i]
    else:
        initial_condition = np.ones(b.num_nodes) * b.T_ini

    # inputs = (total,)  # here or just before time loop
    ts = [t[i], t[i + 1]]
    result = solve_ivp(model, ts, initial_condition,
                       method='RK45', args=inputs,
                       first_step=control_interval)

    output[:, i + 1] = np.transpose(result.y[:, -1])
    T_ret[i + 1] = r.T_return

plt.figure(figsize=(10, 5))
for i in range(len(output)):
    plt.plot(t / 3600, output[i, :], label=f'$T_{i}$')
plt.plot(t / 3600, T_ret, label='r.T_return', color='r')
# plt.ylim(20, 90)
plt.legend(loc='best')
# plt.xlim(0, 24)
# plt.ylim(40, 100)
plt.title("Stratified Buffervessel Validation")
plt.show()
