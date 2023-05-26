# Example Script - calculates Radiator Return Temperature
# prepared by Hakan İbrahim Tol, PhD

# INPUTS
q = 0.9  # [kW] Actual heat demand rate
Ts = 70   # [°C] Actual radiator supply temperature
Ti = 20   # [°C] Actual indoor (set) temperature

q_o = 1    # [kW] Design (peak) heat demand rate
Ts_o = 75  # [°C] Design radiator supply temperature
Tr_o = 65  # [°C] Design radiator return temperature
Ti_o = 20  # [°C] Design indoor (set) temperature

n = 1.3    # [-]  Empirical radiator constant


from housemodel.tools.radiator_performance import ReturnTemperature as Tr

Tr_AMTD = Tr.Tr_AMTD(q, Ts, Ti, q_o, Ts_o, Tr_o, Ti_o, n)
Tr_GMTD = Tr.Tr_GMTD(q, Ts, Ti, q_o, Ts_o, Tr_o, Ti_o, n)
Tr_LMTD = Tr.Tr_LMTD(q, Ts, Ti, q_o, Ts_o, Tr_o, Ti_o, n)

# Print to Screen as Table
print(f"q/q_0 = {q/q_o}")
table = {'Tr via AMTD': Tr_AMTD, 'Tr via GMTD': Tr_GMTD, 'Tr via LMTD': Tr_LMTD}
for name, value in table.items():
    print(f'{name:10} ==> {value:10f}')

cp = 4190
m_o = (q_o*1000) / (cp*(Ts_o - Tr_o))  # in
print(f"mass flow_zero: {m_o:6f} [kg/s] ({m_o*1000:3f} ml/s)")

m = (q*1000) / (cp*(Ts - Tr_LMTD))  # in
print(f"mass flow_zero: {m:6f} [kg/s] ({m*1000:3f} ml/s)")
